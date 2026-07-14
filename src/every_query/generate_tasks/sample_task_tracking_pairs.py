"""Per-task pos/neg pair sampler for cheap in-training AUROC tracking.

Consumes the dense per-shard label parquets written by ``EQ_generate_evaluation_tasks``
(typically with ``split=tuning``) and, for each ``(query, duration_days)`` task, samples
exactly one row with ``boolean_value=True`` and one with ``boolean_value=False``.  Tasks
missing either class are dropped (AUROC is undefined for a single-class task; mirrors
``evaluate/metrics.py``'s ``_auroc_or_none``).

The output is a single small parquet — two rows per surviving task — meant to be read by
the training-time ``TaskAurocTrackingCallback`` every validation pass.  Because
AUROC for a task equals ``P(score(positive) > score(negative))`` for a random positive/
negative pair, scoring this one pair per task gives an unbiased (if high-variance)
per-task AUROC estimate; macro-averaging the resulting win/tie/loss indicators across
tasks estimates macro AUROC at a cost of ``O(n_tasks)`` forward passes instead of
``O(tuning-split size)``.

Pipeline position:
``EQ_generate_evaluation_tasks(split=tuning) -> EQ_sample_task_tracking_pairs -> EQ_train``.
"""

import logging
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from every_query.data.schema import TaskQuerySchema
from every_query.generate_tasks.sample_tasks import _atomic_write_parquet, _require_path_arg
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure primitive
# ---------------------------------------------------------------------------


def sample_task_tracking_pairs(labels_df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Sample one positive + one negative labeled row per ``(query, duration_days)`` task.

    Args:
        labels_df: ``TaskQuerySchema``-shaped rows with a non-null-filterable
            ``boolean_value`` column (rows with a null ``boolean_value``, i.e. censored,
            are dropped before sampling — the label is undefined there).
        seed: PRNG seed.  Deterministic in ``(labels_df, seed)``.

    Returns:
        A frame with the same columns as ``labels_df``, restricted to exactly two rows
        (one ``True``, one ``False``) per task that had both classes available. Tasks
        with only one class present are dropped entirely (a warning is logged with the
        drop count). Sorted by ``(query, duration_days, boolean_value)``.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3, 4, 5],
        ...     "prediction_time": [datetime(2024, 1, i) for i in range(1, 6)],
        ...     "query": ["A", "A", "A", "B", "B"],
        ...     "duration_days": pl.Series([30.0, 30.0, 30.0, 30.0, 30.0], dtype=pl.Float32),
        ...     "boolean_value": [True, True, False, True, True],
        ... })
        >>> out = sample_task_tracking_pairs(df, seed=0)

        Task ``A`` has both classes, so it survives with exactly 2 rows; task ``B`` is
        all-positive and is dropped entirely:

        >>> sorted(out["query"].unique().to_list())
        ['A']
        >>> out.height
        2
        >>> sorted(out["boolean_value"].to_list())
        [False, True]

        Determinism — same seed, same output:

        >>> a = sample_task_tracking_pairs(df, seed=42)
        >>> b = sample_task_tracking_pairs(df, seed=42)
        >>> a.equals(b)
        True

        Empty input yields an empty frame:

        >>> sample_task_tracking_pairs(df.head(0), seed=0).height
        0
    """
    required = {
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
        TaskQuerySchema.boolean_value_name,
    }
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels_df is missing required column(s) {sorted(missing)}")

    label_col = TaskQuerySchema.boolean_value_name
    task_cols = [TaskQuerySchema.query_name, TaskQuerySchema.duration_days_name]

    labeled = labels_df.filter(pl.col(label_col).is_not_null())
    if labeled.height == 0:
        return labeled

    # Per-(task, class) sample: derive a per-row hash key from
    # (subject_id, prediction_time, query, duration_days, seed), rank within each
    # (task, class) group by that key, and keep rank 0 — same deterministic-without-
    # positional-bias trick as sample_evaluation_tasks.sample_prediction_times_per_subject.
    # (Not df.sample(): that's positional, so re-sharding/reordering the input changes the
    # pick for the same seed; hashing row identity is stable across both.)
    ranked = (
        labeled.with_columns(
            pl.concat_str(
                [
                    pl.col(TaskQuerySchema.subject_id_name).cast(pl.Utf8),
                    pl.col(TaskQuerySchema.prediction_time_name).cast(pl.Utf8),
                    pl.col(TaskQuerySchema.query_name).cast(pl.Utf8),
                    pl.col(TaskQuerySchema.duration_days_name).cast(pl.Utf8),
                    pl.lit(str(seed)),
                ],
                separator="|",
            )
            .hash()
            .alias("_sample_key")
        )
        .sort([*task_cols, label_col, "_sample_key"])
        .with_columns(pl.int_range(0, pl.len()).over([*task_cols, label_col]).alias("_rank"))
    )

    selected = ranked.filter(pl.col("_rank") == 0).drop(["_sample_key", "_rank"])

    counts = selected.group_by(task_cols).agg(pl.col(label_col).n_unique().alias("_n_classes"))
    complete_tasks = counts.filter(pl.col("_n_classes") == 2).select(task_cols)

    n_total_tasks = counts.height
    n_dropped = n_total_tasks - complete_tasks.height
    if n_dropped:
        logger.warning(
            "Dropped %d/%d tracked task(s) missing a positive or negative example.",
            n_dropped,
            n_total_tasks,
        )

    return (
        selected.join(complete_tasks, on=task_cols, how="semi")
        .sort([*task_cols, label_col])
        .select(labels_df.columns)
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_eval_labels(eval_labels_dir: Path, split: str) -> pl.DataFrame:
    split_dir = eval_labels_dir / split
    parquets = sorted(split_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found under {split_dir}")
    return pl.scan_parquet(parquets).collect()


def _out_fp(out_dir: Path, split: str) -> Path:
    return out_dir / split / "0.parquet"


def run(
    eval_labels_dir: Path,
    out_dir: Path,
    split: str,
    seed: int,
    overwrite: bool = False,
) -> Path | None:
    """Sample task-tracking pairs for one split and write them to a single parquet.

    Returns the written parquet path, or ``None`` if output existed and
    ``overwrite=False``.
    """
    out_fp = _out_fp(out_dir, split)
    if not overwrite and out_fp.exists():
        logger.info("Task tracking pairs already exist at %s, skipping.", out_fp)
        return None

    labels_df = _read_eval_labels(eval_labels_dir, split)
    logger.info("Loaded %d labeled eval rows from %s", labels_df.height, eval_labels_dir / split)

    sample_seed = derive_seed(seed, "task_tracking_pairs", split)
    pairs = sample_task_tracking_pairs(labels_df, seed=sample_seed)
    n_tasks = pairs.height // 2
    if n_tasks == 0:
        logger.warning(
            "No task survived tracking-pair sampling (every task lacked a positive or negative "
            "example); writing an empty file to %s — in-training AUROC tracking will be a no-op.",
            out_fp,
        )
    logger.info("Sampled %d pos/neg pairs (%d tasks) for tracking.", pairs.height, n_tasks)

    aligned = TaskQuerySchema.align(pairs.to_arrow())
    _atomic_write_parquet(pl.from_arrow(aligned), out_fp)
    logger.info("Wrote %d task-tracking rows to %s", pairs.height, out_fp)
    return out_fp


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_task_tracking_pairs_config")
def main(cfg: DictConfig) -> None:
    """Produce one compact task-tracking-pairs parquet for one split.

    Usage:
        EQ_sample_task_tracking_pairs \\
            eval_labels_dir=$EVAL_TASKS_DIR/eval out_dir=$TASK_TRACKING_DIR split=tuning
    """
    eval_labels_dir = _require_path_arg(cfg.get("eval_labels_dir"), "eval_labels_dir")
    out_dir = _require_path_arg(cfg.get("out_dir"), "out_dir")

    run(
        eval_labels_dir=eval_labels_dir,
        out_dir=out_dir,
        split=cfg.get("split", "tuning"),
        seed=int(cfg.seed),
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
