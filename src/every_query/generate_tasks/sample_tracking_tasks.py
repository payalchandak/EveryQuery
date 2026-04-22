"""Sampled-tasks label generator for in-training per-task AUROC monitoring.

Sibling to ``sample_tasks`` (pretraining-shape, scattered tasks) and
``sample_evaluation_tasks`` (dense ``codes x durations`` grid).  Where the eval
generator builds the full grid for offline evaluation, this generator picks ``N``
random ``(code, duration_days)`` tuples once and labels them across the full
target split — the row shape needed by ``PerTaskAurocCallback`` to compute
per-task AUROC during training.

Pipeline (single CLI invocation, iterates all shards of one split in series):

    1. Sample ``N`` ``(code, duration_days)`` tuples once via :func:`sample_tasks`
       (uniform code x log-uniform duration), deterministic in ``seed``.
    2. For each shard of the target split:
       a. Sample up to ``K`` prediction times per subject via
          :func:`sample_prediction_times_per_subject`.
       b. Cross-join those prediction times with the ``N`` sampled tuples (paired
          — *not* ``codes x durations``).
       c. Label via :func:`evaluate_index_df`.
    3. Write one labeled parquet per shard at ``<out_dir>/labels/<shard>.parquet``
       (MTD-friendly — point ``task_labels_dir`` at ``<out_dir>/labels`` and the
       existing ``EveryQueryPytorchDataset`` consumes it).
    4. Write a sibling ``<out_dir>/tracking_tasks.parquet`` manifest listing the
       ``N`` sampled tuples + the seed, for reproducibility / debugging.  Lives
       *outside* the labels directory so MTD doesn't try to read it as a label
       shard.
"""

import logging
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from meds import DataSchema
from omegaconf import DictConfig

from every_query.data.schema import TaskQuerySchema, empty_task_query_df
from every_query.generate_tasks.sample_evaluation_tasks import (
    sample_prediction_times_per_subject,
)
from every_query.generate_tasks.sample_tasks import (
    TaskSpec,
    _atomic_write_parquet,
    _read_event_shard,
    _resolve_path,
    compute_max_time_per_subject,
    evaluate_index_df,
    read_query_codes,
    sample_tasks,
)
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure primitives
# ---------------------------------------------------------------------------


def build_paired_index_df(
    prediction_times: pl.DataFrame,
    tasks: list[TaskSpec],
) -> pl.DataFrame:
    """Cross-join prediction times with a *paired* list of ``(code, duration_days)`` tuples.

    Distinct from :func:`build_evaluation_index_df`, which cross-joins
    ``codes x durations``.  Here each ``TaskSpec`` is a specific ``(code,
    duration_days)`` pair, so the output row count is
    ``prediction_times.height * len(tasks)``.

    Examples:
        >>> from datetime import datetime
        >>> pt = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "prediction_time": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
        ... })
        >>> tasks = [TaskSpec(code="A", duration_days=7), TaskSpec(code="B", duration_days=30)]
        >>> out = build_paired_index_df(pt, tasks)
        >>> out.height
        4
        >>> out.columns
        ['subject_id', 'prediction_time', 'query', 'duration_days']
        >>> out["duration_days"].dtype
        Float32
        >>> sorted({(q, int(d)) for q, d in zip(out["query"], out["duration_days"])})
        [('A', 7), ('B', 30)]

        Empty inputs yield an empty frame with the right schema:

        >>> empty = pl.DataFrame({"subject_id": [], "prediction_time": []}, schema={
        ...     "subject_id": pl.Int64, "prediction_time": pl.Datetime("us"),
        ... })
        >>> build_paired_index_df(empty, tasks).height
        0
        >>> build_paired_index_df(pt, []).height
        0
    """
    out_schema = {
        TaskQuerySchema.subject_id_name: prediction_times.schema.get(
            TaskQuerySchema.subject_id_name, pl.Int64
        ),
        TaskQuerySchema.prediction_time_name: prediction_times.schema.get(
            TaskQuerySchema.prediction_time_name, pl.Datetime("us")
        ),
        TaskQuerySchema.query_name: pl.Utf8,
        TaskQuerySchema.duration_days_name: pl.Float32,
    }
    if prediction_times.height == 0 or not tasks:
        return pl.DataFrame(schema=out_schema)

    grid = pl.DataFrame(
        {
            TaskQuerySchema.query_name: [t.code for t in tasks],
            TaskQuerySchema.duration_days_name: [float(t.duration_days) for t in tasks],
        },
        schema={
            TaskQuerySchema.query_name: pl.Utf8,
            TaskQuerySchema.duration_days_name: pl.Float32,
        },
    )
    return prediction_times.join(grid, how="cross").select(list(out_schema))


def write_tracking_manifest(tasks: list[TaskSpec], seed: int, manifest_fp: Path) -> None:
    """Write a small parquet describing the sampled tasks.

    Lives outside the labels directory so MTD won't try to read it as a label shard.
    """
    df = pl.DataFrame(
        {
            TaskQuerySchema.query_name: [t.code for t in tasks],
            TaskQuerySchema.duration_days_name: [float(t.duration_days) for t in tasks],
            "seed": [int(seed)] * len(tasks),
        },
        schema={
            TaskQuerySchema.query_name: pl.Utf8,
            TaskQuerySchema.duration_days_name: pl.Float32,
            "seed": pl.Int64,
        },
    )
    _atomic_write_parquet(df, manifest_fp)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _labels_fp(labels_dir: Path, input_shard: str) -> Path:
    return labels_dir / f"{input_shard}.parquet"


def _label_one_shard(
    data_dir: Path,
    labels_dir: Path,
    split: str,
    input_shard: str,
    tasks: list[TaskSpec],
    prediction_times_per_subject: int,
    min_context_per_subject: int,
    pt_seed: int,
    overwrite: bool,
) -> Path | None:
    """Label one shard for the fixed task set; mirrors ``sample_evaluation_tasks.run_worker``."""
    out_fp = _labels_fp(labels_dir, input_shard)
    if out_fp.exists() and not overwrite:
        logger.info("Labels already exist at %s, skipping.", out_fp)
        return None

    shard_path = data_dir / "data" / split / f"{input_shard}.parquet"
    events_df = _read_event_shard(shard_path)
    logger.info("Loaded %d events from %s", events_df.height, shard_path)

    pred_times = sample_prediction_times_per_subject(
        events_df=events_df,
        k=prediction_times_per_subject,
        min_context_per_subject=min_context_per_subject,
        seed=pt_seed,
    )
    logger.info(
        "Sampled %d prediction times across %d subjects",
        pred_times.height,
        pred_times[DataSchema.subject_id_name].n_unique() if pred_times.height else 0,
    )

    index_df = build_paired_index_df(pred_times, tasks)
    if index_df.height == 0:
        out_cols = [
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            TaskQuerySchema.boolean_value_name,
            TaskQuerySchema.query_name,
            TaskQuerySchema.duration_days_name,
        ]
        labeled = empty_task_query_df().select(out_cols)
    else:
        max_time_df = compute_max_time_per_subject(events_df)
        labeled = evaluate_index_df(index_df, events_df, max_time_df)

    aligned = TaskQuerySchema.align(labeled.to_arrow())
    _atomic_write_parquet(pl.from_arrow(aligned), out_fp)
    logger.info("Wrote %d labeled rows to %s", labeled.height, out_fp)
    return out_fp


def discover_shards(data_dir: Path, split: str) -> list[str]:
    """Return shard basenames (without ``.parquet``) for a split, sorted lexicographically."""
    split_dir = data_dir / "data" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split dir {split_dir} not found")
    shards = sorted(p.stem for p in split_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {split_dir}")
    return shards


def run(
    data_dir: Path,
    out_dir: Path,
    codes: list[str],
    split: str,
    n_tasks: int,
    duration_min: int,
    duration_max: int,
    prediction_times_per_subject: int,
    min_context_per_subject: int,
    seed: int,
    overwrite: bool = False,
) -> Path:
    """Sample ``n_tasks`` tracking tasks once, label them across every shard of ``split``.

    Returns the labels directory path (the directory the training callback should
    point ``tracking_dir/labels`` at).
    """
    labels_dir = out_dir / "labels"
    manifest_fp = out_dir / "tracking_tasks.parquet"
    labels_dir.mkdir(parents=True, exist_ok=True)

    tasks_seed = derive_seed(seed, "tracking_tasks")
    tasks = sample_tasks(
        n=n_tasks,
        query_codes=codes,
        duration_low=duration_min,
        duration_high=duration_max,
        seed=tasks_seed,
    )
    if manifest_fp.exists() and not overwrite:
        existing = pl.read_parquet(manifest_fp)
        existing_pairs = {
            (q, float(d))
            for q, d in zip(
                existing[TaskQuerySchema.query_name],
                existing[TaskQuerySchema.duration_days_name],
                strict=True,
            )
        }
        new_pairs = {(t.code, float(t.duration_days)) for t in tasks}
        if existing_pairs != new_pairs:
            raise ValueError(
                f"Existing manifest at {manifest_fp} disagrees with the seeded sample "
                f"({len(existing_pairs)} existing vs {len(new_pairs)} new tuples).  "
                f"Pass overwrite=true to regenerate."
            )
        logger.info("Manifest already matches seed at %s; reusing.", manifest_fp)
    else:
        write_tracking_manifest(tasks, seed, manifest_fp)
        logger.info("Wrote tracking manifest with %d tasks to %s", len(tasks), manifest_fp)

    shards = discover_shards(data_dir, split)
    logger.info("Labeling %d tasks across %d shards of split %r", len(tasks), len(shards), split)

    for input_shard in shards:
        pt_seed = derive_seed(seed, "prediction_times", split, input_shard)
        _label_one_shard(
            data_dir=data_dir,
            labels_dir=labels_dir,
            split=split,
            input_shard=input_shard,
            tasks=tasks,
            prediction_times_per_subject=prediction_times_per_subject,
            min_context_per_subject=min_context_per_subject,
            pt_seed=pt_seed,
            overwrite=overwrite,
        )

    return labels_dir


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_tracking_tasks_config")
def main(cfg: DictConfig) -> None:
    """Produce a labeled tuning-split parquet directory for ``N`` sampled tracking tasks.

    Path fallbacks mirror the other generators: ``cfg.data_dir`` -> ``$INTERMEDIATE``,
    ``cfg.codes_dir`` -> ``$PROCESSED``, ``cfg.out_dir`` -> ``$TRACKING_DIR`` (or
    ``$TASK_DIR`` if ``$TRACKING_DIR`` is unset).

    Usage:
        EQ_generate_tracking_tasks split=tuning n_tasks=50 seed=0
    """
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = _resolve_path(cfg.get("data_dir"), "INTERMEDIATE", "data_dir")
    out_dir = _resolve_path(cfg.get("out_dir"), "TRACKING_DIR", "out_dir")

    codes_cfg = cfg.get("codes")
    if codes_cfg is None:
        codes_dir = _resolve_path(cfg.get("codes_dir"), "PROCESSED", "codes_dir")
        codes = read_query_codes(codes_dir)
    else:
        codes = read_query_codes(codes_cfg)

    run(
        data_dir=data_dir,
        out_dir=out_dir,
        codes=codes,
        split=str(cfg.split),
        n_tasks=int(cfg.n_tasks),
        duration_min=int(cfg.duration_min),
        duration_max=int(cfg.duration_max),
        prediction_times_per_subject=int(cfg.prediction_times_per_subject),
        min_context_per_subject=int(cfg.min_context_per_subject),
        seed=int(cfg.seed),
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
