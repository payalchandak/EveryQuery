"""Evaluation-shape task label generator.

Sibling to ``sample_tasks`` (pretraining-shape, scattered tasks).  Where
``sample_tasks`` draws ``N`` random tasks and ``N * M`` random contexts,
``sample_evaluation_tasks`` samples ``K`` prediction times per subject and builds
the dense grid: ``subjects x sampled_times x codes x durations``.  That's the
row shape needed to compute per-``(query, duration_days)`` metrics over a split
— every `(subject, time)` gets scored on every task the caller asked about.

Pipeline:
    1. For the chosen input shard, sample up to ``K`` candidate prediction times
       per subject (any event time at which the subject has accumulated at least
       ``min_context_per_subject`` prior events).
    2. Cross-join with the full ``(codes x durations)`` grid.
    3. Label via :func:`every_query.generate_tasks.sample_tasks.evaluate_index_df`
       (single ``join_asof`` across the whole index frame).
    4. Align to ``TaskQuerySchema`` and write a single parquet per worker.

Seeding:
    Prediction-time sampling is deterministic in ``(seed, input_shard, split)``.
    There is no task-axis analogue of :func:`every_query.utils.seeds.derive_seed`
    here — the task axis is fully specified by ``(codes, durations)``, so only
    the prediction-time sampler needs randomness.
"""

import logging
import math
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from meds import DataSchema
from omegaconf import DictConfig

from every_query.data.schema import TaskQuerySchema, empty_task_query_df
from every_query.generate_tasks.sample_tasks import (
    _atomic_write_parquet,
    _read_event_shard,
    _resolve_path,
    compute_max_time_per_subject,
    evaluate_index_df,
    read_query_codes,
)
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure primitives
# ---------------------------------------------------------------------------


def sample_prediction_times_per_subject(
    events_df: pl.DataFrame,
    k: int,
    min_context_per_subject: int,
    seed: int,
) -> pl.DataFrame:
    """Sample up to ``k`` prediction times per subject from event times.

    A candidate prediction time is any event time at which the subject has
    accumulated at least ``min_context_per_subject`` prior events.  Sampling is
    without replacement within each subject; subjects with fewer than ``k``
    candidates contribute all of them.

    Args:
        events_df: Shard events with columns ``subject_id``, ``time``, ``code``
            (sorted by ``(subject_id, time)``).
        k: Max prediction times per subject.
        min_context_per_subject: Minimum prior events a subject must have
            accumulated before a given event time can be used as a prediction
            time.
        seed: PRNG seed.  Deterministic in ``(events_df, k, min_context_per_subject, seed)``.

    Returns:
        DataFrame with columns ``(subject_id, prediction_time)``, sorted by
        both.  Zero rows if no candidates exist.

    Examples:
        >>> from datetime import datetime
        >>> events = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2, 2],
        ...     "time": [
        ...         datetime(2024, 1, 1), datetime(2024, 1, 2),
        ...         datetime(2024, 1, 3), datetime(2024, 1, 4),
        ...         datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),
        ...     ],
        ...     "code": ["A"] * 7,
        ... })
        >>> out = sample_prediction_times_per_subject(events, k=2, min_context_per_subject=2, seed=0)
        >>> sorted(out["subject_id"].unique().to_list())
        [1, 2]
        >>> # Each subject gets at most 2 sampled times
        >>> out.group_by("subject_id").len().sort("subject_id")["len"].to_list()
        [2, 2]

        ``min_context_per_subject`` filters out subjects who don't have enough
        history yet:

        >>> out = sample_prediction_times_per_subject(events, k=5, min_context_per_subject=10, seed=0)
        >>> out.height
        0

        Determinism — same seed, same output:

        >>> a = sample_prediction_times_per_subject(events, k=2, min_context_per_subject=1, seed=42)
        >>> b = sample_prediction_times_per_subject(events, k=2, min_context_per_subject=1, seed=42)
        >>> a.equals(b)
        True
    """
    if k < 0:
        raise ValueError(f"k must be >= 0 (got {k})")

    candidates = (
        events_df.with_columns(
            pl.col(DataSchema.time_name).cum_count().over(DataSchema.subject_id_name).alias("_ccs")
        )
        .filter(pl.col("_ccs") >= min_context_per_subject)
        .select([DataSchema.subject_id_name, DataSchema.time_name])
        .unique()
        .rename({DataSchema.time_name: TaskQuerySchema.prediction_time_name})
        .sort([DataSchema.subject_id_name, TaskQuerySchema.prediction_time_name])
    )

    if k == 0 or candidates.height == 0:
        return candidates.head(0)

    # Per-subject sample: derive a per-row sampling key from
    # ``hash(subject_id, prediction_time, seed)``, rank within each subject by that
    # key, and keep the first ``k``.  ``pl.int_range(...).shuffle(seed).over(s)``
    # re-applies the same shuffle pattern to every subject, so two subjects with the
    # same candidate count would deterministically receive identical positional
    # selections — a per-row hash avoids that while staying deterministic in
    # ``(events_df, k, min_context_per_subject, seed)``.
    shuffled = (
        candidates.with_columns(
            pl.concat_str(
                [
                    pl.col(DataSchema.subject_id_name).cast(pl.Utf8),
                    pl.col(TaskQuerySchema.prediction_time_name).cast(pl.Utf8),
                    pl.lit(str(seed)),
                ],
                separator="|",
            )
            .hash()
            .alias("_sample_key")
        )
        .sort(
            [
                DataSchema.subject_id_name,
                "_sample_key",
                TaskQuerySchema.prediction_time_name,
            ]
        )
        .with_columns(pl.int_range(0, pl.len()).over(DataSchema.subject_id_name).alias("_rank"))
    )
    return (
        shuffled.filter(pl.col("_rank") < k)
        .drop(["_sample_key", "_rank"])
        .sort([DataSchema.subject_id_name, TaskQuerySchema.prediction_time_name])
    )


def build_evaluation_index_df(
    prediction_times: pl.DataFrame,
    codes: list[str],
    durations: list[int],
) -> pl.DataFrame:
    """Cross-join prediction times with ``(codes x durations)`` into the evaluation grid.

    Args:
        prediction_times: DataFrame with columns ``(subject_id, prediction_time)``.
        codes: Query codes to evaluate at every prediction time.
        durations: Duration-day horizons to evaluate at every prediction time.

    Returns:
        DataFrame with columns ``(subject_id, prediction_time, query, duration_days)``
        whose row count is ``prediction_times.height * len(codes) * len(durations)``.

    Examples:
        >>> from datetime import datetime
        >>> pt = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "prediction_time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 1)],
        ... })
        >>> out = build_evaluation_index_df(pt, codes=["A", "B"], durations=[7, 30])
        >>> out.height
        12
        >>> out.columns
        ['subject_id', 'prediction_time', 'query', 'duration_days']
        >>> out["duration_days"].dtype
        Float32
        >>> sorted(out["query"].unique().to_list())
        ['A', 'B']

        Empty inputs yield an empty frame with the right schema:

        >>> empty = pl.DataFrame({"subject_id": [], "prediction_time": []}, schema={
        ...     "subject_id": pl.Int64, "prediction_time": pl.Datetime("us"),
        ... })
        >>> build_evaluation_index_df(empty, codes=["A"], durations=[30]).height
        0
    """
    if not codes:
        raise ValueError("codes must be non-empty")
    if not durations:
        raise ValueError("durations must be non-empty")
    if any(not isinstance(d, int) for d in durations):
        raise TypeError(f"durations must all be ints (got {[type(d).__name__ for d in durations]})")

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
    if prediction_times.height == 0:
        return pl.DataFrame(schema=out_schema)

    # Materialise the (code, duration) grid once; cross-join with prediction_times.
    grid = pl.DataFrame(
        {
            TaskQuerySchema.query_name: [c for c in codes for _ in durations],
            TaskQuerySchema.duration_days_name: ([d for _ in codes for d in durations]),
        },
        schema={
            TaskQuerySchema.query_name: pl.Utf8,
            TaskQuerySchema.duration_days_name: pl.Float32,
        },
    )
    return prediction_times.join(grid, how="cross").select(list(out_schema))


def subsample_subject_ids(
    events_df: pl.DataFrame,
    fraction: float | None,
    seed: int,
) -> pl.DataFrame:
    """Deterministically retain a fraction of unique subject IDs in ``events_df``.

    Returns ``events_df`` unchanged when ``fraction`` is ``None`` or exactly
    ``1.0``.  Otherwise applies a per-subject hash-threshold filter: each
    subject's stable ``blake2b((subject_id, seed))`` digest is mapped onto
    ``[0, 2**64)`` and the subject is kept iff that value is below
    ``fraction * 2**64``.  This makes the expected fraction kept independent
    of shard size — small shards no longer round up to ``1`` and bias the
    global sample.

    Raises ``ValueError`` for non-finite values or any value outside
    ``(0, 1]`` so misconfigured runs fail fast rather than silently producing
    a full-population evaluation.
    """
    if fraction is None:
        return events_df
    # ``bool`` is an ``int`` subclass in Python, so a stray ``True`` would pass
    # ``isinstance(..., (int, float))`` checks elsewhere and become ``1.0`` here —
    # silently turning a misconfiguration into a full-population run.  Reject it.
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise TypeError(
            f"subject_subsample_fraction must be a number in (0, 1] or None, "
            f"got {type(fraction).__name__}: {fraction!r}"
        )
    if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise ValueError(f"subject_subsample_fraction must be a finite number in (0, 1], got {fraction!r}")
    if fraction == 1.0:
        return events_df

    threshold = int(fraction * (1 << 64))
    return events_df.filter(
        pl.concat_str(
            [
                pl.col(DataSchema.subject_id_name).cast(pl.Utf8),
                pl.lit(str(seed)),
            ],
            separator="|",
        ).hash()
        < pl.lit(threshold, dtype=pl.UInt64)
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _labels_fp(out_dir: Path, split: str, input_shard: str) -> Path:
    return out_dir / split / f"{input_shard}.parquet"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def run_worker(
    data_dir: Path,
    out_dir: Path,
    split: str,
    input_shard: str,
    codes: list[str],
    durations: list[int],
    prediction_times_per_subject: int,
    min_context_per_subject: int,
    seed: int,
    overwrite: bool = False,
    subject_subsample_fraction: float | None = None,
) -> Path | None:
    """Generate one evaluation-tasks parquet for one input shard + split.

    Returns the written parquet path, or ``None`` if output existed and
    ``overwrite=False``.
    """
    labels_fp = _labels_fp(out_dir, split, input_shard)
    if labels_fp.exists() and not overwrite:
        logger.info("Labels already exist at %s, skipping.", labels_fp)
        return None

    shard_path = data_dir / "data" / split / f"{input_shard}.parquet"
    events_df = _read_event_shard(shard_path)
    logger.info("Loaded %d events from %s", events_df.height, shard_path)

    if subject_subsample_fraction is not None:
        subj_seed = derive_seed(seed, "subject_subsample", split, input_shard)
        events_df = subsample_subject_ids(events_df, subject_subsample_fraction, subj_seed)
        logger.info(
            "Subsampled to %d events / %d subjects (fraction=%.4f)",
            events_df.height,
            events_df[DataSchema.subject_id_name].n_unique(),
            subject_subsample_fraction,
        )

    pt_seed = derive_seed(seed, "prediction_times", split, input_shard)
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

    index_df = build_evaluation_index_df(pred_times, codes=codes, durations=durations)

    # Empty-input fast path: worker that sampled zero prediction times still writes
    # an empty TaskQuerySchema parquet so downstream consumers (EQ_predict) see a
    # well-formed input dir even on sparse splits.
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
    _atomic_write_parquet(pl.from_arrow(aligned), labels_fp)
    logger.info("Wrote %d labeled eval rows to %s", labeled.height, labels_fp)
    return labels_fp


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_evaluation_tasks_config")
def main(cfg: DictConfig) -> None:
    """Produce one evaluation-tasks parquet per (split, input_shard) pair.

    Path fallback mirrors ``sample_tasks``: ``cfg.data_dir`` -> ``$INTERMEDIATE``,
    ``cfg.codes_dir`` -> ``$PROCESSED``, ``cfg.out_dir`` -> ``$TASK_DIR`` (new
    subdir ``eval/`` underneath so training-task parquets and evaluation-task
    parquets don't collide in one directory).

    Usage (single worker):
        EQ_generate_evaluation_tasks \\
            split=held_out input_shard=0 \\
            prediction_times_per_subject=5 \\
            'codes=[HR, TEMP]' 'durations=[1, 7, 30]'

    Sweep across shards with
    ``python -m every_query.generate_tasks.sample_evaluation_tasks -m input_shard=0,1,2,...``.
    """
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = _resolve_path(cfg.get("data_dir"), "INTERMEDIATE", "data_dir")
    out_dir = _resolve_path(cfg.get("out_dir"), "TASK_DIR", "out_dir")

    codes_cfg = cfg.get("codes")
    if codes_cfg is None:
        codes_dir = _resolve_path(cfg.get("codes_dir"), "PROCESSED", "codes_dir")
        codes = read_query_codes(codes_dir)
    else:
        codes = read_query_codes(codes_cfg)

    # Durations must be whole-day ints — silent truncation (e.g. ``0.5 → 0``) would
    # change the window semantics.  ``TaskQuerySchema.duration_days`` is float32 on
    # disk, but the CLI contract here is integer days; fractional durations are a
    # deliberate-future-feature (see #129), not a silent-today feature.
    durations: list[int] = []
    for i, d in enumerate(cfg.durations):
        if isinstance(d, bool) or not isinstance(d, int | float):
            raise TypeError(f"cfg.durations[{i}] must be a number, got {type(d).__name__}: {d!r}")
        if not float(d).is_integer():
            raise ValueError(
                f"cfg.durations[{i}] must be a whole-day integer, got {d!r}.  "
                f"Fractional horizons aren't supported by this CLI yet — see #129."
            )
        durations.append(int(d))

    split = cfg.split
    input_shard = str(cfg.input_shard)

    # Reject booleans up front: Hydra/OmegaConf parses ``subject_subsample_fraction=true``
    # as a Python ``True``, which would otherwise become ``1.0`` and silently disable
    # subsampling — the opposite of the fail-fast contract documented in the helper.
    ssf_raw = cfg.get("subject_subsample_fraction")
    if isinstance(ssf_raw, bool) or (ssf_raw is not None and not isinstance(ssf_raw, int | float)):
        raise TypeError(
            f"cfg.subject_subsample_fraction must be a number in (0, 1] or null, "
            f"got {type(ssf_raw).__name__}: {ssf_raw!r}"
        )
    subject_subsample_fraction = None if ssf_raw is None else float(ssf_raw)

    run_worker(
        data_dir=data_dir,
        out_dir=Path(out_dir) / "eval",
        split=split,
        input_shard=input_shard,
        codes=codes,
        durations=durations,
        prediction_times_per_subject=int(cfg.prediction_times_per_subject),
        min_context_per_subject=int(cfg.min_context_per_subject),
        seed=int(cfg.seed),
        overwrite=bool(cfg.get("overwrite", False)),
        subject_subsample_fraction=subject_subsample_fraction,
    )


if __name__ == "__main__":
    main()
