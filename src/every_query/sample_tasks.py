"""Sampling-first task label generator for pre-training.

Architecture:
    1. Sample N tasks from a task distribution (uniform over codes x log-uniform over durations).
    2. Sample N x M patient contexts ``(subject_id, prediction_time)`` from the shard.
    3. Evaluate each ``(task, context)`` pair to produce ``(boolean_value, occurs)`` labels.

The script runs on one MEDS shard at a time and is parameterized by ``(input_shard, task_shard, seed)``
so that a ``hydra -m`` sweep covers the ``(task axis x patient axis)`` cartesian product with one output
file per worker.

Three stages execute inside a single ``main()`` call and are **idempotent**: the sampled task list,
the unlabeled index DataFrame, and the final labeled task parquet are each written to disk on first
run and loaded (without resampling) on subsequent runs. Set ``overwrite=true`` to force regeneration.

Design decisions (see issue #33):
- **Seeding**: tasks-seed depends only on ``(seed, task_shard)``, contexts-seed depends on
  ``(seed, input_shard, task_shard)``. Fixing ``task_shard`` and varying ``input_shard`` evaluates the
  *same* tasks on *different* patients; fixing ``input_shard`` and varying ``task_shard`` evaluates
  *different* tasks on *different* patients; the full sweep covers the product.
- **Task composition**: draw ``N`` tasks once and ``N x M`` contexts once, then zip them. Mathematically
  equivalent to ``N`` independent per-task draws under iid sampling, with one seed per side.
- **Censoring**: computed from ``max_time`` per subject (one groupby up-front per shard). Censored
  rows get ``boolean_value=True`` and ``occurs=False`` regardless of whether the event actually
  fired in the window.
- **Single-pass evaluation**: one ``join_asof(strategy="forward", by=["subject_id","query"])`` across
  the whole ``index_df`` against the events table, regardless of how many distinct codes are present.
"""

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    """A single pre-training task: one query code and one prediction-window duration (in days)."""

    code: str
    duration_days: int


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def derive_seed(*parts: int | str) -> int:
    """Stable 31-bit int seed derived from a tuple of ints/strings via blake2b.

    Python's builtin ``hash`` is not stable across processes, so hydra multirun workers would draw
    inconsistent samples.  Blake2b is cross-process stable and fast enough to be irrelevant at this scale.

    Examples:
        >>> derive_seed(1, "tasks", 0) == derive_seed(1, "tasks", 0)
        True
        >>> derive_seed(1, "tasks", 0) != derive_seed(1, "tasks", 1)
        True
        >>> derive_seed(1, "contexts", "shard_a", 0) != derive_seed(1, "contexts", "shard_b", 0)
        True
        >>> 0 <= derive_seed(1, "tasks", 0) < 2**31
        True
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")  # unit separator to avoid prefix collisions
    return int.from_bytes(h.digest(), "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Pure primitives
# ---------------------------------------------------------------------------


def sample_tasks(
    n: int,
    query_codes: list[str],
    duration_low: int,
    duration_high: int,
    seed: int,
) -> list[TaskSpec]:
    """Draw ``n`` iid ``TaskSpec`` s from a uniform x log-uniform distribution.

    The task distribution is:
        - ``code``: uniform over ``query_codes``
        - ``duration_days``: ``round(exp(Uniform(log(low), log(high))))``, clipped to ``[low, high]``

    Log-uniform preferentially samples shorter durations.

    Examples:
        >>> tasks = sample_tasks(
        ...     n=5, query_codes=["A", "B", "C"],
        ...     duration_low=1, duration_high=365, seed=0,
        ... )
        >>> len(tasks)
        5
        >>> all(t.code in {"A", "B", "C"} for t in tasks)
        True
        >>> all(1 <= t.duration_days <= 365 for t in tasks)
        True

        Determinism — same seed yields identical output:

        >>> sample_tasks(3, ["A", "B"], 1, 365, seed=42) == sample_tasks(3, ["A", "B"], 1, 365, seed=42)
        True

        Different seeds yield different draws (with overwhelming probability at n=10):

        >>> sample_tasks(10, ["A", "B"], 1, 365, seed=1) != sample_tasks(10, ["A", "B"], 1, 365, seed=2)
        True

        ``n=0`` is valid and returns an empty list:

        >>> sample_tasks(0, ["A"], 1, 365, seed=0)
        []

    Raises:
        ValueError: If ``n < 0``, ``query_codes`` is empty, or the duration range is invalid.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n})")
    if not query_codes:
        raise ValueError("query_codes must be non-empty")
    # Durations are quantized in days; non-integer bounds would silently bias the rounding step.
    if not isinstance(duration_low, int) or not isinstance(duration_high, int):
        raise TypeError(
            f"duration_low and duration_high must be ints (got {type(duration_low).__name__}, "
            f"{type(duration_high).__name__})"
        )
    if duration_low < 1:
        raise ValueError(f"duration_low must be >= 1 (got {duration_low}); log-uniform needs positive bounds")
    if duration_high < duration_low:
        raise ValueError(f"duration_high ({duration_high}) must be >= duration_low ({duration_low})")

    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    code_indices = rng.integers(0, len(query_codes), size=n)
    log_low, log_high = np.log(duration_low), np.log(duration_high)
    raw_durations = np.exp(rng.uniform(log_low, log_high, size=n))
    durations = np.clip(np.round(raw_durations).astype(int), duration_low, duration_high)

    return [
        TaskSpec(code=query_codes[int(code_indices[i])], duration_days=int(durations[i])) for i in range(n)
    ]


def sample_contexts(
    events_df: pl.DataFrame,
    n: int,
    min_context_per_subject: int,
    seed: int,
) -> pl.DataFrame:
    """Sample ``n`` ``(subject_id, prediction_time)`` contexts iid from the shard with replacement.

    A candidate prediction time is any event time at which the subject has already accumulated at
    least ``min_context_per_subject`` prior events.
    Sampling is with replacement so the caller can request ``n > n_candidates`` — natural for PT where
    context iid-ness matters more than strict coverage.

    Args:
        events_df: A shard of events with columns ``subject_id``, ``time``, ``code`` (sorted by
            ``(subject_id, time)``).
        n: Number of contexts to draw.
        min_context_per_subject: Minimum number of prior events a subject must have accumulated
            before a given event time can be used as a prediction time.
        seed: PRNG seed.

    Returns:
        ``DataFrame`` with columns ``(subject_id, prediction_time)`` and exactly ``n`` rows, or zero
        rows if no candidates exist.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n})")

    candidates = (
        events_df.with_columns(pl.col("time").cum_count().over("subject_id").alias("_ccs"))
        .filter(pl.col("_ccs") >= min_context_per_subject)
        .select(["subject_id", "time"])
        .unique()
        .rename({"time": "prediction_time"})
        # Sort explicitly: polars' `.unique()` is hash-based and order-unstable across
        # repeated calls, which would make `.sample(seed=...)` non-deterministic because
        # the sampler operates on row positions.
        .sort(["subject_id", "prediction_time"])
    )

    if n == 0 or candidates.height == 0:
        return candidates.head(0)

    return candidates.sample(n=n, with_replacement=True, seed=seed)


def build_index_df(
    tasks: list[TaskSpec],
    contexts: pl.DataFrame,
) -> pl.DataFrame:
    """Zip sampled tasks with sampled contexts to produce the unlabeled index DataFrame.

    ``contexts`` must have length ``len(tasks) * M`` for some integer ``M >= 0``.  Contexts are split
    into per-task chunks of size ``M`` and each chunk is labeled with its task's ``code`` and
    ``duration_days`` fields.  Mathematically equivalent under iid sampling to drawing ``M`` contexts
    independently per task.

    Returns:
        ``DataFrame`` with columns ``(task_id, subject_id, prediction_time, query, duration_days)``.
    """
    n_tasks = len(tasks)

    def _empty(pt_dtype: pl.DataType, sid_dtype: pl.DataType) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "task_id": pl.Int64,
                "subject_id": sid_dtype,
                "prediction_time": pt_dtype,
                "query": pl.Utf8,
                "duration_days": pl.Int64,
            }
        )

    pt_dtype = contexts.schema.get("prediction_time", pl.Datetime("us"))
    sid_dtype = contexts.schema.get("subject_id", pl.Int64)

    if n_tasks == 0 or contexts.height == 0:
        return _empty(pt_dtype, sid_dtype)

    if contexts.height % n_tasks != 0:
        raise ValueError(f"contexts.height ({contexts.height}) must be divisible by len(tasks) ({n_tasks})")
    contexts_per_task = contexts.height // n_tasks

    task_ids = pl.Series(
        "task_id",
        np.repeat(np.arange(n_tasks, dtype=np.int64), contexts_per_task),
    )
    query_col = pl.Series(
        "query",
        np.repeat([t.code for t in tasks], contexts_per_task),
        dtype=pl.Utf8,
    )
    duration_col = pl.Series(
        "duration_days",
        np.repeat([t.duration_days for t in tasks], contexts_per_task).astype(np.int64),
        dtype=pl.Int64,
    )

    return contexts.with_columns(task_ids, query_col, duration_col).select(
        "task_id", "subject_id", "prediction_time", "query", "duration_days"
    )


def compute_max_time_per_subject(events_df: pl.DataFrame) -> pl.DataFrame:
    """Return a ``(subject_id, max_time)`` DataFrame.

    Used once per shard to turn the per-row censoring check into an O(1) lookup inside
    ``evaluate_index_df``.
    """
    return events_df.group_by("subject_id").agg(pl.col("time").max().alias("max_time"))


def evaluate_index_df(
    index_df: pl.DataFrame,
    events_df: pl.DataFrame,
    max_time_per_subject: pl.DataFrame,
) -> pl.DataFrame:
    """Label an index DataFrame with ``(boolean_value, occurs)`` via a single ``join_asof``.

    Semantics:
        - ``censored = (prediction_time + duration_days) > max_time[subject_id]``
        - ``occurs = (not censored) AND (next event with matching code falls strictly within
          (prediction_time, prediction_time + duration_days))``
        - ``boolean_value = censored``

    The ``>`` on event time is enforced by shifting the asof key by ``+1µs`` since datetimes are
    stored at microsecond precision, which turns ``strategy="forward"``'s ``>=`` into a strict ``>``.

    Args:
        index_df: Output of ``build_index_df``. Must have columns ``subject_id``, ``prediction_time``,
            ``query``, ``duration_days``. If ``task_id`` is present it is ignored and dropped from
            the output.
        events_df: Shard events with columns ``subject_id``, ``time``, ``code``.
        max_time_per_subject: Output of ``compute_max_time_per_subject``.

    Returns:
        DataFrame with columns ``(subject_id, prediction_time, boolean_value, occurs, query,
        duration_days)``.
    """
    out_schema = {
        "subject_id": index_df.schema.get("subject_id", pl.Int64),
        "prediction_time": index_df.schema.get("prediction_time", pl.Datetime("us")),
        "boolean_value": pl.Boolean,
        "occurs": pl.Boolean,
        "query": pl.Utf8,
        "duration_days": pl.Int64,
    }
    if index_df.height == 0:
        return pl.DataFrame(schema=out_schema)

    # Left side: index rows with a +1µs-shifted prediction_time for the strict-> asof key.
    left = index_df.with_columns(
        (pl.col("prediction_time") + pl.duration(microseconds=1)).alias("_pt_shifted")
    ).sort(["subject_id", "query", "_pt_shifted"])

    # Right side: events renamed so the join-by column name matches the left.
    right = (
        events_df.rename({"code": "query"})
        .select(["subject_id", "query", "time"])
        .sort(["subject_id", "query", "time"])
    )

    joined = left.join_asof(
        right,
        by=["subject_id", "query"],
        left_on="_pt_shifted",
        right_on="time",
        strategy="forward",
    )
    joined = joined.join(max_time_per_subject, on="subject_id", how="left")

    # Rows whose subject is not present in max_time_per_subject (typically a pre-seeded or
    # hand-edited index_df referencing subjects outside this shard) come out of the left join
    # with max_time=null.  A naïve comparison would produce null booleans, which would break
    # downstream torch conversion.  Policy: treat unknown-subject rows as fully censored
    # (boolean_value=True, occurs=False) — the same outcome a real "no future data observed"
    # row would produce.  We log a warning so the condition is visible.
    n_unknown = joined.filter(pl.col("max_time").is_null()).height
    if n_unknown > 0:
        logger.warning(
            "%d index_df row(s) reference subjects not present in events_df; "
            "they will be labeled as censored (boolean_value=True, occurs=False).",
            n_unknown,
        )

    duration_expr = pl.duration(days=pl.col("duration_days"))
    window_end = pl.col("prediction_time") + duration_expr
    # `(window_end > max_time).fill_null(True)` resolves the missing-subject case to censored.
    censored = (window_end > pl.col("max_time")).fill_null(True)
    event_in_window = pl.col("time").is_not_null() & (pl.col("time") < window_end)

    return (
        joined.with_columns(censored.alias("_censored"))
        .with_columns(
            pl.col("_censored").alias("boolean_value"),
            (pl.col("_censored").not_() & event_in_window).alias("occurs"),
        )
        .select("subject_id", "prediction_time", "boolean_value", "occurs", "query", "duration_days")
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_event_shard(file_path: str | Path) -> pl.DataFrame:
    """Read a shard parquet and return ``(subject_id, time, code)`` sorted by subject+time.

    The parquet schema is normalized explicitly so the returned frame is type-stable regardless
    of how the source shard encoded strings or timestamps:

    - ``code`` is cast to ``pl.Utf8`` so it compares against the ``query`` column of ``index_df``
      (also ``Utf8``) in ``evaluate_index_df``'s ``join_asof(by=["subject_id","query"])``.  Mixed
      ``Categorical``/``Utf8`` or ``<integer vocab index>``/``Utf8`` joins would either raise or
      silently produce zero matches.  Upstream stages may store codes as categoricals or integer
      vocab indices; casting to ``Utf8`` here avoids coupling to either representation.
    - ``time`` is cast to ``pl.Datetime("us")`` because ``evaluate_index_df`` implements strict
      ``>`` via a ``+1µs`` shift on the asof key.  At millisecond precision that shift would
      round to zero and silently turn the comparison into ``>=``.
    """
    return (
        pl.read_parquet(file_path)
        .select(["subject_id", "time", "code"])
        .with_columns(
            pl.col("time").cast(pl.Datetime("us")),
            pl.col("code").cast(pl.Utf8),
        )
        .unique()
        .sort(["subject_id", "time"])
    )


def _read_query_codes(codes_dir: str | Path) -> list[str]:
    """Read the universe of query codes from ``{codes_dir}/metadata/codes.parquet``.

    ``codes_dir`` is the metadata root (``$PROCESSED`` in the standard layout), **not** the event
    shard root (``$INTERMEDIATE``).  Event shards live under ``$INTERMEDIATE/data/...`` while query
    codes live under ``$PROCESSED/metadata/codes.parquet``; these are typically distinct
    subdirectories of ``$DATA_DIR`` (see ``.env.example``) and the sampler should not conflate them.

    The ``.unique()`` result is explicitly sorted because polars' default hash-based unique is
    order-unstable across distinct DataFrame instances, which would make ``sample_tasks``
    non-deterministic with respect to the tasks seed across workers reading the same metadata file.
    """
    codes_fp = Path(codes_dir) / "metadata" / "codes.parquet"
    return pl.read_parquet(codes_fp).select("code").unique().sort("code").to_series().to_list()


def _unique_tmp_path(fp: Path) -> Path:
    """Allocate a unique sibling tmpfile next to ``fp``.

    ``tempfile.mkstemp`` returns a process-unique filename in the target directory, so two
    workers with the same ``fp`` (e.g. a SLURM array retry racing a still-running original, or
    a manual rerun while an old job is still going) won't clobber each other's tmpfile.  We
    close the fd immediately — the caller writes via its own handle — and rely on ``os.replace``
    at the end to be the atomicity primitive.
    """
    fp.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{fp.name}.",
        suffix=".tmp",
        dir=str(fp.parent),
    )
    os.close(fd)
    return Path(tmp_name)


def _atomic_write_text(fp: Path, text: str) -> None:
    """Write ``text`` to ``fp`` atomically via a unique sibling tmpfile + ``os.replace``.

    A failed mid-write leaves at most an orphan tmpfile; the target ``fp`` either has its
    previous contents or the fully-written new contents.  Concurrent writers to the same ``fp``
    don't corrupt each other because the tmp filename is unique per call.
    """
    tmp = _unique_tmp_path(fp)
    try:
        tmp.write_text(text)
        os.replace(tmp, fp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_parquet(df: pl.DataFrame, fp: Path) -> None:
    """Write ``df`` to ``fp`` atomically via a unique sibling tmpfile + ``os.replace``."""
    tmp = _unique_tmp_path(fp)
    try:
        df.write_parquet(tmp)
        os.replace(tmp, fp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def save_tasks(tasks: list[TaskSpec], fp: Path) -> None:
    """Serialize a task list to JSON with one entry per task (atomic)."""
    _atomic_write_text(
        fp,
        json.dumps(
            [{"code": t.code, "duration_days": t.duration_days} for t in tasks],
            indent=2,
        ),
    )


def load_tasks(fp: Path) -> list[TaskSpec]:
    """Load a task list previously written by :func:`save_tasks`."""
    data = json.loads(Path(fp).read_text())
    return [TaskSpec(code=d["code"], duration_days=int(d["duration_days"])) for d in data]


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def _artifact_paths(
    out_dir: Path,
    split: str,
    input_shard: str,
    task_shard: int,
) -> tuple[Path, Path, Path, Path]:
    """Resolve the ``(tasks_fp, index_fp, labels_fp, run_meta_fp)`` quadruple for one worker."""
    worker_id = f"{input_shard}__{task_shard:04d}"
    tasks_fp = out_dir / "_artifacts" / "tasks" / split / f"{worker_id}.json"
    index_fp = out_dir / "_artifacts" / "index" / split / f"{worker_id}.parquet"
    labels_fp = out_dir / split / f"{worker_id}.parquet"
    run_meta_fp = out_dir / "_artifacts" / "runs" / split / f"{worker_id}.json"
    return tasks_fp, index_fp, labels_fp, run_meta_fp


def _build_run_meta(
    seed: int,
    n_tasks: int,
    contexts_per_task: int,
    duration_min: int,
    duration_max: int,
    min_context_per_subject: int,
) -> dict[str, int]:
    """Capture the sampling parameters that must match for a cached worker's artifacts to be reusable.

    The file-level identity of the worker (``split``, ``input_shard``, ``task_shard``) is encoded
    in the output path itself and therefore intentionally omitted here — two artifacts with different
    worker identities can never collide on disk, so comparing them here would be redundant.
    """
    return {
        "seed": int(seed),
        "n_tasks": int(n_tasks),
        "contexts_per_task": int(contexts_per_task),
        "duration_min": int(duration_min),
        "duration_max": int(duration_max),
        "min_context_per_subject": int(min_context_per_subject),
    }


def _validate_run_meta(run_meta_fp: Path, current: dict[str, int]) -> None:
    """Raise ``ValueError`` if an on-disk worker meta file exists and disagrees with ``current``.

    Absent meta is tolerated: legacy artifacts from before the meta sidecar, or hand-pre-seeded
    tasks/index files dropped by a user debugging the pipeline, have no sidecar to validate against
    and are assumed to be the user's intent.  Mismatched meta is a hard failure — the whole point
    of the sidecar is to prevent silent stale-artifact reuse.
    """
    if not run_meta_fp.exists():
        return
    on_disk = json.loads(run_meta_fp.read_text())
    if on_disk == current:
        return
    diff_lines = []
    all_keys = sorted(set(on_disk) | set(current))
    for key in all_keys:
        on_disk_val = on_disk.get(key, "<missing>")
        current_val = current.get(key, "<missing>")
        if on_disk_val != current_val:
            diff_lines.append(f"  {key}: on_disk={on_disk_val!r}, current={current_val!r}")
    raise ValueError(
        "Refusing to reuse cached sampler artifacts: the worker config on disk at "
        f"{run_meta_fp} differs from the requested config:\n"
        + "\n".join(diff_lines)
        + "\nPass overwrite=true to regenerate this worker's artifacts, "
        "or delete the _artifacts/ subdirectory if you want a clean run."
    )


def run_worker(
    data_dir: Path,
    out_dir: Path,
    codes_dir: Path,
    split: str,
    input_shard: str,
    task_shard: int,
    seed: int,
    n_tasks: int,
    contexts_per_task: int,
    duration_min: int,
    duration_max: int,
    min_context_per_subject: int,
    overwrite: bool = False,
) -> Path | None:
    """Run the three-stage pipeline for one worker.

    Each stage is idempotent: the sampled task list, the unlabeled index DataFrame, and the final
    labeled parquet are each written on first run and loaded (without resampling) on subsequent
    runs. Set ``overwrite=True`` to force regeneration.

    A per-worker meta sidecar at ``_artifacts/runs/{split}/{worker_id}.json`` captures the
    sampling parameters the cached artifacts were generated with.  On every non-``overwrite`` rerun
    those parameters are compared against the requested parameters and any mismatch raises
    ``ValueError`` rather than silently reusing stale artifacts.  Missing meta (e.g. legacy
    artifacts or hand-preseeded tasks.json) is tolerated.

    Returns:
        The path of the labeled parquet, or ``None`` if the labels already existed, the meta
        sidecar matched the current config, and ``overwrite=False``.

    Raises:
        ValueError: If cached artifacts exist alongside a meta sidecar whose recorded config
            disagrees with the requested config and ``overwrite`` is False.
    """
    tasks_fp, index_fp, labels_fp, run_meta_fp = _artifact_paths(out_dir, split, input_shard, task_shard)
    current_meta = _build_run_meta(
        seed=seed,
        n_tasks=n_tasks,
        contexts_per_task=contexts_per_task,
        duration_min=duration_min,
        duration_max=duration_max,
        min_context_per_subject=min_context_per_subject,
    )

    if not overwrite:
        # Reject cached artifacts if their on-disk config disagrees with what we were asked for.
        _validate_run_meta(run_meta_fp, current_meta)

    if labels_fp.exists() and not overwrite:
        logger.info("Labels already exist at %s, skipping.", labels_fp)
        return None

    # Stage 1: tasks
    if tasks_fp.exists() and not overwrite:
        logger.info("Loading tasks from %s", tasks_fp)
        tasks = load_tasks(tasks_fp)
    else:
        query_codes = _read_query_codes(codes_dir)
        tasks_seed = derive_seed(seed, "tasks", task_shard)
        tasks = sample_tasks(
            n=n_tasks,
            query_codes=query_codes,
            duration_low=duration_min,
            duration_high=duration_max,
            seed=tasks_seed,
        )
        save_tasks(tasks, tasks_fp)
        logger.info("Wrote %d tasks to %s", len(tasks), tasks_fp)

    # Load the shard once.
    shard_path = data_dir / "data" / split / f"{input_shard}.parquet"
    events_df = _read_event_shard(shard_path)
    logger.info("Loaded %d events from %s", events_df.height, shard_path)

    # Stage 2: index
    if index_fp.exists() and not overwrite:
        logger.info("Loading index from %s", index_fp)
        index_df = pl.read_parquet(index_fp)
    else:
        contexts_seed = derive_seed(seed, "contexts", input_shard, task_shard)
        contexts = sample_contexts(
            events_df=events_df,
            n=len(tasks) * contexts_per_task,
            min_context_per_subject=min_context_per_subject,
            seed=contexts_seed,
        )
        index_df = build_index_df(tasks, contexts)
        _atomic_write_parquet(index_df, index_fp)
        logger.info("Wrote %d index rows to %s", index_df.height, index_fp)

    # Stage 3: labels
    max_time_df = compute_max_time_per_subject(events_df)
    labeled = evaluate_index_df(index_df, events_df, max_time_df)
    # Downstream MEDS dataloader does not use task_id; it is intentionally absent from the
    # published schema.
    _atomic_write_parquet(labeled, labels_fp)
    logger.info("Wrote %d labeled rows to %s", labeled.height, labels_fp)

    # Persist the meta sidecar last so that a crash partway through leaves the cache in a state
    # where the next rerun will detect an incomplete artifact set (labels missing) and redo the
    # work, rather than trusting a meta that points at nonexistent artifacts.
    _atomic_write_text(run_meta_fp, json.dumps(current_meta, indent=2, sort_keys=True))
    return labels_fp


def _resolve_path(cfg_value: str | None, env_var: str, name: str) -> Path:
    """Prefer an explicit cfg value; fall back to ``$env_var``; otherwise raise.

    Used by ``main`` to resolve the three path roots (``data_dir``, ``out_dir``, ``codes_dir``).
    Factored out so tests can exercise the fallback matrix without spinning up a full Hydra run.
    """
    if cfg_value is not None:
        return Path(str(cfg_value))
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value)
    raise ValueError(
        f"{name} must be set: pass {name}=... on the CLI, set it in sample_tasks_config.yaml, "
        f"or export ${env_var} (or define it in .env — sample_tasks calls load_dotenv())."
    )


@hydra.main(version_base=None, config_path=".", config_name="sample_tasks_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point.

    Loads ``.env`` via python-dotenv before resolving paths, following the repo convention where
    ``$INTERMEDIATE`` / ``$PROCESSED`` / ``$TASK_DIR`` live in a gitignored ``.env`` file rather
    than being exported by the user.  Path fallbacks: ``cfg.data_dir`` falls back to
    ``$INTERMEDIATE``, ``cfg.codes_dir`` to ``$PROCESSED``, ``cfg.out_dir`` to ``$TASK_DIR``.

    See :func:`run_worker` for the per-worker pipeline.
    """
    # Late import so `load_dotenv()` doesn't run at module import time (which would be an
    # unexpected side effect for programmatic callers / tests of the pure primitives).
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = _resolve_path(cfg.get("data_dir"), "INTERMEDIATE", "data_dir")
    out_dir = _resolve_path(cfg.get("out_dir"), "TASK_DIR", "out_dir")
    codes_dir = _resolve_path(cfg.get("codes_dir"), "PROCESSED", "codes_dir")

    run_worker(
        data_dir=data_dir,
        out_dir=out_dir,
        codes_dir=codes_dir,
        split=str(cfg.split),
        input_shard=str(cfg.input_shard),
        task_shard=int(cfg.task_shard),
        seed=int(cfg.seed),
        n_tasks=int(cfg.n_tasks),
        contexts_per_task=int(cfg.contexts_per_task),
        duration_min=int(cfg.duration_min),
        duration_max=int(cfg.duration_max),
        min_context_per_subject=int(cfg.min_context_per_subject),
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
