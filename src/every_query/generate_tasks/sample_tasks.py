"""Sampling-first task label generator for pre-training.

Architecture:
    1. Sample N tasks from a task distribution (uniform over codes x log-uniform over durations).
    2. Sample N x M patient contexts ``(subject_id, prediction_time)`` from the shard.
    3. Evaluate each ``(task, context)`` pair to produce ``(boolean_value, occurs)`` labels.

The script runs on one MEDS shard at a time and is parameterized by ``(input_shard, task_shard, seed)``
so that a ``hydra -m`` sweep covers the ``(task axis x patient axis)`` cartesian product with one output
file per worker.

The worker is a pure function of its inputs: all three stages execute in-memory inside a single
``main()`` call and write only the final labeled parquet to disk.  Reruns with identical inputs
produce identical labels — determinism comes from :func:`~every_query.utils.seeds.derive_seed`
splitting the task and context axes, not from persisted intermediate state.  Set
``overwrite=true`` to regenerate labels for a worker whose output file already exists.

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

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from meds import DataSchema
from omegaconf import DictConfig, ListConfig

from every_query.data.schema import TaskQuerySchema, empty_task_query_df
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    """A single pre-training task: one query code and one prediction-window duration (in days)."""

    code: str
    duration_days: int


@dataclass(frozen=True)
class QuerySpec:
    """A single sampled query: one code and one float prediction-window duration (in days).

    The redesign's Stage 1 output (see ``redesign-spec.md``).  Unlike the legacy :class:`TaskSpec`,
    ``duration_days`` is a **float** — durations are not rounded to whole days.
    """

    code: str
    duration_days: float


@dataclass(frozen=True)
class QueryDistribution:
    """The Stage 1 query distribution: owns both the code draw and the duration draw.

    Stage 1 of the redesigned sampler is just ``query_dist.sample(num_queries, rng)``.  Code-universe
    resolution stays **outside** this dataclass (the caller runs :func:`read_query_codes` and passes the
    result in), so the dataclass does no file I/O.

    Args:
        query_codes: Already-resolved code universe (one code per query).  ``query_universe_size`` is
            derived as ``len(query_codes)``.
        min_duration: Lower duration bound in days (must be > 0).
        max_duration: Upper duration bound in days (must be >= ``min_duration``).
        duration_distribution: ``"uniform"`` or ``"log-uniform"``.  Log-uniform preferentially samples
            shorter durations.

    Examples:
        >>> import numpy as np
        >>> dist = QueryDistribution(["A", "B", "C"], min_duration=1.0, max_duration=365.0,
        ...                          duration_distribution="log-uniform")
        >>> dist.query_universe_size
        3
        >>> specs = dist.sample(5, np.random.default_rng(0))
        >>> len(specs)
        5
        >>> all(s.code in {"A", "B", "C"} for s in specs)
        True
        >>> all(1.0 <= s.duration_days <= 365.0 for s in specs)
        True

        Durations are floats, not rounded to whole days:

        >>> any(s.duration_days != round(s.duration_days) for s in specs)
        True

        Determinism — same seed yields identical output:

        >>> from every_query.utils.seeds import derive_seed
        >>> seed = derive_seed(42, "queries")
        >>> a = dist.sample(3, np.random.default_rng(seed))
        >>> b = dist.sample(3, np.random.default_rng(seed))
        >>> a == b
        True

        ``num_queries=0`` is valid and returns an empty list:

        >>> dist.sample(0, np.random.default_rng(0))
        []
    """

    query_codes: list[str]
    min_duration: float
    max_duration: float
    duration_distribution: str

    _VALID_DISTRIBUTIONS = ("uniform", "log-uniform")

    def __post_init__(self) -> None:
        if not self.query_codes:
            raise ValueError("query_codes must be non-empty")
        if self.min_duration <= 0:
            raise ValueError(
                f"min_duration must be > 0 (got {self.min_duration}); durations must be positive days "
                "and log-uniform needs positive bounds"
            )
        if self.max_duration < self.min_duration:
            raise ValueError(
                f"max_duration ({self.max_duration}) must be >= min_duration ({self.min_duration})"
            )
        if self.duration_distribution not in self._VALID_DISTRIBUTIONS:
            raise ValueError(
                f"duration_distribution must be one of {self._VALID_DISTRIBUTIONS} "
                f"(got {self.duration_distribution!r})"
            )

    @property
    def query_universe_size(self) -> int:
        """Number of distinct query codes (``len(query_codes)``)."""
        return len(self.query_codes)

    @classmethod
    def from_config(cls, cfg: DictConfig, query_codes: list[str]) -> "QueryDistribution":
        """Build from a Hydra config plus a caller-resolved ``query_codes`` list.

        Reads ``min_duration``, ``max_duration``, and ``duration_distribution`` from ``cfg``.  Code
        resolution via :func:`read_query_codes` stays outside this dataclass (no file I/O here).
        """
        return cls(
            query_codes=query_codes,
            min_duration=float(cfg.min_duration),
            max_duration=float(cfg.max_duration),
            duration_distribution=str(cfg.duration_distribution),
        )

    def sample(self, num_queries: int, rng: np.random.Generator) -> list[QuerySpec]:
        """Draw ``num_queries`` iid :class:`QuerySpec` s.

        Codes are uniform over ``[0, query_universe_size)``; durations are drawn over
        ``[min_duration, max_duration]`` per ``duration_distribution`` as **floats** (no rounding).

        The caller owns the ``rng`` and its seed — for the redesign, seed the query axis via
        ``np.random.default_rng(derive_seed(seed, "queries"))``.  Draws happen in a fixed order (codes
        then durations) so output is deterministic for a fixed ``rng``.

        Raises:
            ValueError: If ``num_queries < 0``.
        """
        if num_queries < 0:
            raise ValueError(f"num_queries must be >= 0 (got {num_queries})")
        if num_queries == 0:
            return []

        code_indices = rng.integers(0, self.query_universe_size, size=num_queries)
        if self.duration_distribution == "log-uniform":
            durations = np.exp(
                rng.uniform(np.log(self.min_duration), np.log(self.max_duration), size=num_queries)
            )
        else:  # "uniform"
            durations = rng.uniform(self.min_duration, self.max_duration, size=num_queries)

        codes = np.array(self.query_codes, dtype=object)
        selected_codes = codes[code_indices]

        return [
            QuerySpec(code=c, duration_days=float(d)) for c, d in zip(selected_codes, durations, strict=True)
        ]


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

    # ``task_id`` is intermediate-only (ignored by ``evaluate_index_df``) so it's not on
    # TaskQuerySchema; everything else comes from the schema's exported ``_name`` attrs.
    task_id_col = "task_id"
    index_cols = [
        task_id_col,
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ]

    def _empty(pt_dtype: pl.DataType, sid_dtype: pl.DataType) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                task_id_col: pl.Int64,
                TaskQuerySchema.subject_id_name: sid_dtype,
                TaskQuerySchema.prediction_time_name: pt_dtype,
                TaskQuerySchema.query_name: pl.Utf8,
                # ``Float32`` matches ``TaskQuerySchema.duration_days`` (``pa.float32``);
                # emitting Float32 directly means downstream ``TaskQuerySchema.align()``
                # calls don't need to coerce types at the write boundary.
                TaskQuerySchema.duration_days_name: pl.Float32,
            }
        )

    pt_dtype = contexts.schema.get(TaskQuerySchema.prediction_time_name, pl.Datetime("us"))
    sid_dtype = contexts.schema.get(TaskQuerySchema.subject_id_name, pl.Int64)

    if n_tasks == 0 or contexts.height == 0:
        return _empty(pt_dtype, sid_dtype)

    if contexts.height % n_tasks != 0:
        raise ValueError(f"contexts.height ({contexts.height}) must be divisible by len(tasks) ({n_tasks})")
    contexts_per_task = contexts.height // n_tasks

    task_ids = pl.Series(
        task_id_col,
        np.repeat(np.arange(n_tasks, dtype=np.int64), contexts_per_task),
    )
    query_col = pl.Series(
        TaskQuerySchema.query_name,
        np.repeat([t.code for t in tasks], contexts_per_task),
        dtype=pl.Utf8,
    )
    # Float32 matches TaskQuerySchema's duration_days type so the final labeled output
    # aligns to the schema without a type-coercion step at the write boundary.
    duration_col = pl.Series(
        TaskQuerySchema.duration_days_name,
        np.repeat([t.duration_days for t in tasks], contexts_per_task).astype(np.float32),
        dtype=pl.Float32,
    )

    return contexts.with_columns(task_ids, query_col, duration_col).select(index_cols)


def compute_max_time_per_subject(events_df: pl.DataFrame) -> pl.DataFrame:
    """Return a ``(subject_id, max_time)`` DataFrame.

    Used once per shard to turn the per-row censoring check into an O(1) lookup inside
    ``evaluate_index_df``.
    """
    return events_df.group_by(DataSchema.subject_id_name).agg(
        pl.col(DataSchema.time_name).max().alias("max_time")
    )


def evaluate_index_df(
    index_df: pl.DataFrame,
    events_df: pl.DataFrame,
    max_time_per_subject: pl.DataFrame,
) -> pl.DataFrame:
    """Label an index DataFrame with the single nullable ``boolean_value`` column via a single ``join_asof``.

    Three-valued semantics (matches ``TaskQuerySchema`` + ``LabelSchema``'s nullable
    ``boolean_value``):

        - ``boolean_value = null``: censored — ``(prediction_time + duration_days) >
          max_time[subject_id]``; observation ended before we could see whether the event
          fired.
        - ``boolean_value = True``: not censored and an event with matching ``query`` code
          fell strictly within ``(prediction_time, prediction_time + duration_days)``.
        - ``boolean_value = False``: not censored and no such event fell within the window.

    The ``>`` on event time is enforced by shifting the asof key by ``+1µs`` since datetimes are
    stored at microsecond precision, which turns ``strategy="forward"``'s ``>=`` into a strict ``>``.

    Args:
        index_df: Output of ``build_index_df``. Must have columns ``subject_id``, ``prediction_time``,
            ``query``, ``duration_days``. If ``task_id`` is present it is ignored and dropped from
            the output.
        events_df: Shard events with columns ``subject_id``, ``time``, ``code``.
        max_time_per_subject: Output of ``compute_max_time_per_subject``.

    Returns:
        DataFrame with columns ``(subject_id, prediction_time, boolean_value, query,
        duration_days)``.  ``boolean_value`` is nullable (``null`` = censored).
    """
    # Output column set lives on ``TaskQuerySchema`` — the 4 required columns plus the
    # inherited (optional) ``boolean_value`` for the collapsed label.  Defining it once
    # here avoids drift between the empty-path shape, the non-empty-path projection, and
    # the ``run_worker`` write-boundary validation.
    out_cols = [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.boolean_value_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ]

    # Empty-input fast-path: use the schema-driven empty builder in ``every_query.data``
    # rather than hand-rolling a matching polars schema dict — same column set, same
    # dtypes, guaranteed to pass ``TaskQuerySchema.align()`` downstream.
    if index_df.height == 0:
        return empty_task_query_df().select(out_cols)

    # Left side: index rows with a +1µs-shifted prediction_time for the strict-> asof key.
    left = index_df.with_columns(
        (pl.col(TaskQuerySchema.prediction_time_name) + pl.duration(microseconds=1)).alias("_pt_shifted")
    ).sort([TaskQuerySchema.subject_id_name, TaskQuerySchema.query_name, "_pt_shifted"])

    # Right side: events renamed so the join-by column name matches the left.  The events
    # frame uses ``DataSchema.code_name`` for the code column; we rename it to the
    # TaskQuerySchema ``query`` column so the asof ``by=`` key aligns on both sides.
    right = (
        events_df.rename({DataSchema.code_name: TaskQuerySchema.query_name})
        .select([TaskQuerySchema.subject_id_name, TaskQuerySchema.query_name, DataSchema.time_name])
        .sort([TaskQuerySchema.subject_id_name, TaskQuerySchema.query_name, DataSchema.time_name])
    )

    joined = left.join_asof(
        right,
        by=[TaskQuerySchema.subject_id_name, TaskQuerySchema.query_name],
        left_on="_pt_shifted",
        right_on=DataSchema.time_name,
        strategy="forward",
    )
    joined = joined.join(max_time_per_subject, on=TaskQuerySchema.subject_id_name, how="left")

    # Rows whose subject is not present in max_time_per_subject (typically a pre-seeded or
    # hand-edited index_df referencing subjects outside this shard) come out of the left join
    # with max_time=null.  A naïve comparison would produce null booleans, which is the
    # correct "censored" signal under the collapsed label semantics — so we just log a
    # warning for visibility and let the `(window_end > max_time).fill_null(True)` below
    # resolve the missing-subject case to censored.
    n_unknown = joined.filter(pl.col("max_time").is_null()).height
    if n_unknown > 0:
        logger.warning(
            "%d index_df row(s) reference subjects not present in events_df; "
            "they will be labeled as censored (boolean_value=null).",
            n_unknown,
        )

    duration_expr = pl.duration(days=pl.col(TaskQuerySchema.duration_days_name))
    window_end = pl.col(TaskQuerySchema.prediction_time_name) + duration_expr
    # `(window_end > max_time).fill_null(True)` resolves the missing-subject case to censored.
    censored = (window_end > pl.col("max_time")).fill_null(True)
    event_in_window = pl.col(DataSchema.time_name).is_not_null() & (pl.col(DataSchema.time_name) < window_end)

    # Collapsed nullable label: null when censored, else True/False from event_in_window.
    boolean_value = pl.when(censored).then(pl.lit(None, dtype=pl.Boolean)).otherwise(event_in_window)

    return joined.with_columns(boolean_value.alias(TaskQuerySchema.boolean_value_name)).select(out_cols)


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


def read_query_codes(codes_or_path: list[str] | ListConfig | str | Path | None) -> list[str]:
    """Resolve a query-code list — from default metadata, an explicit list, or a file path.

    Accepts:
    - ``None`` (load ``$PROCESSED/metadata/codes.parquet``),
    - an explicit list (from Hydra ``query_codes: [A, B, C]`` or a code-group YAML default),
    - a metadata root directory (``codes.parquet`` is expected at ``{dir}/metadata/codes.parquet``
      — matches the ``$PROCESSED`` layout), or
    - a direct path to a ``codes.parquet`` file.

    The ``.unique().sort()`` makes the returned list deterministic across workers reading
    the same metadata file (polars' default hash-based unique is order-unstable across
    DataFrame instances, which would make any seed tied to the list non-deterministic).

    Shared by both ``sample_tasks.main`` and ``sample_evaluation_tasks.main``.
    """
    if isinstance(codes_or_path, list | ListConfig):
        # Order-preserving dedup: duplicates in a user-provided ``codes: [A, A, B]``
        # list would silently inflate the task grid / sampling distribution.  The
        # parquet branch below already dedups via ``.unique().sort()``.
        seen: set[str] = set()
        return [c for c in codes_or_path if not (c in seen or seen.add(c))]
    if codes_or_path is None:
        processed = os.environ.get("PROCESSED")
        if not processed:
            raise ValueError(
                "query_codes is null and $PROCESSED is not set; pass query_codes=... or export "
                "$PROCESSED so the sampler can read $PROCESSED/metadata/codes.parquet."
            )
        p = Path(processed)
    else:
        p = Path(str(codes_or_path))
    if p.suffix in {".yaml", ".yml"}:
        import yaml

        with open(p) as f:
            data = yaml.safe_load(f)
        raw = data["codes"] if isinstance(data, dict) and "codes" in data else data
        if raw is None:
            raise ValueError(f"{p} must contain a YAML list or a mapping with a `codes` key")
        if not isinstance(raw, list | ListConfig):
            raise ValueError(f"{p} must contain a list of codes, got {type(raw).__name__}")
        seen: set[str] = set()
        return [c for c in raw if not (c in seen or seen.add(c))]
    if p.is_dir():
        p = p / "metadata" / "codes.parquet"
    return pl.read_parquet(p, columns=["code"])["code"].unique().sort().to_list()


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


def _atomic_write_parquet(df: pl.DataFrame, fp: Path) -> None:
    """Write ``df`` to ``fp`` atomically via a unique sibling tmpfile + ``os.replace``.

    The temp is always a *sibling* of ``fp`` (``_unique_tmp_path`` uses ``dir=fp.parent``) because
    ``os.replace`` requires the temp and the final path to share a filesystem.  This is the spec's
    one exception to the two-root layout (invariant 7): when Stage 4 writes a
    :func:`final_output_path`, the hidden ``.{shard}.parquet.tmp.*`` temp lands in the final-output
    split dir, not the artifacts root (which may be a different mount).  It is hidden and does not
    end in ``.parquet``, so a ``{split}/*.parquet`` glob of the final root never sees it.
    """
    tmp = _unique_tmp_path(fp)
    try:
        df.write_parquet(tmp)
        os.replace(tmp, fp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_json(obj: object, fp: Path) -> None:
    """Write ``obj`` as JSON to ``fp`` atomically via a unique sibling tmpfile + ``os.replace``.

    Same sibling-temp + ``os.replace`` pattern as :func:`_atomic_write_parquet`, so a present file is
    always complete — used for Stage 0's cache sidecar, which is the commit marker for a finished run.
    """
    tmp = _unique_tmp_path(fp)
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
        os.replace(tmp, fp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def _labels_fp(out_dir: Path, split: str, input_shard: str, task_shard: int) -> Path:
    """Resolve the labeled-parquet output path for one worker."""
    worker_id = f"{input_shard}__{task_shard:04d}"
    return out_dir / split / f"{worker_id}.parquet"


def run_worker(
    data_dir: Path,
    out_dir: Path,
    query_codes: list[str],
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
    """Run the three-stage sampling pipeline for one worker in-memory.

    The worker is a pure function of ``(data_dir, query_codes, config, seed, split, input_shard,
    task_shard)`` — no per-stage checkpoints on disk, no meta sidecar, no cache-validation
    protocol.  Determinism comes from :func:`derive_seed` separating the task and context
    axes; a rerun with identical inputs produces identical labels.

    Returns:
        The path of the labeled parquet, or ``None`` if labels already existed at that path
        and ``overwrite=False``.
    """
    labels_fp = _labels_fp(out_dir, split, input_shard, task_shard)

    if labels_fp.exists() and not overwrite:
        logger.info("Labels already exist at %s, skipping.", labels_fp)
        return None

    tasks_seed = derive_seed(seed, "tasks", task_shard)
    tasks = sample_tasks(
        n=n_tasks,
        query_codes=query_codes,
        duration_low=duration_min,
        duration_high=duration_max,
        seed=tasks_seed,
    )

    shard_path = data_dir / "data" / split / f"{input_shard}.parquet"
    events_df = _read_event_shard(shard_path)
    logger.info("Loaded %d events from %s", events_df.height, shard_path)

    contexts_seed = derive_seed(seed, "contexts", input_shard, task_shard)
    contexts = sample_contexts(
        events_df=events_df,
        n=len(tasks) * contexts_per_task,
        min_context_per_subject=min_context_per_subject,
        seed=contexts_seed,
    )
    index_df = build_index_df(tasks, contexts)

    max_time_df = compute_max_time_per_subject(events_df)
    labeled = evaluate_index_df(index_df, events_df, max_time_df)
    # Downstream MEDS dataloader does not use task_id; it is intentionally absent from the
    # published schema.
    #
    # Align the output against TaskQuerySchema at the write boundary.  ``align`` is
    # preferred over ``validate`` here because it (a) reorders columns into the schema's
    # canonical order and (b) casts mistyped columns to the declared dtypes — so even if
    # an upstream edit silently changes one type (e.g., reverting ``build_index_df`` to
    # emit ``Int64`` durations), the on-disk parquet still conforms.  Missing or extra
    # columns remain hard errors (``align`` re-raises on those).
    aligned = TaskQuerySchema.align(labeled.to_arrow())
    _atomic_write_parquet(pl.from_arrow(aligned), labels_fp)
    logger.info("Wrote %d labeled rows to %s", labeled.height, labels_fp)
    return labels_fp


def _resolve_path(cfg_value: str | None, env_var: str, name: str) -> Path:
    """Prefer an explicit cfg value; fall back to ``$env_var``; otherwise raise.

    Used by ``main`` to resolve path roots such as ``data_dir`` and ``out_dir``.
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


# ---------------------------------------------------------------------------
# Redesign (issue #203): config & path resolution for the 5-stage sampler
# ---------------------------------------------------------------------------
#
# These helpers establish the redesign's input surface (see ``redesign-spec.md``); the legacy
# pipeline above is untouched and the eventual ``main`` cutover lands with the stage issues
# (#205-#210).  Kept as pure path functions (no file I/O, no dir creation) so they are unit-testable
# without a Hydra run — same rationale as ``_resolve_path``.


def default_artifacts_dir(training_tasks_dir: Path) -> Path:
    """Return the sibling intermediate-artifacts root for ``training_tasks_dir``.

    ``training_task_artifacts_dir`` defaults to ``{parent}/{name}_artifacts`` — a *sibling* of the
    final-output root, never a child — so the final-output and intermediate trees stay disjoint and
    never-nested (spec invariant 7) and ``rm -rf`` of either cannot touch the other.

    Examples:
        >>> default_artifacts_dir(Path("/x/y/tasks"))
        PosixPath('/x/y/tasks_artifacts')
    """
    return training_tasks_dir.parent / f"{training_tasks_dir.name}_artifacts"


def resolve_training_task_paths() -> tuple[Path, Path, Path]:
    """Resolve the redesigned sampler's three path roots from env vars only.

    These roots are intentionally *not* config keys: they are machine-specific and live in ``.env``
    (loaded by ``main`` via ``load_dotenv()``), so there is no CLI/config override to reason about.

    - ``path_to_data`` — MEDS dataset root; read from ``$INTERMEDIATE``.
    - ``training_tasks_dir`` — final-output-only root; read from ``$TRAINING_TASKS_DIR`` (the user
      exports this before invoking).
    - ``training_task_artifacts_dir`` — intermediate-artifacts root.  Has no env var of its own: it
      is always :func:`default_artifacts_dir` (the ``{name}_artifacts`` sibling of
      ``training_tasks_dir``), which keeps the two output trees disjoint and never-nested by
      construction (spec invariant 7).

    Both required roots go through :func:`_resolve_path`, which raises a clear message when the env
    var is unset.

    Returns:
        ``(path_to_data, training_tasks_dir, training_task_artifacts_dir)`` as ``Path``s.
    """
    path_to_data = _resolve_path(None, "INTERMEDIATE", "path_to_data")
    training_tasks_dir = _resolve_path(None, "TRAINING_TASKS_DIR", "training_tasks_dir")
    training_task_artifacts_dir = default_artifacts_dir(training_tasks_dir)

    return path_to_data, training_tasks_dir, training_task_artifacts_dir


# ---------------------------------------------------------------------------
# Redesign (issue #204): artifact layout — two disjoint, never-nested roots
# ---------------------------------------------------------------------------
#
# Every redesigned stage reads/writes against exactly two roots (spec invariant 7):
#
#   training_tasks_dir/{split}/{shard}.parquet            <- final outputs ONLY (Stage 4)
#   training_task_artifacts_dir/{split}/                  <- all intermediates
#       _prediction_time_counts.parquet                   <- Stage 0 summary
#       _prediction_times/{shard}.parquet                 <- Stage 0 map
#       _index/{shard}.parquet                            <- Stage 3 index
#
# The two roots are disjoint and never nested (guaranteed by ``default_artifacts_dir``'s sibling
# rule), so cleanup is a single ``rm -rf`` of the artifacts root that *cannot* touch the dataset —
# no bespoke cleanup helper needed.  These are pure path functions (no I/O, no mkdir) — the writing
# stages create parents at write time via the atomic helpers, same as the legacy pipeline.
#
# The intermediate entry names are ``_``-prefixed and centralized here so (a) Stages 0/3 and any
# consumer resolve identical paths with no string drift and (b) the "final root holds nothing but
# {shard}.parquet" invariant stays auditable from one place.

PREDICTION_TIME_COUNTS_NAME = "_prediction_time_counts.parquet"
PREDICTION_TIMES_DIRNAME = "_prediction_times"
PREDICTION_TIMES_META_NAME = "_prediction_times_meta.json"
INDEX_DIRNAME = "_index"


def final_output_path(training_tasks_dir: Path, split: str, shard: str) -> Path:
    """Stage 4 final per-shard output: ``training_tasks_dir/{split}/{shard}.parquet``.

    The final-output root holds **nothing but** these files at rest — no ``_``-prefixed entries —
    so its split dir is directly glob-consumable as ``{split}/*.parquet`` (spec invariant 7).  The
    only transient siblings are Stage 4's hidden atomic-write temps, which exist solely mid-write;
    see :func:`_atomic_write_parquet`.

    Examples:
        >>> final_output_path(Path("/x/tasks"), "train", "0")
        PosixPath('/x/tasks/train/0.parquet')
    """
    return training_tasks_dir / split / f"{shard}.parquet"


def prediction_time_counts_path(training_task_artifacts_dir: Path, split: str) -> Path:
    """Stage 0 subject-level summary: ``{artifacts}/{split}/_prediction_time_counts.parquet``.

    Examples:
        >>> prediction_time_counts_path(Path("/x/tasks_artifacts"), "train")
        PosixPath('/x/tasks_artifacts/train/_prediction_time_counts.parquet')
    """
    return training_task_artifacts_dir / split / PREDICTION_TIME_COUNTS_NAME


def prediction_times_path(training_task_artifacts_dir: Path, split: str, shard: str) -> Path:
    """Stage 0 canonical map partition: ``{artifacts}/{split}/_prediction_times/{shard}.parquet``.

    Examples:
        >>> prediction_times_path(Path("/x/tasks_artifacts"), "train", "0")
        PosixPath('/x/tasks_artifacts/train/_prediction_times/0.parquet')
    """
    return training_task_artifacts_dir / split / PREDICTION_TIMES_DIRNAME / f"{shard}.parquet"


def index_path(training_task_artifacts_dir: Path, split: str, shard: str) -> Path:
    """Stage 3 partitioned index: ``{artifacts}/{split}/_index/{shard}.parquet``.

    Examples:
        >>> index_path(Path("/x/tasks_artifacts"), "train", "0")
        PosixPath('/x/tasks_artifacts/train/_index/0.parquet')
    """
    return training_task_artifacts_dir / split / INDEX_DIRNAME / f"{shard}.parquet"


def prediction_times_meta_path(training_task_artifacts_dir: Path, split: str) -> Path:
    """Stage 0 cache sidecar: ``{artifacts}/{split}/_prediction_times_meta.json``.

    The artifact paths above do not encode ``min_prediction_times_per_subject``, but Stage 0 bakes
    that eligibility threshold into the persisted artifacts (it filters before writing).  This JSON
    sidecar records the ``min`` that produced the on-disk artifacts so :func:`build_prediction_times`
    can tell a reusable cache from a stale one (a ``min`` change ⇒ auto-rebuild).  It is written last,
    so its presence is the commit marker for a complete Stage 0 run.

    Examples:
        >>> prediction_times_meta_path(Path("/x/tasks_artifacts"), "train")
        PosixPath('/x/tasks_artifacts/train/_prediction_times_meta.json')
    """
    return training_task_artifacts_dir / split / PREDICTION_TIMES_META_NAME


# ---------------------------------------------------------------------------
# Redesign (issue #205): Stage 0 — build + cache the prediction-time map
# ---------------------------------------------------------------------------
#
# Stage 0 scans a split's shards *once* and persists the canonical prediction-time indexing
# artifacts that the rest of the driver (Stages 1-3) and the labeling fan-out (Stage 4) rely on:
#
#   _prediction_times/{shard}.parquet  -- canonical map: (subject_id, prediction_time_index) -> time
#   _prediction_time_counts.parquet    -- derived subject summary (subject_id, shard, n_prediction_times)
#   _prediction_times_meta.json        -- cache sidecar recording the min used (commit marker)
#
# The indexing space is distinct ``(subject_id, time)`` rows (invariant 1); the index is a gapless
# zero-based dense rank per subject (invariant 2) so Stage 2's array-bounded draw is in-bounds.


def _read_prediction_time_shard(file_path: str | Path, shard: str) -> pl.DataFrame:
    """Read one shard's distinct ``(subject_id, time)`` rows, tagged with ``shard``.

    Reads only ``subject_id``/``time`` (Stage 0 never needs event payloads).  Null-``time`` rows (e.g.
    MEDS static measurements like demographics) are dropped: they are not valid prediction times for
    the downstream ``+1µs`` strict-after asof rule, and — because ``sort(["subject_id", "time"])``
    places nulls first — an unfiltered null would otherwise claim ``prediction_time_index = 0`` and
    inflate ``n_prediction_times`` past the eligibility boundary.  ``time`` is cast to
    ``pl.Datetime("us")`` for the same reason as :func:`_read_event_shard`: the label window uses a
    ``+1µs`` strict-after shift downstream, and a coarser precision would silently round it to zero.
    Dedups to distinct ``(subject_id, time)`` so the per-subject row count is a count of *prediction
    times*, not events.
    """
    return (
        pl.read_parquet(file_path, columns=["subject_id", "time"])
        .filter(pl.col("time").is_not_null())
        .with_columns(pl.col("time").cast(pl.Datetime("us")))
        .unique()
        .with_columns(pl.lit(shard).alias("shard"))
    )


def _split_shards(path_to_data: Path, split: str) -> list[str]:
    """Discover shard names for a split: the parquet file stems under ``path_to_data/data/{split}``.

    Sorted for deterministic iteration order.
    """
    split_dir = path_to_data / "data" / split
    return sorted(p.stem for p in split_dir.glob("*.parquet"))


def _prediction_time_cache_valid(
    training_task_artifacts_dir: Path,
    split: str,
    min_prediction_times_per_subject: int,
) -> bool:
    """True iff the cached Stage 0 artifacts are reusable for this ``(split, min)``.

    The sidecar is written last as the commit marker and is treated as *authoritative*: rather than
    re-scanning the ``subject_id`` column across every ``_prediction_times/`` partition, validation
    cross-checks the cheap counts parquet against the subject/row totals recorded in the sidecar.

    Reuse requires *all* of:
    - the sidecar exists, parses, and its recorded ``min`` matches the requested ``min``;
    - the counts parquet exists and every map partition named in the sidecar exists;
    - the counts parquet agrees with the sidecar totals: ``counts.height == meta["n_subjects"]`` and
      ``sum(n_prediction_times) == meta["n_prediction_time_rows"]``.

    Any mismatch means the sidecar is stale/corrupt relative to ``_prediction_times/`` and Stage 0
    must rebuild.  Read-only: never writes or deletes.
    """
    meta_fp = prediction_times_meta_path(training_task_artifacts_dir, split)
    counts_fp = prediction_time_counts_path(training_task_artifacts_dir, split)
    if not meta_fp.exists() or not counts_fp.exists():
        return False

    try:
        meta = json.loads(meta_fp.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if meta.get("min_prediction_times_per_subject") != min_prediction_times_per_subject:
        return False

    shards = meta.get("shards")
    if not isinstance(shards, list) or not all(
        prediction_times_path(training_task_artifacts_dir, split, s).exists() for s in shards
    ):
        return False

    counts = pl.read_parquet(counts_fp)
    if counts.height != meta.get("n_subjects"):
        return False
    return int(counts["n_prediction_times"].sum()) == meta.get("n_prediction_time_rows")


def _assert_no_subject_spans_shards(distinct: pl.DataFrame) -> None:
    """Invariant 4: a subject lives in exactly one shard (Stage 4 derives ``max_time`` from a single
    shard).  Raises ``ValueError`` naming the offenders ``{subject_id: shards}`` on violation.
    """
    spanning = (
        distinct.group_by("subject_id")
        .agg(pl.col("shard").n_unique().alias("n_shards"), pl.col("shard").unique().alias("shards"))
        .filter(pl.col("n_shards") > 1)
        .sort("subject_id")
    )
    if spanning.height == 0:
        return
    offenders = {row["subject_id"]: sorted(row["shards"]) for row in spanning.iter_rows(named=True)}
    raise ValueError(
        "Subjects span multiple shards (invariant 4 violation); each must live in exactly one "
        f"shard. Offenders {{subject_id: shards}}: {offenders}"
    )


def build_prediction_times(
    path_to_data: Path,
    training_task_artifacts_dir: Path,
    split: str,
    min_prediction_times_per_subject: int,
    overwrite: bool = False,
) -> int:
    """Stage 0: build (and cache) the canonical prediction-time map + derived subject summary.

    Scans ``path_to_data/data/{split}/*.parquet`` once, deduping to distinct ``(subject_id, time)``
    rows, assigns each subject a gapless zero-based ``prediction_time_index`` over its
    ascending-sorted distinct times, filters to eligible subjects
    (``n_prediction_times >= min_prediction_times_per_subject + 1`` — the ``+1`` keeps Stage 2's
    ``[min, n)`` draw range non-empty), and writes:

    - ``_prediction_times/{shard}.parquet`` -- canonical map ``(subject_id, prediction_time_index, time)``.
    - ``_prediction_time_counts.parquet``   -- summary ``(subject_id, shard, n_prediction_times)``,
      sorted by ``subject_id`` so its row position is the ``subject_idx`` Stage 2 gathers by.
    - ``_prediction_times_meta.json``       -- cache sidecar (written last as the commit marker).

    Enforces invariant 4 (a subject may not span shards) as a hard error.  Reuses a valid cache
    unless ``overwrite`` is set; a change in ``min_prediction_times_per_subject`` invalidates the
    cache automatically via the sidecar.

    Returns:
        ``patient_universe_size`` — the number of eligible subjects (rows in the counts summary).
    """
    if not overwrite and _prediction_time_cache_valid(
        training_task_artifacts_dir, split, min_prediction_times_per_subject
    ):
        counts_fp = prediction_time_counts_path(training_task_artifacts_dir, split)
        height = pl.read_parquet(counts_fp).height
        logger.info("Stage 0: reusing cached prediction-time artifacts (%d subjects).", height)
        return height

    shards = _split_shards(path_to_data, split)
    if not shards:
        raise FileNotFoundError(
            f"No shards found under {path_to_data / 'data' / split}; expected {{i}}.parquet files."
        )

    # distinct columns: (subject_id, time, shard) -- one row per distinct (subject_id, time).
    distinct = pl.concat(
        [
            _read_prediction_time_shard(path_to_data / "data" / split / f"{shard}.parquet", shard)
            for shard in shards
        ]
    )

    _assert_no_subject_spans_shards(distinct)

    # Gapless zero-based index over each subject's ascending distinct times.  No within-subject ties
    # (step above deduped), so int_range is identical to a dense rank and cheaper (invariant 2).
    # prediction_times columns: (subject_id, time, shard, prediction_time_index).
    prediction_times = distinct.sort(["subject_id", "time"]).with_columns(
        pl.int_range(pl.len()).over("subject_id").alias("prediction_time_index")
    )

    # counts columns: (subject_id, shard, n_prediction_times) -- one row per subject.
    counts = prediction_times.group_by("subject_id").agg(
        pl.col("shard").first(),
        pl.len().alias("n_prediction_times"),
    )

    eligible = counts.filter(pl.col("n_prediction_times") >= min_prediction_times_per_subject + 1).sort(
        "subject_id"
    )

    # keep only the prediction_times rows whose subject_id appears in eligible, dropping the rest
    prediction_times = prediction_times.join(
        eligible.select("subject_id"),
        on="subject_id",
        how="semi",
    )

    # Rebuild from scratch: drop any stale partitions so a shrunken shard set leaves no orphans.
    map_dir = training_task_artifacts_dir / split / PREDICTION_TIMES_DIRNAME
    if map_dir.exists():
        shutil.rmtree(map_dir)

    # Write one prediction_time index parquet per shard, mirroring the source shard layout so each output
    # partition lines up with its input shard.
    for shard, part in prediction_times.group_by("shard"):
        shard_name = shard[0] if isinstance(shard, tuple) else shard
        _atomic_write_parquet(
            part.select(["subject_id", "prediction_time_index", "time"]).sort(
                ["subject_id", "prediction_time_index"]
            ),
            prediction_times_path(training_task_artifacts_dir, split, str(shard_name)),
        )

    # INVARIANT: row position in this table defines subject_idx for Stage 2. `eligible` is sorted by
    # subject_id above; preserve that ordering on any refactor -- reordering rows silently changes the
    # subject_idx -> subject_id mapping (the sampling universe).
    _atomic_write_parquet(
        eligible.select(["subject_id", "shard", "n_prediction_times"]),
        prediction_time_counts_path(training_task_artifacts_dir, split),
    )

    # Sidecar last = commit marker. Only shards that survived the eligibility filter have partitions.
    # ``n_subjects``/``n_prediction_time_rows`` make the sidecar authoritative for cache validation,
    # so reuse never re-scans the subject_id column across the map partitions.
    written_shards = sorted(eligible["shard"].unique().to_list())
    _atomic_write_json(
        {
            "min_prediction_times_per_subject": min_prediction_times_per_subject,
            "split": split,
            "shards": written_shards,
            "n_subjects": eligible.height,
            "n_prediction_time_rows": prediction_times.height,
        },
        prediction_times_meta_path(training_task_artifacts_dir, split),
    )

    logger.info(
        "Stage 0: built prediction-time artifacts for split=%s (%d eligible subjects across %d shards).",
        split,
        eligible.height,
        len(written_shards),
    )
    return eligible.height


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_tasks_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point.

    Loads ``.env`` via python-dotenv before resolving paths, following the repo convention where
    ``$INTERMEDIATE`` / ``$PROCESSED`` / ``$TASK_DIR`` live in a gitignored ``.env`` file rather
    than being exported by the user.  Path fallbacks: ``cfg.data_dir`` falls back to
    ``$INTERMEDIATE`` and ``cfg.out_dir`` to ``$TASK_DIR``.  Query codes are resolved by
    :func:`read_query_codes`, which falls back to ``$PROCESSED`` when ``cfg.query_codes`` is null.

    See :func:`run_worker` for the per-worker pipeline.
    """
    # Late import so `load_dotenv()` doesn't run at module import time (which would be an
    # unexpected side effect for programmatic callers / tests of the pure primitives).
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = _resolve_path(cfg.get("data_dir"), "INTERMEDIATE", "data_dir")
    out_dir = _resolve_path(cfg.get("out_dir"), "TASK_DIR", "out_dir")
    query_codes = read_query_codes(cfg.get("query_codes"))

    run_worker(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=query_codes,
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
