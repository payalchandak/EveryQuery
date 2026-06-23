"""Sampling-first task label generator for pre-training (redesigned 5-stage pipeline).

The whole pipeline runs from one console script — ``EQ_generate_training_tasks`` — in a single
process on a single node (see ``redesign-spec.md`` and epic #202):

    Stage 0  build + cache the canonical prediction-time map and subject counts (scan shards once)
    Stage 1  sample ``num_queries`` queries ``(code, duration_days)``
    Stage 2  sample ``N = num_queries * num_contexts_per_query`` patient contexts
    Stage 3  resolve ``prediction_time_index -> prediction_time``, zip, write per-shard index
    Stage 4  label each index shard independently and write the final dataset parquet

Stages 0-3 run sequentially in the driver (:func:`main`); Stage 4 fans out one worker per shard via
``concurrent.futures.ProcessPoolExecutor`` (:func:`label_one_shard`).  Workers are passed shard ids,
never DataFrames — each does its own parquet I/O and atomic output, so they never contend.

Determinism comes from :func:`~every_query.utils.seeds.derive_seed` splitting the query and context
axes; labeling is a pure function of the resolved index partition plus shard events.  Atomic writes
make Stage 4 restartable: a present ``{shard}.parquet`` is always complete and is skipped on rerun
unless ``overwrite=true``.
"""

import os

# Pin polars to a single thread BEFORE importing polars (or anything that transitively imports it —
# meds, every_query.data.schema).  Stage 4 workers inherit this env.  With 200+ shards, process-level
# fan-out already saturates cores; intra-op polars threads on top would oversubscribe (N x cores).
# ``setdefault`` so an operator who exports a different value (e.g. for debugging) is respected.  A
# transitive ``import polars`` above this line would silently defeat the setting (see #210).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import json
import logging
import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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
class QuerySpec:
    """A single sampled query: one code and one float prediction-window duration (in days).

    The Stage 1 output (see ``redesign-spec.md``).  ``duration_days`` is a **float** — durations are
    not rounded to whole days.
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
    def from_config(cls, cfg: DictConfig) -> "QueryDistribution":
        """Build from a Hydra config plus a caller-resolved ``query_codes`` list.

        Reads ``min_duration``, ``max_duration``, and ``duration_distribution`` from ``cfg``.  Code
        resolution via :func:`read_query_codes` stays outside this dataclass (no file I/O here).
        """
        query_codes = read_query_codes(cfg.query_codes)

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
# Stage 2 (redesign): vectorized patient-context sampling
# ---------------------------------------------------------------------------


def sample_patient_contexts(
    prediction_time_counts: pl.DataFrame,
    n: int,
    min_prediction_times_per_subject: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Draw ``n`` patient contexts ``(subject_id, shard, prediction_time_index)`` (redesign Stage 2).

    A patient context is a ``(subject_idx, prediction_time_index)`` pair; the timestamp itself is
    resolved later (Stage 3).  Sampling is two vectorized RNG draws over the Stage 0
    ``_prediction_time_counts`` table — whose **row position is ``subject_idx``** (the table is sorted
    by ``subject_id``), so per-subject ``n_prediction_times`` is gathered by row index, not a dict
    lookup.  ``patient_universe_size`` is *derived* as the table height, not passed in.

    The caller owns the ``rng`` and its seed — for the redesign, seed the context axis via
    ``np.random.default_rng(derive_seed(seed, "contexts"))``.  Draws happen in a fixed order — all
    ``subject_idx`` (Step A), then all ``prediction_time_index`` (Step B) — so output is deterministic
    for a fixed ``rng`` (spec invariant 5).

    Args:
        prediction_time_counts: Stage 0 summary, one row per eligible subject with columns
            ``subject_id``, ``shard``, ``n_prediction_times``, sorted by ``subject_id`` so row
            position equals ``subject_idx``.
        n: Number of contexts to draw (``N = num_queries * num_contexts_per_query``).  Subjects are
            drawn **with replacement** (``N`` typically exceeds the eligible universe; iid-ness matters
            more than coverage, duplicate rows allowed).
        min_prediction_times_per_subject: Minimum prior prediction times required.  Sets the draw
            ``low``; since ``prediction_time_index`` is a zero-based rank, ``low`` selects the
            ``(low + 1)``-th distinct timestamp with exactly ``low`` before it (spec invariant 2).
        rng: Caller-owned NumPy generator.

    Returns:
        Length-``n`` ``DataFrame`` with columns ``(subject_id, shard, prediction_time_index)``;
        ``subject_id``/``shard`` keep the table's dtypes and ``prediction_time_index`` is ``Int64``
        (matches the Stage 0 ``_prediction_times`` map for the Stage 3 join).

    Raises:
        ValueError: If ``n < 0``, or ``n > 0`` while ``prediction_time_counts`` is empty, or a counts
            row has ``n_prediction_times <= min_prediction_times_per_subject`` (a Stage 0 eligibility
            violation ⇒ stale/corrupt counts table, which would make the Step B range empty).

    Examples:
        >>> import numpy as np
        >>> import polars as pl
        >>> from every_query.utils.seeds import derive_seed
        >>> counts = pl.DataFrame(
        ...     {
        ...         "subject_id": [10, 20, 30],
        ...         "shard": ["0", "0", "1"],
        ...         "n_prediction_times": [60, 80, 120],
        ...     }
        ... )
        >>> rng = np.random.default_rng(derive_seed(0, "contexts"))
        >>> ctx = sample_patient_contexts(counts, n=100, min_prediction_times_per_subject=50, rng=rng)
        >>> ctx.height
        100
        >>> ctx.columns
        ['subject_id', 'shard', 'prediction_time_index']

        Every draw lands in ``[min, n_prediction_times)`` for its subject:

        >>> checked = ctx.join(counts, on="subject_id", how="left")
        >>> bool(
        ...     (checked["prediction_time_index"] >= 50).all()
        ...     and (checked["prediction_time_index"] < checked["n_prediction_times"]).all()
        ... )
        True

        Determinism — same seed yields identical output:

        >>> a = sample_patient_contexts(counts, 32, 50, np.random.default_rng(derive_seed(7, "contexts")))
        >>> b = sample_patient_contexts(counts, 32, 50, np.random.default_rng(derive_seed(7, "contexts")))
        >>> a.equals(b)
        True

        ``n=0`` is valid and returns an empty, correctly-typed frame:

        >>> sample_patient_contexts(counts, 0, 50, rng).height
        0
    """
    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n})")

    sid_dtype = prediction_time_counts.schema.get("subject_id", pl.Int64)
    shard_dtype = prediction_time_counts.schema.get("shard", pl.Utf8)

    if n == 0:
        return pl.DataFrame(
            schema={
                "subject_id": sid_dtype,
                "shard": shard_dtype,
                "prediction_time_index": pl.Int64,
            }
        )

    patient_universe_size = prediction_time_counts.height
    if patient_universe_size == 0:
        raise ValueError(
            f"prediction_time_counts is empty but n={n} contexts were requested; Stage 0 should "
            "produce a non-empty eligible subject universe"
        )

    # Step A — subject indices (consumed first), with replacement.  One row-gather over the table
    # (same indices for every column) instead of three per-column gathers.  Drawn via `rng.integers`
    # rather than `prediction_time_counts.sample(..., with_replacement=True)` — both do the same
    # index-then-gather, but `pl.DataFrame.sample` takes only an int seed, not the caller-owned numpy
    # Generator.  Using it would fork RNG state away from Step B, breaking the single-generator
    # fixed-order determinism contract (spec invariant 5).
    subject_idx = rng.integers(0, patient_universe_size, size=n)
    sampled = prediction_time_counts[subject_idx]
    subject_id = sampled["subject_id"]
    shard = sampled["shard"]
    n_prediction_times = sampled["n_prediction_times"].to_numpy()

    # Stage 0 eligibility guarantees a non-empty Step B range; a violation means a stale/corrupt
    # counts table, which would make `rng.integers(low, high)` illegal (low >= high) for some rows.
    if not (n_prediction_times > min_prediction_times_per_subject).all():
        raise ValueError(
            "prediction_time_counts contains a subject with "
            f"n_prediction_times <= min_prediction_times_per_subject ({min_prediction_times_per_subject}); "
            "Stage 0 eligibility requires n_prediction_times > min_prediction_times_per_subject — "
            "the counts table is stale or corrupt and must be rebuilt from _prediction_times/"
        )

    # Step B — prediction-time indices (consumed second): one array-bounded draw, one per row in row
    # order.  `high` exclusive ⇒ the subject's last prediction time (index n-1) is eligible.
    prediction_time_index = rng.integers(low=min_prediction_times_per_subject, high=n_prediction_times)

    return pl.DataFrame(
        {
            "subject_id": subject_id,
            "shard": shard,
            "prediction_time_index": pl.Series(prediction_time_index, dtype=pl.Int64),
        }
    )


# ---------------------------------------------------------------------------
# Pure primitives
# ---------------------------------------------------------------------------


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

        - ``boolean_value = True``: an event with matching ``query`` code fell strictly
          within ``(prediction_time, prediction_time + duration_days]``, even if the window
          extends past ``max_time[subject_id]``.
        - ``boolean_value = null``: no matching event in the observed window **and**
          ``(prediction_time + duration_days) > max_time[subject_id]`` (censored).
        - ``boolean_value = False``: no matching event and the full window is observed.

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

    duration_expr = pl.duration(seconds=pl.col(TaskQuerySchema.duration_days_name) * 86_400)
    window_end = pl.col(TaskQuerySchema.prediction_time_name) + duration_expr
    # `(window_end > max_time).fill_null(True)` resolves the missing-subject case to censored.
    censored = (window_end > pl.col("max_time")).fill_null(True)
    event_in_window = pl.col(DataSchema.time_name).is_not_null() & (
        pl.col(DataSchema.time_name) <= window_end
    )

    # Occurrence takes priority: True even if the window extends past max_time (spec §Stage 4).
    boolean_value = (
        pl.when(event_in_window)
        .then(pl.lit(True))
        .when(censored)
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.lit(False))
    )

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
    # The random token sits *after* ``.tmp.`` (prefix=".{name}.tmp.", empty suffix) so the produced
    # name is ``.{name}.tmp.<random>`` — matching the ``.{shard}.parquet.tmp.*`` glob in
    # ``_clean_stale_temps``.  Putting ``tmp`` in a ``suffix=".tmp"`` instead would emit
    # ``.{name}.<random>.tmp``, which that glob never matches, so orphaned temps would never be
    # cleaned (the random token lands between ``.parquet.`` and ``.tmp``).
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{fp.name}.tmp.",
        suffix="",
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
        f"{name} must be set: pass {name}=... on the CLI, set it in sample_training_tasks_config.yaml, "
        f"or export ${env_var} (or define it in .env — sample_tasks calls load_dotenv())."
    )


# ---------------------------------------------------------------------------
# Config & path resolution for the 5-stage sampler (issue #203)
# ---------------------------------------------------------------------------
#
# These helpers establish the sampler's input surface (see ``redesign-spec.md``).  Kept as pure path
# functions (no file I/O, no dir creation) so they are unit-testable without a Hydra run — same
# rationale as ``_resolve_path``.


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


def resolve_training_task_paths(cfg: DictConfig | None = None) -> tuple[Path, Path, Path]:
    """Resolve the redesigned sampler's three path roots (``override > env var > raise``).

    The two input roots are machine-specific.  Each takes an optional Hydra override
    (``cfg.data_dir`` / ``cfg.out_dir``) and falls back to its env var when the override is null —
    the same ``override > env > raise`` contract that ``EQ_process_data`` and the sibling
    :mod:`sample_evaluation_tasks` use, so ``.env`` keeps working for cluster runs while CLI/test
    callers can point the sampler at an arbitrary directory.

    - ``path_to_data`` — MEDS dataset root; ``cfg.data_dir`` else ``$INTERMEDIATE``.
    - ``training_tasks_dir`` — final-output-only root; ``cfg.out_dir`` else ``$TRAINING_TASKS_DIR``.
    - ``training_task_artifacts_dir`` — intermediate-artifacts root.  Has no key/env var of its own:
      it is always :func:`default_artifacts_dir` (the ``{name}_artifacts`` sibling of
      ``training_tasks_dir``), which keeps the two output trees disjoint and never-nested by
      construction (spec invariant 7).

    Both required roots go through :func:`_resolve_path`, which raises a clear message when neither
    the override nor the env var is set.

    Args:
        cfg: Resolved sampler config; ``cfg.data_dir`` / ``cfg.out_dir`` (when non-null) override the
            env vars.  ``None`` (the default) resolves from env vars only.

    Returns:
        ``(path_to_data, training_tasks_dir, training_task_artifacts_dir)`` as ``Path``s.
    """
    data_dir_override = cfg.get("data_dir") if cfg is not None else None
    out_dir_override = cfg.get("out_dir") if cfg is not None else None
    path_to_data = _resolve_path(data_dir_override, "INTERMEDIATE", "data_dir")
    training_tasks_dir = _resolve_path(out_dir_override, "TRAINING_TASKS_DIR", "out_dir")
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
# stages create parents at write time via the atomic helpers.
#
# The intermediate entry names are ``_``-prefixed and centralized here so (a) Stages 0/3 and any
# consumer resolve identical paths with no string drift and (b) the "final root holds nothing but
# {shard}.parquet" invariant stays auditable from one place.

PREDICTION_TIME_COUNTS_NAME = "_prediction_time_counts.parquet"
PREDICTION_TIMES_DIRNAME = "_prediction_times"
PREDICTION_TIMES_META_NAME = "_prediction_times_meta.json"
INDEX_DIRNAME = "_index"
LABELED_DIRNAME = "_labeled"


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


def labeled_fingerprint_path(training_task_artifacts_dir: Path, split: str, shard: str) -> Path:
    """Stage 4 per-shard label-provenance sidecar: ``{artifacts}/{split}/_labeled/{shard}.json``.

    Records the :func:`_index_fingerprint` of the Stage 3 index partition that produced the current
    final ``{shard}.parquet`` output.  It lives in the **artifacts** root, never the final-output root,
    so the final root keeps holding nothing but ``{shard}.parquet`` (invariant 7).  Stage 4 compares
    the recorded fingerprint against the current index's fingerprint to decide whether an existing
    output is still valid (skip) or was produced by a different index (stale ⇒ relabel).

    Examples:
        >>> labeled_fingerprint_path(Path("/x/tasks_artifacts"), "train", "0")
        PosixPath('/x/tasks_artifacts/train/_labeled/0.json')
    """
    return training_task_artifacts_dir / split / LABELED_DIRNAME / f"{shard}.json"


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
# Redesign (issue #209): Stage 4 — per-shard labeling worker
# ---------------------------------------------------------------------------
#
# Stage 4 is the parallelizable labeling fan-out.  Each worker processes one shard
# independently: it reads the Stage 3 index partition (which already carries a resolved
# ``prediction_time`` — invariant 3) plus the shard's event payload, labels via
# ``evaluate_index_df``, aligns to ``TaskQuerySchema``, and writes the final dataset
# shard atomically.  Workers are fully independent (own index partition, own event file,
# own output file) so the orchestrator (#210) fans them out with ProcessPoolExecutor.


def _index_fingerprint(index_df: pl.DataFrame) -> str:
    """Serialization-independent fingerprint of a Stage 3 index partition's *logical* content.

    Stage 3 unconditionally rebuilds (``rmtree`` + rewrite) the index every run, so an existence-only
    Stage 4 skip would keep stale labels after a sampling-config change (different ``seed`` /
    ``num_queries`` / ``query_codes`` / duration params) when ``overwrite=False``.  This fingerprint
    lets the worker tell "same index as last time" (skip) from "different index" (relabel).

    Built from polars' vectorized :meth:`~polars.DataFrame.hash_rows` summed over rows, combined with
    the row count.  Summing is order-independent and counts duplicates, so the fingerprint depends only
    on the *multiset* of index rows — exactly what determines the labels — and is stable across the
    parquet re-serialization Stage 3 performs each run (a byte-hash of the parquet file would not be,
    which would break ``overwrite=False`` idempotency).  Not collision-proof, but a collision only
    causes an over-skip on content that already hashed and counted identically.
    """
    row_hash_sum = int(index_df.hash_rows(seed=0).sum()) if index_df.height else 0
    return f"{index_df.height}:{row_hash_sum & 0xFFFFFFFFFFFFFFFF:016x}"


def _clean_stale_temps(out_dir: Path, shard: str) -> int:
    """Remove orphaned atomic-write temp files for ``shard`` in ``out_dir``.

    Crashed workers leave ``.{shard}.parquet.tmp.*`` files that are never renamed into
    place.  Cleaning them on worker entry prevents accumulation across retries.

    Returns the number of files removed.
    """
    removed = 0
    for tmp in out_dir.glob(f".{shard}.parquet.tmp.*"):
        tmp.unlink(missing_ok=True)
        removed += 1
    return removed


def label_one_shard(
    shard: str,
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> tuple[str, str]:
    """Label one shard's index partition and write the final dataset parquet.

    This is the Stage 4 worker function — called once per shard, either directly or via
    a ``ProcessPoolExecutor`` in the orchestrator (#210).

    Args:
        shard: Shard name (e.g. ``"0"``).
        index_dir: Directory containing Stage 3 index partitions (``_index/``).
        data_dir: Directory containing the MEDS event shards (``data/{split}/``).
        out_dir: Directory for final output (``training_tasks_dir/{split}/``).
        overwrite: If ``True``, relabel even if the output already exists.

    Returns:
        ``(shard, status)`` where status is ``"skipped"`` or ``"labeled"``.

    The skip is keyed on the :func:`_index_fingerprint` of the current Stage 3 index partition, not
    on mere output existence: an existing output is reused only when a recorded fingerprint matches
    the current index (a genuine restart of the *same* run).  A changed sampling config rewrites the
    index with a different fingerprint, so the stale output is relabeled even under ``overwrite=False``.
    A missing/unreadable fingerprint (e.g. a pre-fingerprint output, or a crash between the parquet
    and sidecar writes) is treated as stale ⇒ relabel.
    """
    final = out_dir / f"{shard}.parquet"
    fingerprint_fp = index_dir.parent / LABELED_DIRNAME / f"{shard}.json"

    index_df = pl.read_parquet(index_dir / f"{shard}.parquet")
    current_fingerprint = _index_fingerprint(index_df)

    if not overwrite and final.exists():
        try:
            recorded = json.loads(fingerprint_fp.read_text()).get("index_fingerprint")
        except (OSError, json.JSONDecodeError, AttributeError):
            recorded = None
        if recorded == current_fingerprint:
            return shard, "skipped"

    _clean_stale_temps(out_dir, shard)

    events_df = _read_event_shard(data_dir / f"{shard}.parquet")
    max_time = compute_max_time_per_subject(events_df)

    labeled = evaluate_index_df(index_df, events_df, max_time)
    aligned = TaskQuerySchema.align(labeled.to_arrow())

    _atomic_write_parquet(pl.from_arrow(aligned), final)
    # Record the index fingerprint *after* the output is committed so a present sidecar always
    # describes a present, complete output (the parquet is the value, the sidecar is its provenance).
    _atomic_write_json({"index_fingerprint": current_fingerprint}, fingerprint_fp)
    return shard, "labeled"


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
    (``n_prediction_times > min_prediction_times_per_subject`` — strictly more than the minimum,
    which keeps Stage 2's ``[min, n)`` draw range non-empty), and writes:

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
    eligible = counts.filter(pl.col("n_prediction_times") > min_prediction_times_per_subject).sort(
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

    return eligible.height


# ---------------------------------------------------------------------------
# Redesign (issue #208): Stage 3 — resolve prediction times and build index
# ---------------------------------------------------------------------------
#
# Stage 3 is the single index-resolution point (spec invariant 3): it zips Stage 1 queries with
# Stage 2 contexts, resolves ``prediction_time_index → prediction_time`` per shard via the Stage 0
# ``_prediction_times/`` map, and writes the partitioned ``_index/{shard}.parquet`` files that
# Stage 4 consumes directly.  Stage 4 never resolves indices — it receives timestamps.


def build_index(
    queries: list[QuerySpec],
    contexts: pl.DataFrame,
    training_task_artifacts_dir: Path,
    split: str,
    num_contexts_per_query: int,
) -> int:
    """Stage 3: zip queries with contexts, resolve prediction times, write partitioned index.

    ``np.repeat`` s the Stage 1 queries ``num_contexts_per_query`` times and zips them with the
    Stage 2 contexts (length ``len(queries) * num_contexts_per_query``).  For each shard, joins
    against the Stage 0 ``_prediction_times/{shard}.parquet`` map on
    ``(subject_id, prediction_time_index)`` to resolve the timestamp, then writes
    ``_index/{shard}.parquet`` with columns
    ``["subject_id", "prediction_time", "query", "duration_days"]``.

    The join is **per shard** (not one global join) so the driver holds only one shard's
    payload-free map at a time, keeping memory flat.  The join is total (same eligibility as
    Stage 2's bound) — a null ``prediction_time`` after the join is a hard error.

    Args:
        queries: Stage 1 output — ``num_queries`` :class:`QuerySpec` instances.
        contexts: Stage 2 output — ``(subject_id, shard, prediction_time_index)`` frame of
            length ``len(queries) * num_contexts_per_query``.
        training_task_artifacts_dir: Intermediate-artifacts root.
        split: Dataset split name (e.g. ``"train"``).
        num_contexts_per_query: Number of patient contexts per query (the ``M`` multiplier).

    Returns:
        ``n_shards`` — the number of index partitions written (0 when there are no queries/contexts).

    Raises:
        ValueError: If ``contexts.height != len(queries) * num_contexts_per_query``, or if
            any context fails to resolve a prediction time (null after join).
    """
    n_queries = len(queries)
    expected = n_queries * num_contexts_per_query

    if contexts.height != expected:
        raise ValueError(
            f"contexts.height ({contexts.height}) must equal "
            f"len(queries) * num_contexts_per_query ({n_queries} * {num_contexts_per_query} = {expected})"
        )

    if n_queries == 0 or contexts.height == 0:
        return 0

    query_col = pl.Series(
        TaskQuerySchema.query_name,
        np.repeat([q.code for q in queries], num_contexts_per_query),
        dtype=pl.Utf8,
    )
    duration_col = pl.Series(
        TaskQuerySchema.duration_days_name,
        np.repeat([q.duration_days for q in queries], num_contexts_per_query).astype(np.float32),
        dtype=pl.Float32,
    )

    combined = contexts.with_columns(query_col, duration_col)

    index_dir = training_task_artifacts_dir / split / INDEX_DIRNAME
    if index_dir.exists():
        shutil.rmtree(index_dir)

    output_cols = [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ]

    join_keys = ["subject_id", "prediction_time_index"]

    n_shards = 0
    # One ``read_parquet`` per shard is deliberate: the map is read exactly once and the driver
    # holds only the current shard's payload-free map, keeping memory flat as shard count grows.
    # Caching every shard's map would trade that guarantee away for no IO win within a single call.
    # ``group_by`` (not ``partition_by``) for the same reason — it streams one group at a time
    # rather than materializing every shard's frame up front.

    # Sort + ``maintain_order=True`` make the order shards are processed (and logged) deterministic;
    # ``group_by`` alone is not order-preserving.  Output content is unaffected (shards are
    # independent), but the deterministic order keeps reruns and logs stable.
    combined = combined.sort("shard")
    for shard_key, shard_group in combined.group_by("shard", maintain_order=True):
        (shard_name,) = shard_key
        pt_map = pl.read_parquet(prediction_times_path(training_task_artifacts_dir, split, str(shard_name)))

        # Guard against silent all-null joins from join-key dtype drift between the Stage 2
        # contexts and the Stage 0 map (e.g. Int64 vs UInt32 ``subject_id``). A mismatch is an
        # upstream bug, so fail loudly rather than papering over it with a cast.
        for key in join_keys:
            ctx_dtype, map_dtype = shard_group.schema[key], pt_map.schema[key]
            if ctx_dtype != map_dtype:
                raise ValueError(
                    f"Join key {key!r} dtype mismatch: contexts has {ctx_dtype}, the "
                    f"_prediction_times map has {map_dtype}. A mismatch silently produces null "
                    "prediction_times; fix the dtype upstream."
                )

        joined = shard_group.join(
            pt_map,
            on=join_keys,
            how="left",
        ).rename({"time": TaskQuerySchema.prediction_time_name})

        # Left-join-then-raise (rather than an inner join that silently drops rows) keeps the
        # failure explicit and debuggable. The join is total by design — same eligibility bound
        # as Stage 2 — so any null is a hard error, and a small sample helps pinpoint the cause.
        null_rows = joined.filter(pl.col(TaskQuerySchema.prediction_time_name).is_null())
        if null_rows.height > 0:
            sample = null_rows.select(join_keys).head(5).to_dicts()
            raise ValueError(
                f"Shard {shard_name}: {null_rows.height} contexts have null prediction_time "
                "after join. The _prediction_times map may be stale or contexts reference invalid "
                f"(subject_id, prediction_time_index) pairs. Sample of offending rows: {sample}"
            )

        _atomic_write_parquet(
            joined.select(output_cols),
            index_path(training_task_artifacts_dir, split, str(shard_name)),
        )
        n_shards += 1

    return n_shards


# ---------------------------------------------------------------------------
# Redesign (issue #210): orchestration & parallelism
# ---------------------------------------------------------------------------


def resolve_workers(max_workers: int | None = None) -> int:
    """Resolve the Stage 4 worker-pool size: cores-on-this-node, optionally capped downward.

    Reads cores from ``$SLURM_CPUS_PER_TASK`` → ``$SLURM_CPUS_ON_NODE`` (the cores allocated to this
    task/node) → ``os.cpu_count()`` (a research server with no SLURM).  ``$SLURM_NTASKS``/``srun`` are
    deliberately *not* used: ``ProcessPoolExecutor`` forks workers on the driver's node only, so the
    correct knob is cores-on-this-node, not whole-allocation task count (see ``redesign-spec.md``).

    ``max_workers`` caps the result **downward only** (``min(cores, max_workers)``) — set it when a run
    OOMs, since each worker holds a full shard payload plus ``join_asof`` intermediates and memory may
    bind before cores do.  A ``max_workers`` larger than ``cores`` is ignored.

    Examples:
        >>> import os
        >>> resolve_workers(2) <= 2
        True
        >>> resolve_workers() >= 1
        True
    """
    cores = None
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        # A set-but-unparseable value (e.g. SLURM_CPUS_PER_TASK="") is treated as unset and falls
        # through to the next source rather than raising ValueError mid-run.
        try:
            cores = int(os.environ[var])
            break
        except (KeyError, ValueError):
            continue
    if cores is None:
        cores = os.cpu_count() or 1
    return min(cores, max_workers) if max_workers else cores


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


def run(cfg: DictConfig) -> None:
    """Execute the 5-stage pipeline for a fully-resolved config (no Hydra/dotenv side effects).

    Split out from :func:`main` so it is callable directly (tests, programmatic drivers) without
    triggering Hydra arg parsing or ``load_dotenv()``; :func:`main` is the thin Hydra entry point that
    loads ``.env`` and delegates here.  Path roots come from :func:`resolve_training_task_paths`
    (``cfg.data_dir`` / ``cfg.out_dir`` override, else ``$INTERMEDIATE`` / ``$TRAINING_TASKS_DIR``),
    plus the sibling ``_artifacts`` intermediate root; query codes via :func:`read_query_codes`
    (falls back to ``$PROCESSED`` when ``cfg.query_codes`` is null).

    Stages 0-3 run sequentially in this driver process and produce the partitioned Stage 3 index;
    Stage 4 then fans out one :func:`label_one_shard` worker per shard via a ``ProcessPoolExecutor``
    sized by :func:`resolve_workers` (capped by ``cfg.max_workers``).  The query and context axes are
    seeded independently via :func:`derive_seed` so they reproduce separately for a fixed ``cfg.seed``.
    """
    path_to_data, training_tasks_dir, training_task_artifacts_dir = resolve_training_task_paths(cfg)

    # Stage 0: precompute & cache subject prediction_time_indexes and number of prediction_times_per_subject
    n_subjects = build_prediction_times(
        path_to_data=path_to_data,
        training_task_artifacts_dir=training_task_artifacts_dir,
        split=cfg.split,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        overwrite=cfg.overwrite,
    )
    logger.info("Stage 0: %s eligible subject(s) for split=%s.", f"{n_subjects:,}", cfg.split)

    # Independent RNG streams per axis (invariant 5): the query and context draws reproduce separately
    # for a fixed ``cfg.seed``.  Cross-process RNG-order determinism tests land in #211.
    query_rng = np.random.default_rng(derive_seed(cfg.seed, "queries"))
    context_rng = np.random.default_rng(derive_seed(cfg.seed, "contexts"))

    # Stage 1: Sample num_queries QuerySpecs
    query_dist = QueryDistribution.from_config(cfg)
    sampled_queries = query_dist.sample(cfg.num_queries, query_rng)
    logger.info(
        "Stage 1: sampled %s quer%s from a %s-code universe (%s durations over [%g, %g] days).",
        f"{len(sampled_queries):,}",
        "y" if len(sampled_queries) == 1 else "ies",
        f"{query_dist.query_universe_size:,}",
        query_dist.duration_distribution,
        query_dist.min_duration,
        query_dist.max_duration,
    )

    # Stage 2: Sample (num_queries*num_contexts_per_query) patient contexts
    total_rows = cfg.num_queries * cfg.num_contexts_per_query
    prediction_time_counts_df = pl.read_parquet(
        prediction_time_counts_path(training_task_artifacts_dir, cfg.split)
    )

    sampled_patient_contexts = sample_patient_contexts(
        prediction_time_counts=prediction_time_counts_df,
        n=total_rows,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        rng=context_rng,
    )
    logger.info(
        "Stage 2: sampled %s patient context(s) (%s quer%s * %s context(s) each).",
        f"{total_rows:,}",
        f"{cfg.num_queries:,}",
        "y" if cfg.num_queries == 1 else "ies",
        f"{cfg.num_contexts_per_query:,}",
    )

    # Stage 3: zip queries with contexts, resolve prediction time, write the per-shard index.
    n_index_shards = build_index(
        queries=sampled_queries,
        contexts=sampled_patient_contexts,
        training_task_artifacts_dir=training_task_artifacts_dir,
        split=cfg.split,
        num_contexts_per_query=cfg.num_contexts_per_query,
    )
    logger.info(
        "Stage 3: wrote partitioned index for split=%s (%s rows across %s shards).",
        cfg.split,
        f"{sampled_patient_contexts.height:,}",
        f"{n_index_shards:,}",
    )

    # Stage 4: fan one labeling worker out per shard.  Shards are exactly the Stage 3 index
    # partitions; workers receive ids/paths (never DataFrames) and write their own atomic output.
    index_dir = training_task_artifacts_dir / cfg.split / INDEX_DIRNAME
    data_dir = path_to_data / "data" / cfg.split
    out_dir = training_tasks_dir / cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)  # driver creates out_dir once, before the pool

    shards = sorted(p.stem for p in index_dir.glob("*.parquet"))

    # Prune stale outputs from a previous run with a different shard set: unlike index_dir (rebuilt
    # via rmtree in build_index), out_dir persists across runs, so a leftover {shard}.parquet would
    # pollute the {split}/*.parquet union AND get miscounted by the row-count guard below.  Drop any
    # output whose shard isn't in this run's set so the final split dir holds exactly these shards
    # (mirrors Stage 0/3 dropping stale partitions).  Current shards are kept so overwrite=False can
    # still skip already-labeled ones.
    current_shards = set(shards)
    for stale in out_dir.glob("*.parquet"):
        if stale.stem not in current_shards:
            stale.unlink()
    # Mirror the prune onto the per-shard label-provenance sidecars so the _labeled/ dir never
    # carries fingerprints for shards the final root no longer has.
    labeled_dir = training_task_artifacts_dir / cfg.split / LABELED_DIRNAME
    if labeled_dir.exists():
        for stale_fp in labeled_dir.glob("*.json"):
            if stale_fp.stem not in current_shards:
                stale_fp.unlink()

    n_workers = resolve_workers(cfg.max_workers)
    logger.info("Stage 4: labeling %s shard(s) across %s worker(s).", f"{len(shards):,}", f"{n_workers:,}")

    # Use the "spawn" start method, not the Linux default "fork": by Stage 4 the driver has already
    # run polars (which starts a rayon threadpool), and forking a process while those threads hold
    # locks leaves the child with inherited-but-locked mutexes -> the worker deadlocks in futex the
    # moment label_one_shard touches polars (see #210).  spawn gives each worker a fresh interpreter.
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as ex:
        futs = {ex.submit(label_one_shard, s, index_dir, data_dir, out_dir, cfg.overwrite): s for s in shards}
        for fut in as_completed(futs):
            fut.result()  # re-raise so a failed shard aborts the run loudly

    # The final dataset is the union of the shard parquets; its row count must equal the sampling
    # budget N = num_queries * num_contexts_per_query (spec Stage 4).  Resolve the file list
    # explicitly (rather than a glob string) so the degenerate empty-budget case (num_queries=0 or
    # num_contexts_per_query=0 -> no index, no output) reports 0 rows instead of tripping polars'
    # "no files found" on an empty glob.
    out_files = sorted(out_dir.glob("*.parquet"))
    written = int(pl.scan_parquet(out_files).select(pl.len()).collect().item()) if out_files else 0
    if written != total_rows:
        raise ValueError(
            f"Stage 4 wrote {written} rows but expected {total_rows} "
            f"(num_queries={cfg.num_queries} * num_contexts_per_query={cfg.num_contexts_per_query})."
        )
    logger.info("Pipeline complete: wrote %s labeled rows to %s.", f"{written:,}", out_dir)

    if not out_files:
        # Empty budget: nothing to summarize, and the summary scan below would hit the same empty-glob
        # error the row-count scan just side-stepped.
        return

    # Summary of the final dataset's coverage: how many distinct subjects and distinct queries the
    # written rows span (one scan over the {split}/*.parquet union).  A "query" is the full
    # QuerySpec identity -- the (query-code, duration_days) pair -- so it's counted as the number of
    # distinct (query, duration_days) combinations, not distinct codes alone.
    summary = (
        pl.scan_parquet(out_files)
        .select(
            pl.col(TaskQuerySchema.subject_id_name).n_unique().alias("n_subjects"),
            pl.struct(TaskQuerySchema.query_name, TaskQuerySchema.duration_days_name)
            .n_unique()
            .alias("n_queries"),
            # boolean_value is nullable (null = censored); count each of the three label outcomes so
            # the final class balance (and censoring rate) is visible at a glance.
            pl.col(TaskQuerySchema.boolean_value_name).null_count().alias("n_null"),
            (pl.col(TaskQuerySchema.boolean_value_name) == False).sum().alias("n_false"),  # noqa: E712
            (pl.col(TaskQuerySchema.boolean_value_name) == True).sum().alias("n_true"),  # noqa: E712
        )
        .collect()
    )
    n_subjects = int(summary["n_subjects"].item())
    n_queries = int(summary["n_queries"].item())
    n_null = int(summary["n_null"].item())
    n_false = int(summary["n_false"].item())
    n_true = int(summary["n_true"].item())
    logger.info(
        "Summary: %s row(s) across %s unique subject_id(s) and %s unique quer%s.",
        f"{written:,}",
        f"{n_subjects:,}",
        f"{n_queries:,}",
        "y" if n_queries == 1 else "ies",
    )
    logger.info(
        "Summary: boolean_value label counts -- %s null (censored), %s false, %s true.",
        f"{n_null:,}",
        f"{n_false:,}",
        f"{n_true:,}",
    )


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_training_tasks_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point (``EQ_generate_training_tasks``) — loads ``.env``, then runs :func:`run`.

    Loads ``.env`` via python-dotenv first (the repo convention where machine paths live in a
    gitignored ``.env`` rather than being exported by the user), then delegates the whole 5-stage
    pipeline to :func:`run`.
    """
    # Late import so `load_dotenv()` doesn't run at module import time (which would be an
    # unexpected side effect for programmatic callers / tests of the pure primitives).
    from dotenv import load_dotenv

    load_dotenv()
    run(cfg)


if __name__ == "__main__":
    main()
