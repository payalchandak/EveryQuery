"""Shared synthetic fixtures for the multitask sampler suite (``tests/multitask/``).

Everything here is synthetic: no real cohort is read.  The doctest-namespace override mirrors
``tests/sampler/conftest.py`` so this layer stays offline and never builds the HF demo model.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

CODES = [f"C//{i}" for i in range(11)] + ["TIMELINE//END"]
K = 5


@pytest.fixture(autouse=True)
def _setup_doctest_namespace():
    yield


def make_codes_parquet(root: Path, codes: list[str] = CODES, *, first_index: int = 1) -> Path:
    """Write ``{root}/metadata/codes.parquet`` with ``code/vocab_index`` starting at ``first_index``."""
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"code": codes, "code/vocab_index": list(range(first_index, first_index + len(codes)))}
    ).write_parquet(meta / "codes.parquet")
    return meta / "codes.parquet"


def make_events(
    rng: np.random.Generator,
    subject_ids: list[int],
    *,
    n_times: tuple[int, int] = (20, 40),
    horizon_days: int = 400,
    codes: list[str] = CODES,
    with_static: bool = True,
) -> pl.DataFrame:
    """Random events for ``subject_ids``: multiple codes per timestamp, an EOS row, a null-time row."""
    base = datetime(2020, 1, 1)  # noqa: DTZ001 — naive timestamps are fine for synthetic fixtures
    rows = []
    obs = [c for c in codes if c != "TIMELINE//END"]
    for sid in subject_ids:
        n = int(rng.integers(*n_times))
        times = np.sort(rng.integers(0, horizon_days, n))
        for t in times:
            for _ in range(int(rng.integers(1, 3))):
                rows.append(
                    {
                        "subject_id": sid,
                        "time": base + timedelta(days=int(t)),
                        "code": obs[int(rng.integers(0, len(obs)))],
                    }
                )
        if "TIMELINE//END" in codes:
            rows.append(
                {
                    "subject_id": sid,
                    "time": base + timedelta(days=int(times[-1]) + 1),
                    "code": "TIMELINE//END",
                }
            )
        if with_static:
            rows.append({"subject_id": sid, "time": None, "code": obs[0]})
    return pl.DataFrame(rows).with_columns(
        pl.col("time").cast(pl.Datetime("us")), pl.col("subject_id").cast(pl.Int64)
    )


def write_cohort(
    root: Path, shard_to_events: dict[str, pl.DataFrame], split: str = "train", codes: list[str] = CODES
) -> Path:
    """Write a synthetic MEDS root: ``data/{split}/{shard}.parquet`` + ``metadata/codes.parquet``."""
    for shard, df in shard_to_events.items():
        d = root / "data" / split
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f"{shard}.parquet")
    make_codes_parquet(root, codes)
    return root


def write_cohort_ontology(
    cohort_dir: Path, out: Path, decay: float = 0.5, *, swap: tuple[str, str] | None = None
) -> Path:
    """The three ``EQ_build_ontology`` artifacts for ``{cohort_dir}/metadata/codes.parquet``, in ``out``.

    What ``EQ_build_ontology`` writes, minus the CLI: leaf ids are the cohort's own ``code/vocab_index``,
    ancestors are appended above them.  Returned so tests can build a ``V_ext``-wide model or datamodule
    against the session fixture cohort.

    ``swap=(a, b)`` exchanges the two codes' indices before building: a *same-width* ontology of the same
    codes at a permuted numbering - identical ``V`` and ``V_ext``, internally consistent artifacts - that
    every width check accepts and only the cohort-identity check can refuse.
    """
    from every_query.data.ontology import (
        EMBEDDING_MIX_FILE,
        EVENT_TO_QUERY_NODES_FILE,
        ONTOLOGY_VOCAB_FILE,
        build_event_to_query_nodes,
        build_ontology,
    )

    codes = pl.read_parquet(Path(cohort_dir) / "metadata" / "codes.parquet").filter(
        pl.col("code/vocab_index").is_not_null() & pl.col("code").is_not_null()
    )
    if swap is not None:
        a, b = swap
        index_of = dict(zip(codes["code"].to_list(), codes["code/vocab_index"].to_list(), strict=True))
        if a not in index_of or b not in index_of:
            raise KeyError(f"swap codes {swap} are not both in the cohort vocabulary")
        codes = codes.with_columns(
            pl.when(pl.col("code") == a)
            .then(pl.lit(index_of[b]))
            .when(pl.col("code") == b)
            .then(pl.lit(index_of[a]))
            .otherwise(pl.col("code/vocab_index"))
            .cast(codes.schema["code/vocab_index"])
            .alias("code/vocab_index")
        )
    nodes, mix = build_ontology(codes.select("code", "code/vocab_index"), decay=decay)
    out.mkdir(parents=True, exist_ok=True)
    nodes.write_parquet(out / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(out / EMBEDDING_MIX_FILE)
    build_event_to_query_nodes(nodes, mix).write_parquet(out / EVENT_TO_QUERY_NODES_FILE)
    return out


@pytest.fixture
def synthetic_cohort(tmp_path: Path) -> Path:
    """Two-shard synthetic cohort with seven subjects per shard."""
    rng = np.random.default_rng(1)
    shards = {shard: make_events(rng, [int(shard) * 100 + s for s in range(1, 8)]) for shard in ("0", "1")}
    return write_cohort(tmp_path / "cohort", shards)


def base_cfg(cohort: Path, out_dir: Path, **overrides) -> dict:
    cfg = {
        "data_dir": str(cohort),
        "out_dir": str(out_dir),
        "query_codes": str(cohort),
        "split": "train",
        "seed": 3,
        "num_training_examples": 120,
        "num_bounds": K,
        "duration_min": 1,
        "duration_max": 100,
        "duration_distribution": "log-uniform",
        "eventbound_fraction": 0.5,
        "boundary_codes": None,
        "min_prediction_times_per_subject": 5,
        "max_workers": 2,
        "label_chunk_rows": 7,
        "ontology_dir": None,
        "ontology_mode": None,
        "overwrite": False,
    }
    cfg.update(overrides)
    return cfg


def make_index(
    contexts: list[tuple[int, datetime]],
    bounds: list[list[tuple[float, str | None]]],
    conditions: list[list[str]] | None = None,
    *,
    fill_condition: str = "A",
    starts: list[list[tuple[float, str | None]]] | None = None,
) -> pl.DataFrame:
    """Build a supplied multitask index; ``bounds[i][k]`` is ``(duration_days, bound_event)``.

    ``conditions[i]`` holds the ``K-1`` conditioning codes; by default every slot is ``fill_condition``.
    ``starts[i][k]`` is the issue #24 ``(start_duration_days, start_event)``; when ``None`` (default)
    the start columns are omitted entirely, which the labeler reads as prediction-time starts.
    """
    if conditions is None:
        conditions = [[fill_condition] * (len(row) - 1) for row in bounds]
    cols = {
        "subject_id": pl.Series([c[0] for c in contexts], dtype=pl.Int64),
        "prediction_time": pl.Series([c[1] for c in contexts], dtype=pl.Datetime("us")),
    }
    if starts is not None:
        cols["start_durations"] = pl.Series(
            [[s[0] for s in row] for row in starts], dtype=pl.List(pl.Float32)
        )
        cols["start_events"] = pl.Series([[s[1] for s in row] for row in starts], dtype=pl.List(pl.Utf8))
    cols["durations"] = pl.Series([[b[0] for b in row] for row in bounds], dtype=pl.List(pl.Float32))
    cols["bound_events"] = pl.Series([[b[1] for b in row] for row in bounds], dtype=pl.List(pl.Utf8))
    cols["condition_codes"] = pl.Series(conditions, dtype=pl.List(pl.Utf8))
    return pl.DataFrame(cols)


def condition_answers_oracle(
    meta: pl.DataFrame, dense: np.ndarray, vocab, ontology_dir: Path | None = None
) -> np.ndarray:
    """``(N, K-1)`` bool: ``dense[i, j, index(condition_codes[i, j])]``, computed slot by slot.

    ``dense`` is the ``(N, K, V)`` *leaf* target block.  An ontology-node conditioning code has no
    column there, so with an ``ontology_dir`` its answer is computed the way the contract defines it:
    the OR over the node's closure descendants, read straight off ``event_to_query_nodes.parquet``.
    Independent of the sampler's interval-lookup implementation, which is the point.
    """
    c2i = vocab.boundary_code_to_index() if ontology_dir is not None else vocab.code_to_index()
    leaves: dict[str, list[int]] = {}
    if ontology_dir is not None:
        from every_query.data.ontology import load_event_to_query_nodes

        base = vocab.code_to_index()
        closure = load_event_to_query_nodes(ontology_dir)
        for leaf, node in zip(closure["event_code"].to_list(), closure["query_node"].to_list(), strict=True):
            if leaf in base:
                leaves.setdefault(node, []).append(base[leaf])

    def answer(i: int, j: int, code: str) -> bool:
        idx = c2i[code]
        if idx < dense.shape[2]:
            return bool(dense[i, j, idx])
        return bool(dense[i, j, leaves[code]].any())

    rows = meta["condition_codes"].to_list()
    return np.array(
        [[answer(i, j, c) for j, c in enumerate(row)] for i, row in enumerate(rows)], dtype=bool
    ).reshape(meta.height, -1)


def scalar_oracle(
    index_df: pl.DataFrame, events_df: pl.DataFrame, codes: list[str], num_bounds: int
) -> np.ndarray:
    """Every ``(context, boundary, code)`` through ``label_with_event_bounds``; ``(N, K, len(codes))`` bool.

    Rows follow ``index_df``'s order.  The scalar oracle knows only ``(prediction_time, boundary)``
    windows, so ``index_df`` must have prediction-time starts (no start columns, or all
    ``start_durations == 0``); :func:`resolved_start_scalar_oracle` covers explicit starts.
    """
    if "start_durations" in index_df.columns:
        assert (index_df["start_durations"].explode() == 0).all(), (
            "scalar_oracle needs prediction-time starts"
        )
    return _scalar_oracle_rows(index_df, events_df, codes, num_bounds, index_df["prediction_time"].to_list())


def resolved_start_scalar_oracle(
    index_df: pl.DataFrame,
    events_df: pl.DataFrame,
    codes: list[str],
    num_bounds: int,
    resolved_starts: np.ndarray,
) -> np.ndarray:
    """Issue #24 differential: ``label_with_event_bounds`` fed each window's *resolved start* as its
    prediction time.

    ``resolved_starts`` is ``(N, K)`` int64 µs; windows whose start is ``INF`` are
    returned all-false (the scalar path has no notion of a window that never opens).
    """
    from every_query.generate_tasks.interval_table import INF

    n = index_df.height
    out = np.zeros((n, num_bounds, len(codes)), dtype=bool)
    for k in range(num_bounds):
        rows = [i for i in range(n) if resolved_starts[i, k] != INF]
        if not rows:
            continue
        sub = index_df[rows].with_columns(
            pl.Series("prediction_time", resolved_starts[rows, k].astype(np.int64)).cast(pl.Datetime("us")),
            pl.col("durations").list.slice(k, 1),
            pl.col("bound_events").list.slice(k, 1),
        )
        out[rows, k] = _scalar_oracle_rows(sub, events_df, codes, 1, sub["prediction_time"].to_list())[:, 0]
    return out


def _scalar_oracle_rows(
    index_df: pl.DataFrame, events_df: pl.DataFrame, codes: list[str], num_bounds: int, pts: list
) -> np.ndarray:
    from every_query.generate_tasks.query_sequence_labeling import label_with_event_bounds

    recs = []
    for i, r in enumerate(index_df.iter_rows(named=True)):
        for k in range(num_bounds):
            for j, code in enumerate(codes):
                recs.append(
                    {
                        "_ctx_id": i * num_bounds + k,
                        "_position": j,
                        "subject_id": r["subject_id"],
                        "prediction_time": pts[i],
                        "query": code,
                        "duration_days": r["durations"][k],
                        "bound_event": r["bound_events"][k],
                    }
                )
    idx = pl.DataFrame(recs).with_columns(
        pl.col("_ctx_id").cast(pl.UInt32),
        pl.col("duration_days").cast(pl.Float32),
        pl.col("prediction_time").cast(pl.Datetime("us")),
        pl.col("bound_event").cast(pl.Utf8),
    )
    lab = label_with_event_bounds(idx, events_df.filter(pl.col("time").is_not_null()))
    return np.array(lab["answers"].to_list(), dtype=bool).reshape(index_df.height, num_bounds, len(codes))
