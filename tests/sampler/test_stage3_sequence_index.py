"""Stage 3': ``build_sequence_index`` and its ``resolve_prediction_times`` rank resolver.

The sequence analogue of Stage 3 (``sample_tasks.build_index``, covered in
``test_stage3_build_index.py``).  Where that function zips one query across N contexts, this one
expands each context into the ``L`` queries of its sequence, tagged with ``_ctx_id`` /
``_position`` so Stage 4' can reassemble the list columns.

Both functions survive the deletion of the conditional query-sequence *model*:

* ``build_sequence_index`` (``query_sequence_labeling.py:558``) is called at ``:1382`` by the
  Hydra ``run`` that is the sampler differential's training-pipeline arm, and
  ``sample_evaluation_query_sequences`` pins its own index shape and dtypes against this
  function *by name*, so the eval grid depends on it too.
* ``resolve_prediction_times`` is a shared seam: ``sample_multitask_sequences`` imports it at
  ``:116`` and calls it at ``:1102`` as ``EQ_generate_multitask_sequences``' Stage 3 join.

The two guards on the join exist because the *silent* failure is the dangerous one:

* a **join-key dtype mismatch** does not raise in polars — it matches nothing, and every row
  comes back with a null ``prediction_time``;
* a **left join** is used deliberately so unresolvable ``(subject_id, prediction_time_index)``
  pairs surface as nulls and raise, where an inner join would quietly drop those contexts and
  train on a silently smaller cohort.

These tests were lost when the conditional query-sequence pipeline was deleted; the functions they
cover were not.
"""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from every_query.data.query_seq_dataset import EOS_CODE
from every_query.generate_tasks.query_sequence_labeling import (
    CTX_ID_COL,
    POSITION_COL,
    build_sequence_index,
    resolve_prediction_times,
)
from every_query.generate_tasks.sample_tasks import QuerySpec, index_path, prediction_times_path


def _write_prediction_times_map(artifacts: Path, split: str, shard: str, rows: list[tuple]) -> None:
    """Write a Stage 0 ``_prediction_times/{shard}.parquet`` map with upstream's exact dtypes."""
    fp = prediction_times_path(artifacts, split, shard)
    fp.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "prediction_time_index": [r[1] for r in rows],
            "time": [r[2] for r in rows],
        },
        schema={
            "subject_id": pl.Int64,
            "prediction_time_index": pl.Int64,
            "time": pl.Datetime("us"),
        },
    ).write_parquet(fp)


def _stage2_contexts(rows: list[tuple]) -> pl.DataFrame:
    """A Stage 2 output frame: ``(subject_id, shard, prediction_time_index)``."""
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "shard": [r[1] for r in rows],
            "prediction_time_index": [r[2] for r in rows],
        },
        schema={"subject_id": pl.Int64, "shard": pl.Utf8, "prediction_time_index": pl.Int64},
    )


def test_build_sequence_index_partitions_by_shard_and_resolves_times(tmp_path: Path):
    """Stage 3': one index partition per shard, ranks resolved, sequences kept intact.

    The input is deliberately adversarial for a positional zip: the three sequences are handed in
    an order that matches neither shard order nor rank order, and two of them land on the *same*
    subject at different ranks.  A zip that drifted by one, or a group_by that let two sequences
    share a ``_ctx_id``, would mislabel every row without raising.
    """
    artifacts, split = tmp_path / "tasks_artifacts", "train"
    _write_prediction_times_map(
        artifacts, split, "0", [(1, 0, datetime(2024, 1, 1)), (1, 1, datetime(2024, 1, 8))]
    )
    _write_prediction_times_map(artifacts, split, "1", [(2, 0, datetime(2024, 2, 1))])

    sequences = [
        [QuerySpec("A", 3.0), QuerySpec("B", -1.0, "DISCHARGE")],  # -> shard 0, subject 1, rank 1
        [QuerySpec("C", 5.0)],  # -> shard 1, subject 2, rank 0
        [QuerySpec("A", 1.0), QuerySpec(EOS_CODE, 9.0)],  # -> shard 0, subject 1, rank 0
    ]
    contexts = _stage2_contexts([(1, "0", 1), (2, "1", 0), (1, "0", 0)])

    assert build_sequence_index(sequences, contexts, artifacts, split) == 2

    shard0 = pl.read_parquet(index_path(artifacts, split, "0"))
    shard1 = pl.read_parquet(index_path(artifacts, split, "1"))
    assert shard0.height == 4 and shard1.height == 1

    # Ranks resolved to the map's timestamps, and each sequence's queries stayed together, in
    # order, under its own _ctx_id.
    by_ctx = {
        key[0]: (grp["query"].to_list(), grp["prediction_time"].to_list())
        for key, grp in shard0.sort(CTX_ID_COL, POSITION_COL).group_by(CTX_ID_COL, maintain_order=True)
    }
    assert by_ctx[0] == (["A", "B"], [datetime(2024, 1, 8)] * 2)
    assert by_ctx[2] == (["A", EOS_CODE], [datetime(2024, 1, 1)] * 2)
    assert shard1["prediction_time"].to_list() == [datetime(2024, 2, 1)]

    # _ctx_id is globally unique, so no two sequences collide when Stage 4' regroups.
    assert len(set(shard0[CTX_ID_COL].to_list() + shard1[CTX_ID_COL].to_list())) == 3

    # Stage 3' samples nothing: the bound Stage 1' put on a spec is carried into the index as-is,
    # and every shard of the run agrees on the column, bounded rows or not.
    ordered = shard0.sort(CTX_ID_COL, POSITION_COL)
    assert ordered["bound_event"].to_list() == [None, "DISCHARGE", None, None]
    assert ordered["duration_days"].to_list()[1] == -1.0
    assert shard1["bound_event"].to_list() == [None]


def test_build_sequence_index_rejects_length_mismatch(tmp_path: Path):
    """Stage 1' and Stage 2 are zipped positionally; unequal lengths must fail here, not silently truncate."""
    with pytest.raises(ValueError, match=r"must equal contexts\.height"):
        build_sequence_index(
            [[QuerySpec("A", 1.0)]],
            _stage2_contexts([(1, "0", 0), (2, "0", 0)]),
            tmp_path,
            "train",
        )


def test_resolve_prediction_times_raises_on_unresolvable_rank(tmp_path: Path):
    """The join is total by design — a null timestamp is a hard error, never a silent drop."""
    artifacts, split = tmp_path / "a", "train"
    _write_prediction_times_map(artifacts, split, "0", [(1, 0, datetime(2024, 1, 1))])
    with pytest.raises(ValueError, match="null prediction_time"):
        resolve_prediction_times(_stage2_contexts([(1, "0", 99)]), artifacts, split, "0")


def test_resolve_prediction_times_raises_on_join_key_dtype_drift(tmp_path: Path):
    """A dtype mismatch silently produces an all-null join in polars; it must fail loudly."""
    artifacts, split = tmp_path / "a", "train"
    _write_prediction_times_map(artifacts, split, "0", [(1, 0, datetime(2024, 1, 1))])
    drifted = _stage2_contexts([(1, "0", 0)]).with_columns(pl.col("subject_id").cast(pl.UInt32))
    with pytest.raises(ValueError, match="dtype mismatch"):
        resolve_prediction_times(drifted, artifacts, split, "0")
