"""Cross-shard invariants of the sampled 5-stage sequence pipeline, driven as a real subprocess.

``query_sequence_labeling`` has no console script — it is reached as ``python -m
every_query.generate_tasks.query_sequence_labeling`` and survives as the training-pipeline arm of
the sampler differential.  ``tests/sampler/test_sequence_orchestration.py`` drives its ``run()``
in-process against designed synthetic cohorts, but those cohorts are single-shard, so two
invariants that only exist *because* a split spans several shards go unexercised there:

1. **The budget is global, not per-shard.** ``num_training_sequence_examples`` is a split-wide
   total: Stage 2 draws contexts across the whole split weighted by each subject's prediction-time
   count.  A regression that made it per-shard would multiply the training set by the shard count
   and still look correct on any single-shard fixture.
2. **Invariant 7: the two output roots stay disjoint.** MEDS-TorchData ``rglob``s the task-labels
   dir, so a stray parquet or ``_``-prefixed intermediate leaking into the final root is picked up
   as labels and breaks the schema-concat downstream.

The session fixture cohort's ``train`` split is genuinely sharded (``data/train/0.parquet`` and
``data/train/1.parquet`` in ``meds_testing_helpers``' static sample), which is what makes these
assertions bite.  ``tuning`` is single-shard and is checked for the budget only.
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from meds import train_split, tuning_split

from conftest import run_and_check

N_SEQUENCES = 8


@pytest.fixture(scope="module")
def seq_sampler_out_dir(eq_preprocessed_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the sampled 5-stage pipeline for train + tuning against the session cohort.

    Exercises Stage 0 (prediction-time map), Stage 1' (sequence draw), Stage 2 (contexts),
    Stage 3' (per-shard index) and Stage 4' (fanned-out labeling).
    ``min_prediction_times_per_subject=1`` because the demo cohort's subjects have only a handful
    of distinct times each.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("seq_sampler_out")
    for split in (train_split, tuning_split):
        run_and_check(
            [
                sys.executable,
                "-m",
                "every_query.generate_tasks.query_sequence_labeling",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"query_codes={eq_preprocessed_dataset!s}",
                f"split={split}",
                f"num_training_sequence_examples={N_SEQUENCES}",
                "min_queries=1",
                "max_queries=5",
                "duration_min=1",
                "duration_max=30",
                "min_prediction_times_per_subject=1",
                "seed=1",
            ],
            timeout=180.0,
        )
    return out_dir


def test_the_train_split_really_spans_several_shards(seq_sampler_out_dir: Path):
    """Anchors the two invariants below — neither means anything on a single-shard split."""
    shards = sorted((seq_sampler_out_dir / train_split).glob("*.parquet"))
    assert len(shards) > 1, (
        f"expected the train split to produce several output shards, found {[p.name for p in shards]}; "
        "the cross-shard invariants below are vacuous otherwise"
    )


def test_sampled_run_writes_exactly_num_sequences_across_shards(seq_sampler_out_dir: Path):
    """The global budget is honoured: one output row per sampled sequence, split-wide."""
    for split in (train_split, tuning_split):
        total = sum(pl.read_parquet(fp).height for fp in (seq_sampler_out_dir / split).glob("*.parquet"))
        assert total == N_SEQUENCES, (
            f"{split}: expected num_training_sequence_examples={N_SEQUENCES} rows split-wide, found {total}"
        )


def test_sampled_run_keeps_the_two_artifact_roots_disjoint(seq_sampler_out_dir: Path):
    """Invariant 7: the final root holds only ``{shard}.parquet``; intermediates go to the sibling.

    MEDS-TorchData rglobs the task-labels dir, so any stray parquet or ``_``-prefixed entry leaking
    into the final root would be picked up as labels and break the schema-concat downstream.
    """
    artifacts = seq_sampler_out_dir.parent / f"{seq_sampler_out_dir.name}_artifacts"
    assert (artifacts / train_split / "_prediction_time_counts.parquet").exists()
    assert list((artifacts / train_split / "_index").glob("*.parquet"))

    stray = [p.name for p in (seq_sampler_out_dir / train_split).iterdir() if p.name.startswith("_")]
    assert not stray, f"intermediates leaked into the final output root: {stray}"
    assert artifacts not in seq_sampler_out_dir.parents, "artifact roots must never nest"
