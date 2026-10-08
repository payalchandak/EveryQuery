"""End-to-end CLI smoke test with the ported features switched on.

The per-feature suites test the mechanisms; this one tests that the *pipeline* survives them —
that ``EQ_build_ontology`` → label generation → ``EQ_train`` actually runs with RoPE time
positions, event bounds and ancestor queries enabled at once, through real subprocesses
against the fixture cohort.

The two arms consume different label artifacts and so have a fixture each:

``query_sequence_labeling`` → ``featured_tasks_dir``
    ``QuerySeqSchema`` parquets.  The generation assertions below read these directly: they are
    where a query, its answer and its event bound sit side by side in one row.

``EQ_generate_multitask_sequences`` → ``featured_multitask_labels_dir``
    ``MultitaskBoundarySchema`` metadata, packed ``.labels.npy`` sidecars and a
    ``_multitask_manifest.json``.  This is what ``ConditionalMultitaskDataModule`` — the datamodule
    of the only surviving conditional demo config — reads, so it is what the training arm needs.
    The two are not interchangeable.

Marked ``slow``: it trains a model.  Run with ``pytest -m slow tests/test_features_e2e_cli.py``.

Boundary codes are read from the fixture's own vocabulary at runtime rather than hardcoded, so
this does not encode any particular cohort's spellings.
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from meds import train_split, tuning_split

from conftest import run_and_check

# The sampler's ``K`` (``num_bounds``) and the model's window budget (``max_windows``) must agree:
# the model rejects a batch with ``K > max_windows``.  One constant pins both sides below.
MAX_WINDOWS = 5


@pytest.fixture(scope="module")
def ontology_dir(eq_preprocessed_dataset: Path, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ontology")
    run_and_check(
        [
            # Invoked as a module rather than by console script: the venv's entry points were
            # installed before this CLI existed, and re-installing is not this test's job.
            sys.executable,
            "-m",
            "every_query.data.build_ontology",
            f"tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"out_dir={out!s}",
            "decay=0.5",
        ],
        timeout=120.0,
    )
    return out


@pytest.mark.slow
def test_build_ontology_writes_its_three_artifacts(ontology_dir: Path):
    for name in ("ontology_vocab.parquet", "embedding_mix.parquet", "event_to_query_nodes.parquet"):
        assert (ontology_dir / name).is_file(), f"{name} missing"
    nodes = pl.read_parquet(ontology_dir / "ontology_vocab.parquet")
    assert nodes.height > 0
    assert set(nodes.columns) == {"node_name", "token_id", "is_observed_code"}


@pytest.fixture(scope="module")
def featured_tasks_dir(eq_preprocessed_dataset: Path, ontology_dir: Path, tmp_path_factory) -> Path:
    """Generate query sequences with event bounds and ancestor queries both enabled."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("featured_seq_tasks")

    for split in (train_split, tuning_split):
        run_and_check(
            [
                # A module invocation, not a console script: query_sequence_labeling is a library
                # with no `[project.scripts]` entry, and its Hydra `main` is reached this way.
                sys.executable,
                "-m",
                "every_query.generate_tasks.query_sequence_labeling",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"query_codes={eq_preprocessed_dataset!s}",
                f"split={split}",
                "num_training_sequence_examples=64",
                "min_queries=2",
                "max_queries=4",
                "duration_min=1",
                "duration_max=30",
                "min_prediction_times_per_subject=1",
                "seed=1",
                "eventbound_fraction=0.3",
                f"ontology_dir={ontology_dir!s}",
            ],
            timeout=300.0,
        )
    return out_dir


@pytest.mark.slow
def test_generation_emits_every_query_form(featured_tasks_dir: Path, ontology_dir: Path):
    """The generated labels really do contain bounds and ancestors."""
    shards = sorted((featured_tasks_dir / train_split).glob("*.parquet"))
    assert shards, "no output shards"
    df = pl.concat([pl.read_parquet(fp) for fp in shards])

    assert "bound_events" in df.columns, "event bounds were requested but no column was written"
    assert df["bound_events"].explode().null_count() < df["bound_events"].explode().len(), (
        "no query was actually event-bounded"
    )

    # Ancestor queries name a node that is not a leaf of the cohort vocabulary.
    nodes = pl.read_parquet(ontology_dir / "ontology_vocab.parquet")
    ancestors = set(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
    queries = df["queries"].explode().to_list()
    assert ancestors, "the fixture ontology produced no ancestor nodes"
    assert any(q in ancestors for q in queries), "no ancestor query was generated"
    # Boundaries come from the same universe, so ancestor nodes bound queries too.
    bounds = df["bound_events"].explode().drop_nulls().to_list()
    assert any(b in ancestors for b in bounds), "no ancestor node was drawn as a boundary"

    # Answers stay binary and aligned no matter which forms are mixed in.
    assert df["answers"].explode().null_count() == 0
    assert (df["queries"].list.len() == df["answers"].list.len()).all()
    assert (df["queries"].list.len() == df["bound_events"].list.len()).all()


@pytest.fixture(scope="module")
def featured_multitask_labels_dir(eq_preprocessed_dataset: Path, tmp_path_factory) -> Path:
    """Multitask training labels with event-bounded *and* event-started windows both mixed in.

    Modelled on ``conditional_multitask_labels_dir`` in ``test_conditional_multitask_cli``.  Labels
    stay leaf-only even though the training run below sets ``ontology_dir``: ancestor targets are
    derived from the ontology at training time, so no ontology-aware label pass is needed.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("featured_multitask_labels")

    for split in (train_split, tuning_split):
        run_and_check(
            [
                "EQ_generate_multitask_sequences",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"query_codes={eq_preprocessed_dataset!s}",
                f"split={split}",
                "num_training_examples=8",
                f"num_bounds={MAX_WINDOWS}",
                "duration_min=0.01",
                "duration_max=2",
                "eventbound_fraction=0.5",
                "eventstart_fraction=0.25",
                "prediction_time_start_fraction=0.25",
                "start_duration_min=0.01",
                "start_duration_max=2",
                "min_prediction_times_per_subject=1",
                "max_workers=1",
                "label_chunk_rows=2",
                "seed=1",
            ],
            timeout=180.0,
        )
    return out_dir


@pytest.mark.slow
def test_train_runs_with_every_feature_enabled(
    eq_preprocessed_dataset: Path,
    featured_multitask_labels_dir: Path,
    ontology_dir: Path,
    tmp_path_factory,
):
    """The decisive check: a real training run with all three features switched on."""
    out = tmp_path_factory.mktemp("featured_train")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train_conditional_multitask_ar",
            f"output_dir={out!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={featured_multitask_labels_dir!s}",
            # ``datamodule.max_windows`` interpolates this, so the one override pins both the
            # model's block-position table and the datamodule to the sampler's K.
            f"lightning_module.model.max_windows={MAX_WINDOWS}",
            # RoPE time is two halves of one setting.  The production config interpolates them
            # (``strip_delta_tokens: ${lightning_module.model.use_rope_time}``); the demo config
            # leaves them independent, so both must be flipped here or the model refuses the batch.
            "lightning_module.model.use_rope_time=true",
            "datamodule.dataset_kwargs.strip_delta_tokens=true",
            f"lightning_module.model.ontology_dir={ontology_dir!s}",
            "trainer.limit_val_batches=1",
        ],
        timeout=900.0,
    )
    ckpts = list(out.rglob("*.ckpt"))
    assert ckpts, "training produced no checkpoint"
