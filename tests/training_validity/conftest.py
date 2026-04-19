"""Shared fixtures for ``tests/training_validity/``.

The session-scoped ``oracle_dataset`` fixture lives here (not in the test module) so
the README doctest and the integration test share one synthesis run.  An earlier
iteration had a separate ``_oracle_doctest_shards`` fixture that re-ran
``_synthesize_meds`` just for the doctest — wasted CPU plus a latent risk of drift
between what the doctest displayed and what the model actually trained on.  Sharing
one session-scoped fixture collapses both concerns: one synthesis per session, and
byte-identical ``events`` / ``labels`` DataFrames across the two consumers.
"""

from __future__ import annotations

import polars as pl
import pytest
from meds import train_split, tuning_split

from tests.training_validity.test_training_validity import (
    _DATASET_SEED,
    _N_TRAIN_SUBJECTS,
    _N_TUNING_SUBJECTS,
    _compute_labels,
    _synthesize_meds,
)


@pytest.fixture(scope="session")
def oracle_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Synthesize the Design-2 MEDS shards + per-(subject, duration) labels.

    Returns both the on-disk paths (consumed by ``EQ_process_data`` / ``EQ_train``
    subprocesses downstream) and the in-memory DataFrames (consumed by the README
    doctest via the autouse namespace-injection fixture below).
    """
    meds_dir = tmp_path_factory.mktemp("d2_meds")
    event_shards = _synthesize_meds(meds_dir, seed=_DATASET_SEED)

    train_subjects = list(range(1000, 1000 + _N_TRAIN_SUBJECTS))
    tuning_subjects = list(range(2000, 2000 + _N_TUNING_SUBJECTS))

    task_labels_dir = tmp_path_factory.mktemp("d2_labels")
    labels_by_split: dict[str, pl.DataFrame] = {}
    for split_name, subjects in [
        (train_split, train_subjects),
        (tuning_split, tuning_subjects),
    ]:
        labels = _compute_labels(event_shards[split_name], subjects)
        labels_by_split[split_name] = labels
        split_dir = task_labels_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        labels.write_parquet(split_dir / "0.parquet")

    return {
        "meds_dir": meds_dir,
        "task_labels_dir": task_labels_dir,
        "train_subjects": train_subjects,
        "tuning_subjects": tuning_subjects,
        "event_shards": event_shards,
        "labels_by_split": labels_by_split,
    }


@pytest.fixture(autouse=True)
def _inject_oracle_doctest_namespace(doctest_namespace: dict, oracle_dataset: dict):
    """Expose the train-split ``events`` / ``labels`` DataFrames under the names the README doctest's ``>>>``
    blocks reference.

    Autouse is fine here even though it fires for non-doctest items too: the
    underlying ``oracle_dataset`` is session-scoped (one synthesis run regardless),
    and this fixture only does dict updates + a scoped ``pl.Config`` context — no
    meaningful overhead on the slow training-validity test.
    """
    doctest_namespace["events"] = oracle_dataset["event_shards"][train_split]
    doctest_namespace["labels"] = oracle_dataset["labels_by_split"][train_split]
    doctest_namespace["pl"] = pl
    # Default polars display caps at 10 rows; bump it so the README snapshots show the
    # full 13-row code-counts and 10-row marker-pair tables without truncation.  Scoped
    # via ``pl.Config`` as a context manager so the setting is restored after the
    # doctest runs — otherwise ``set_tbl_rows(20)`` would leak into any later-running
    # test that compares polars' formatted output.
    with pl.Config(tbl_rows=20):
        yield
