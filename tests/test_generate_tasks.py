"""End-to-end tests for the ``EQ_generate_training_tasks`` CLI (sample_tasks).

These are integration tests: they run the real console script against the preprocessed
cohort and verify *behavior*, not internal contracts.  Schema-shape checks, dtype checks,
and bounds-checking on the sampler primitives live in ``tests/sampler/`` — unit-level
coverage where those properties are cheap and reliable to assert.

The integration tests here do two things that unit tests can't:

- **Run the CLI**: reproducibility and knob-honor differentials can only be checked
  against the full argparse → Hydra → ``run_worker`` path.
- **Validate labels against the real raw data**: re-compute what each row's
  ``(boolean_value, occurs)`` should be directly from the event shard the sampler read,
  and assert the sampler's output matches exactly across every row.  This catches bugs
  that schema-shape assertions would miss (wrong censoring math, off-by-one on the window
  endpoint, missing subject handling, etc.).
"""

from __future__ import annotations

import hashlib
import io
from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl

from conftest import run_and_check

if TYPE_CHECKING:
    from pathlib import Path


# Shared CLI overrides matching the foundation fixture.  Duplicated (not factored into a helper)
# because the fixture's values live in conftest.py and the per-test-run invocations need to diverge
# from them on specific fields (num_queries, out_dir) — a helper abstraction would hide the differences
# we are explicitly testing for.
_DURATION_MIN = 1
_DURATION_MAX = 30
_MIN_CONTEXT_PER_SUBJECT = 1


def _read_train_labels(tasks_dir: Path) -> pl.DataFrame:
    """Load + concatenate all labeled parquets under the train split of *tasks_dir*.

    Asserts the result is non-empty.  Without that check, an empty-shards regression in
    `EQ_generate_training_tasks` would silently turn `test_reproducible_with_same_seed` and
    `test_n_tasks_knob_is_honored` into vacuous pass cases (e.g., `2 * 0 == 0`).
    """
    fps = sorted((tasks_dir / "train").glob("*.parquet"))
    assert fps, f"no labeled parquets found under {tasks_dir / 'train'}"
    df = pl.concat([pl.read_parquet(fp) for fp in fps], how="vertical")
    assert df.height > 0, (
        f"train labeled parquets under {tasks_dir / 'train'} exist but contain 0 rows; "
        f"the demo fixture has num_queries>0 and should always emit at least one label"
    )
    return df


def _parquet_hash(df: pl.DataFrame) -> str:
    """Return a short hex digest of *df* serialised as Parquet (for assertion messages)."""
    buf = io.BytesIO()
    df.write_parquet(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def test_labels_match_ground_truth(
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
) -> None:
    """For every row the sampler emitted, recompute ``(boolean_value, occurs)`` from the raw event shard and
    assert exact equality — per ``evaluate_index_df`` semantics.

    The sampler's documented semantics (``generate_tasks/sample_tasks.py::evaluate_index_df``):

        window_end    = prediction_time + duration_days
        censored      = window_end > max_time[subject]
        event_fires   = exists event(subject, code=query, time in (prediction_time, window_end])
        boolean_value = True  if event_fires        # occurrence takes priority (spec §Stage 4)
                        else None if censored        # unobserved tail → censored
                        else False                   # full window observed, no event

    Recomputing row-by-row catches semantics-level bugs that a schema/shape assertion can't:
    off-by-one on the window endpoint (``<`` vs ``<=``), censoring using the wrong ``max_time``,
    ``join_asof`` drift across polars versions, subjects whose code never fires being mislabelled,
    etc.  Covers every row produced by the fixture, not a handful of hand-picked examples.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"

    for split in ("train", "tuning"):
        shard_path = intermediate / "data" / split / "0.parquet"
        # Mirror ``_read_event_shard``'s normalization (cast + dedupe + sort) so the ground-
        # truth recomputation runs over the same frame the sampler labeled.  Skipping this
        # would let schema-level drift (e.g., upstream storing ``code`` as Categorical or
        # ``time`` at ns precision) diverge silently between the production path and the
        # test, hiding real bugs.
        events = (
            pl.read_parquet(shard_path)
            .select(["subject_id", "time", "code"])
            .with_columns(
                pl.col("time").cast(pl.Datetime("us")),
                pl.col("code").cast(pl.Utf8),
            )
            .unique()
            .sort(["subject_id", "time"])
        )
        label_paths = sorted((eq_sampled_tasks_dir / split).glob("*.parquet"))
        assert label_paths, f"no labeled parquets found for split={split} in {eq_sampled_tasks_dir / split}"
        labels = pl.concat([pl.read_parquet(p) for p in label_paths], how="vertical")
        # Guard against empty-shards regression — without this, the per-row ground-truth
        # loop below would do zero iterations and the test would pass vacuously.
        assert labels.height > 0, (
            f"split={split} produced labeled parquets but 0 rows; ground-truth loop "
            f"would be vacuous.  Check EQ_generate_training_tasks for empty-output regression."
        )

        max_time_per_subject = dict(
            events.group_by("subject_id").agg(pl.col("time").max().alias("max_time")).iter_rows()
        )

        # Walk every emitted row.  Simple Python — slow for millions of rows but fine for the
        # ~20-row fixture output, and the cost is bounded by CPU + parquet I/O, not test
        # complexity.  Favours readability + ground-truth correctness over vectorisation.
        #
        # Collapsed nullable boolean_value per TaskQuerySchema (occurrence takes priority over
        # censoring — spec §Stage 4 / evaluate_index_df line "Occurrence takes priority"):
        #   True  → event occurred in (prediction_time, window_end], even if the window's tail
        #           extends past max_time (an observed event is a definitive positive)
        #   null  → no event in window AND censored (window_end > max_time OR subject absent)
        #   False → no event in window and the full window is observed
        for row in labels.iter_rows(named=True):
            subj = row["subject_id"]
            window_end = row["prediction_time"] + timedelta(days=row["duration_days"])

            max_time = max_time_per_subject.get(subj)
            expected_censored = max_time is None or window_end > max_time

            # Ground-truth event_fires: any event of the matching code in
            # (prediction_time, prediction_time + duration_days].  Sampler uses strict-> via a
            # 1µs-shifted join_asof key, so we mirror with a strict > lower bound and an
            # inclusive <= upper bound here.
            subj_events = events.filter(pl.col("subject_id") == subj)
            event_fires = not subj_events.filter(
                (pl.col("code") == row["query"])
                & (pl.col("time") > row["prediction_time"])
                & (pl.col("time") <= window_end)
            ).is_empty()

            # Occurrence takes priority: an event observed in-window is True even when the
            # window's tail is censored; null only when there is no event AND the tail is censored.
            if event_fires:
                expected_boolean = True
            elif expected_censored:
                expected_boolean = None
            else:
                expected_boolean = False

            ctx = (
                f"split={split}, subject_id={subj}, query={row['query']!r}, "
                f"prediction_time={row['prediction_time']}, duration_days={row['duration_days']}, "
                f"max_time={max_time}"
            )
            assert row["boolean_value"] == expected_boolean, (
                f"boolean_value mismatch: sampler={row['boolean_value']!r}, "
                f"expected={expected_boolean!r} (censored={expected_censored}, "
                f"event_fires={event_fires}).  {ctx}"
            )


def test_reproducible_with_same_seed(
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
    tmp_path_factory,
) -> None:
    """Two sampler runs with identical seed + sampling params produce identical labels.

    Regression guard: non-deterministic sampling (e.g., forgetting to seed numpy / polars before
    a shuffle) would silently change eval labels between runs and produce spurious AUROC drift.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    rerun_out = tmp_path_factory.mktemp("eq_tasks_rerun")
    run_and_check(
        [
            "EQ_generate_training_tasks",
            f"data_dir={intermediate!s}",
            f"out_dir={rerun_out!s}",
            f"query_codes={eq_preprocessed_dataset!s}",
            "split=train",
            "num_queries=8",
            "num_contexts_per_query=2",
            f"min_duration={_DURATION_MIN}",
            f"max_duration={_DURATION_MAX}",
            f"min_prediction_times_per_subject={_MIN_CONTEXT_PER_SUBJECT}",
            "seed=1",
        ],
        timeout=120.0,
    )

    fixture_labels = _read_train_labels(eq_sampled_tasks_dir)
    rerun_labels = _read_train_labels(rerun_out)

    # `subject_id` originates from sampled contexts/index_df and should never be null on a
    # valid sampler output.  Assert the invariant up-front so any unexpected null rows fail
    # loudly instead of being silently filtered out by an `is_not_null()` guard (which would
    # mask both a real sampler regression and any non-label parquet leaking into the split
    # directory).
    assert fixture_labels["subject_id"].null_count() == 0, (
        f"fixture EQ_generate_training_tasks output has "
        f"{fixture_labels['subject_id'].null_count()} null subject_id rows"
    )
    assert rerun_labels["subject_id"].null_count() == 0, (
        f"rerun EQ_generate_training_tasks output has "
        f"{rerun_labels['subject_id'].null_count()} null subject_id rows"
    )

    # Compare by row content (sorted) since file-byte equality is flaky across polars versions.
    # The invariant we care about: same seed → same labeled rows.
    sort_cols = ["subject_id", "prediction_time", "query", "duration_days"]
    a = fixture_labels.sort(sort_cols)
    b = rerun_labels.sort(sort_cols)
    assert a.equals(b), (
        "two EQ_generate_training_tasks runs with identical seed produced different labeled output.\n"
        f"fixture hash: {_parquet_hash(a)}  rerun hash: {_parquet_hash(b)}"
    )


def test_n_tasks_knob_is_honored(
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
    tmp_path_factory,
) -> None:
    """Differential: raising ``num_queries`` produces more rows (``num_queries * num_contexts_per_query``).

    Catches silent flag-drop — a shape-only test would pass even if ``num_queries`` were ignored.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    big_out = tmp_path_factory.mktemp("eq_tasks_big")
    run_and_check(
        [
            "EQ_generate_training_tasks",
            f"data_dir={intermediate!s}",
            f"out_dir={big_out!s}",
            f"query_codes={eq_preprocessed_dataset!s}",
            "split=train",
            "num_queries=16",  # 2x the fixture's num_queries=8
            "num_contexts_per_query=2",
            f"min_duration={_DURATION_MIN}",
            f"max_duration={_DURATION_MAX}",
            f"min_prediction_times_per_subject={_MIN_CONTEXT_PER_SUBJECT}",
            "seed=1",  # same seed as the fixture; only num_queries differs
        ],
        timeout=120.0,
    )

    small_df = _read_train_labels(eq_sampled_tasks_dir)
    big_df = _read_train_labels(big_out)

    # Stage 4 labels every index row (left `join_asof`), so the output row count is exactly
    # `num_queries * num_contexts_per_query`.  Doubling `num_queries` with `num_contexts_per_query`
    # held fixed should therefore exactly double the output row count.  A weaker tolerance-based
    # check (e.g. >= 1.5x) would miss a bug that silently caps `num_queries` somewhere between small
    # and big (e.g. at 12) — exact-2x catches that.
    assert big_df.height == 2 * small_df.height, (
        f"num_queries=16 produced {big_df.height} rows vs. {small_df.height} at num_queries=8; "
        f"expected exactly {2 * small_df.height} when only num_queries doubles and "
        f"num_contexts_per_query is held fixed.  This points at the `num_queries` flag being "
        f"ignored, silently capped, or context sampling dropping rows that should have "
        f"made it through."
    )
