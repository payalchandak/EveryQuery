"""The evaluation grid's cohort side is ``sample_evaluation_tasks``' cohort side, per shard.

``sample_evaluation_query_sequences`` takes the flat evaluation sampler's knobs
(``prediction_times_per_subject``, ``min_context_per_subject``, ``subject_subsample_fraction``,
``write_unique_prediction_times``, plus a ``contexts_path`` override), runs one worker per shard of
the split, and writes the flat sampler's
layout (``{out_dir}/eval/{split}/{shard}.parquet`` + ``eval_unique/``).  The property that makes the
two grids comparable — same ``(seed, split, K, min_context, fraction)`` ⇒ same ``(subject, time)``
set — is what these tests pin, along with the per-shard layout and the supplied-cohort override.

Fixture cohort: the shared ``synthetic_events`` frame (3 subjects x 30 distinct times, 10d apart)
split across two shards — ``"0"`` holds subjects 1 and 2, ``"1"`` holds subject 3.

Coverage migrated from the deleted ``tests/multitask/test_eval_multitask_grid.py`` (issue #29)
---------------------------------------------------------------------------------------------
The legacy multitask evaluation generator is gone; ``EQ_generate_evaluation_query_sequences`` is the
one evaluation grid and ``EQ_predict_multitask`` consumes it.  Each concern the deleted module pinned
lives on the consolidated path as follows:

- start/end resolution relative to the *resolved* start, open/open endpoints, missing start and
  missing end: ``tests/test_queryseq_starts.py``
  (``test_each_start_form_against_the_hand_computed_oracle``,
  ``test_end_boundary_is_searched_after_the_resolved_start_not_the_prediction_time``,
  ``test_both_endpoints_are_open_for_every_start_form``,
  ``test_unresolved_start_never_becomes_an_infinite_window``) and
  ``tests/test_window_bounds_contract.py`` (the explicit-start deciders in the shared matrix);
- deterministic spec sampling (``test_task_table_is_deterministic_in_the_seed``):
  :func:`test_deterministic_in_seed` here plus
  ``tests/sampler/test_eval_grid_starts.py::test_sampled_start_draw_is_deterministic`` and
  ``::test_legacy_draw_matches_the_pre_start_golden``;
- independently seeded draw components (``test_groups_are_seeded_independently``):
  ``tests/sampler/test_eval_grid_starts.py::test_each_start_axis_is_independent``,
  ``::test_start_knobs_do_not_perturb_the_legacy_draw``,
  ``::test_legacy_knobs_do_not_perturb_the_start_component``;
- stale-output / provenance protection
  (``test_rerun_with_a_new_seed_relabels_instead_of_serving_stale``,
  ``test_cohort_knob_change_relabels``, ``test_missing_provenance_relabels_only_that_shard``,
  ``test_grid_is_reused_unless_overwritten``): :func:`test_changed_cohort_relabels_instead_of_skipping`
  (seed and cohort-knob changes), :func:`test_changed_specs_relabel_instead_of_skipping`,
  :func:`test_missing_provenance_relabels_only_that_shard`,
  :func:`test_existing_outputs_are_skipped_unless_overwrite`,
  :func:`test_a_pre_fingerprint_sidecar_is_stale`, and
  ``tests/sampler/test_eval_grid_starts.py::test_start_knob_change_relabels_instead_of_serving_stale_shards``;
- every context answers every task / row alignment: :func:`test_rows_are_context_major_within_a_shard`,
  :func:`test_specs_are_shared_across_shards`, and on the prediction side
  ``tests/multitask/test_predict_multitask_logic.py::test_predictions_are_row_aligned_one_per_grid_row``
  / ``::test_predictions_to_df_rejects_misaligned_labels_and_scored_codes``
  / ``::test_a_same_label_row_swap_in_the_loader_is_caught_by_the_scored_codes``;
- vocabulary validation:
  ``tests/sampler/test_eval_grid_starts.py::test_unknown_start_codes_are_rejected_against_the_vocabulary``
  and ``tests/multitask/test_predict_multitask_logic.py::test_unknown_codes_fail_at_init``.

Not migrated, by design: the legacy sidecar contract (``eval_meta`` row alignment,
``eval_tasks.parquet``, the per-window resolved/unresolved diagnostics, the ``_eval_summary.json``
row count, the prevalence-weighted ``task_groups`` and ``exclude_boundary_prefixes`` pools) belonged
to the deleted generator and has no counterpart; the QuerySeq grid's lists identify the task, and
the ``.labels.npy`` "PAD is never a target" bit is replaced by the adapter's PAD-mapped-code rejection.
"""

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml
from hydra import compose, initialize_config_dir

from every_query.data.schema import QuerySeqSchema
from every_query.generate_tasks import sample_evaluation_query_sequences as eval_seq
from every_query.generate_tasks import sample_evaluation_tasks as eval_flat
from every_query.generate_tasks.sample_evaluation_tasks import (
    sample_prediction_times_per_subject,
    subsample_subject_ids,
)
from every_query.generate_tasks.sample_tasks import _read_event_shard
from every_query.utils.seeds import derive_seed
from tests.designed_specs import entry

SPLIT = "held_out"
SHARDS = ["0", "1"]
N_SEQ = 3
LEN = 2


@pytest.fixture
def data_dir(tmp_path: Path, synthetic_events: pl.DataFrame, write_split_shards) -> Path:
    """Write ``synthetic_events`` as two shards and return the MEDS root."""
    return write_split_shards(
        tmp_path,
        {
            "0": synthetic_events.filter(pl.col("subject_id") != 3),
            "1": synthetic_events.filter(pl.col("subject_id") == 3),
        },
        split=SPLIT,
    )


@pytest.fixture
def codes_yaml(tmp_path: Path, synthetic_query_codes: list[str]) -> Path:
    fp = tmp_path / "codes.yaml"
    fp.write_text(yaml.safe_dump(synthetic_query_codes))
    return fp


def _run_seq(data_dir: Path, out_dir: Path, codes_yaml: Path, **overrides) -> None:
    """Compose the real eval-seq config and run ``main`` on it, the way the CLI does."""
    kwargs = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "split": SPLIT,
        "num_evaluation_sequences": N_SEQ,
        "min_queries": LEN,
        "max_queries": LEN,
        "prediction_times_per_subject": 3,
        "min_context_per_subject": 5,
        "seed": 1,
        "min_task_positives": "null",  # the production default (40) would drop every tiny-fixture task
    }
    kwargs.update(overrides)
    with initialize_config_dir(config_dir=eval_seq.CONFIGS, version_base=None):
        cfg = compose(
            config_name="sample_evaluation_query_sequences_config",
            overrides=[f"{k}={v}" for k, v in kwargs.items()],
        )
    eval_seq.main.__wrapped__(cfg)


def _run_flat(data_dir: Path, out_dir: Path, codes_yaml: Path, **overrides) -> None:
    """Same, for ``EQ_generate_evaluation_tasks``."""
    kwargs = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "split": SPLIT,
        "durations": "[1]",
        "prediction_times_per_subject": 3,
        "min_context_per_subject": 5,
        "seed": 1,
    }
    kwargs.update(overrides)
    with initialize_config_dir(config_dir=eval_flat.CONFIGS, version_base=None):
        cfg = compose(
            config_name="sample_evaluation_tasks_config",
            overrides=[f"{k}={v}" for k, v in kwargs.items()],
        )
    eval_flat.main.__wrapped__(cfg)


def _labels(out_dir: Path, shard: str) -> pl.DataFrame:
    return pl.read_parquet(out_dir / "eval" / SPLIT / f"{shard}.parquet")


def _unique(out_dir: Path, shard: str) -> pl.DataFrame:
    return pl.read_parquet(out_dir / "eval_unique" / SPLIT / f"{shard}.parquet")


def _spec_keys(df: pl.DataFrame) -> list[str]:
    """One hashable ``queries|durations`` string per row."""
    return (
        df.select(
            pl.concat_str(
                pl.col("queries").list.join("|"),
                pl.col("durations").cast(pl.List(pl.Utf8)).list.join("|"),
                separator="#",
            )
        )
        .to_series()
        .to_list()
    )


# ── layout ───────────────────────────────────────────────────────────────


def test_writes_one_grid_and_one_unique_parquet_per_shard(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml)

    on_disk = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())
    assert on_disk == [
        *(f"eval/{SPLIT}/{s}.parquet" for s in SHARDS),
        *(f"eval_unique/{SPLIT}/{s}.parquet" for s in SHARDS),
    ]
    # Every shard carries its provenance in the artifacts sibling, ontology or not; the ontology
    # component is simply null here.
    for shard in SHARDS:
        recorded = eval_seq._recorded_fingerprint(out_dir, out_dir / "eval" / SPLIT / f"{shard}.parquet")
        assert recorded is not None and recorded["ontology_fingerprint"] is None
        assert recorded["specs_fingerprint"] and recorded["cohort_fingerprint"].startswith("sampled|")

    for shard in SHARDS:
        df = _labels(out_dir, shard)
        QuerySeqSchema.align(pq.read_table(out_dir / "eval" / SPLIT / f"{shard}.parquet"))
        uniq = _unique(out_dir, shard)
        assert uniq.columns == ["subject_id", "prediction_time"]
        assert df.height == uniq.height * N_SEQ
        assert (df["queries"].list.len() == LEN).all()
        assert df["answers"].explode().null_count() == 0
        assert "bound_events" in df.columns, "the sampled grid defaults eventbound_fraction to 0.5"
        bounded = df.explode("durations", "bound_events").filter(pl.col("bound_events").is_not_null())
        assert bounded.height > 0 and (bounded["durations"] == -1.0).all()
        assert uniq.sort("subject_id", "prediction_time").equals(
            df.select("subject_id", "prediction_time").unique().sort("subject_id", "prediction_time")
        )
    # Subjects 1, 2 on shard 0 and 3 on shard 1: three times each.
    assert _unique(out_dir, "0")["subject_id"].to_list() == [1, 1, 1, 2, 2, 2]
    assert _unique(out_dir, "1")["subject_id"].to_list() == [3, 3, 3]


def test_rows_are_context_major_within_a_shard(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    """Row ``i`` is ``(contexts[i // N], specs[i % N])``, and every context sees the same N specs."""
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml)

    for shard in SHARDS:
        df = _labels(out_dir, shard)
        keys = _spec_keys(df)
        first = keys[:N_SEQ]
        assert len(set(first)) == N_SEQ, "the N specs must be distinct for this check to bite"
        for c in range(df.height // N_SEQ):
            block = df.slice(c * N_SEQ, N_SEQ)
            assert keys[c * N_SEQ : (c + 1) * N_SEQ] == first
            assert block.select("subject_id", "prediction_time").n_unique() == 1


def test_eventbound_fraction_zero_keeps_sampled_grid_bound_free(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    """An explicit zero still disables bounds despite the new 0.5 default."""
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml, eventbound_fraction=0)
    assert all("bound_events" not in _labels(out_dir, shard).columns for shard in SHARDS)


def test_specs_are_shared_across_shards(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    """The N sequences are drawn once, not per shard."""
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml)
    per_shard = [set(_spec_keys(_labels(out_dir, s))) for s in SHARDS]
    assert per_shard[0] == per_shard[1]


# ── the cohort is the flat sampler's cohort ──────────────────────────────


def test_cohort_is_exactly_the_flat_samplers_cohort(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    """Same ``(seed, split, K, min_context, fraction)`` ⇒ byte-identical ``eval_unique`` frames.

    Pinned two ways per shard: against ``sample_prediction_times_per_subject`` on the flat sampler's
    seed axes (exact), and against ``EQ_generate_evaluation_tasks``' own ``eval_unique`` output,
    which is a *subset* because that endpoint drops censored rows before deduplicating.
    """
    common = {"prediction_times_per_subject": 2, "min_context_per_subject": 4, "seed": 11}
    _run_seq(data_dir, tmp_path / "seq", codes_yaml, subject_subsample_fraction=0.7, **common)
    _run_flat(data_dir, tmp_path / "flat", codes_yaml, subject_subsample_fraction=0.7, **common)

    for shard in SHARDS:
        events = _read_event_shard(data_dir / "data" / SPLIT / f"{shard}.parquet")
        events = subsample_subject_ids(events, 0.7, derive_seed(11, "subject_subsample", SPLIT, shard))
        expected = sample_prediction_times_per_subject(
            events, k=2, min_context_per_subject=4, seed=derive_seed(11, "prediction_times", SPLIT, shard)
        )
        seq_unique = _unique(tmp_path / "seq", shard)
        assert seq_unique.equals(expected.select("subject_id", "prediction_time"))

        flat_unique = _unique(tmp_path / "flat", shard)
        assert flat_unique.join(seq_unique, on=["subject_id", "prediction_time"], how="anti").height == 0


def test_min_context_gate_can_empty_a_shard_and_still_writes_it(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml, min_context_per_subject=1000)
    for shard in SHARDS:
        assert _labels(out_dir, shard).height == 0
        assert _unique(out_dir, shard).height == 0
        QuerySeqSchema.align(pq.read_table(out_dir / "eval" / SPLIT / f"{shard}.parquet"))


def test_deterministic_in_seed(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    _run_seq(data_dir, tmp_path / "a", codes_yaml, seed=3)
    _run_seq(data_dir, tmp_path / "b", codes_yaml, seed=3)
    _run_seq(data_dir, tmp_path / "c", codes_yaml, seed=4)
    for shard in SHARDS:
        assert _labels(tmp_path / "a", shard).equals(_labels(tmp_path / "b", shard))
    assert not _unique(tmp_path / "a", "0").equals(_unique(tmp_path / "c", "0"))


# ── contexts_path: the supplied-cohort override ──────────────────────────


def test_supplied_cohort_is_partitioned_by_shard_and_bypasses_the_cohort_knobs(
    tmp_path: Path, data_dir: Path, codes_yaml: Path, synthetic_events: pl.DataFrame
):
    cohort = (
        synthetic_events.group_by("subject_id")
        .agg(pl.col("time").sort().slice(10, 2).alias("prediction_time"))
        .explode("prediction_time")
        .sort("subject_id", "prediction_time")
    )
    cohort_fp = tmp_path / "cohort.parquet"
    # A duplicated row is deduped, not double-counted.
    pl.concat([cohort, cohort.head(1)]).write_parquet(cohort_fp)

    out_dir = tmp_path / "grid"
    # min_context_per_subject=1000 would sample nothing; the supplied cohort must be labeled verbatim.
    _run_seq(data_dir, out_dir, codes_yaml, contexts_path=cohort_fp, min_context_per_subject=1000)

    assert _unique(out_dir, "0").equals(cohort.filter(pl.col("subject_id") != 3))
    assert _unique(out_dir, "1").equals(cohort.filter(pl.col("subject_id") == 3))
    assert _labels(out_dir, "0").height == 4 * N_SEQ and _labels(out_dir, "1").height == 2 * N_SEQ


def test_supplied_event_bounded_sequence_is_labeled_end_to_end(
    tmp_path: Path, data_dir: Path, codes_yaml: Path, synthetic_events: pl.DataFrame
):
    """A mixed designed spec retains its bounds and uses them to decide the answers.

    A boundary that never occurs leaves the window open through the end of the patient's record.
    The third query therefore finds its target, while the fourth query confirms that an absent
    target still answers false rather than becoming an automatic positive. The fifth query is an
    ordinary one-day horizon and confirms that the later ``MED//E05`` does not count.
    """
    cohort = synthetic_events.group_by("subject_id").agg(pl.col("time").min().alias("prediction_time"))
    cohort_fp = tmp_path / "cohort.parquet"
    cohort.write_parquet(cohort_fp)

    never_boundary = "BOUNDARY//NEVER"
    never_target = "TARGET//NEVER"
    extended_codes_fp = tmp_path / "codes_with_never_boundary.yaml"
    extended_codes_fp.write_text(
        yaml.safe_dump([*yaml.safe_load(codes_yaml.read_text()), never_boundary, never_target])
    )

    specs_fp = tmp_path / "event_bounded.yaml"
    specs_fp.write_text(
        yaml.safe_dump(
            {
                "before_d": [
                    entry("ICD//C03", bound_event="MED//D04"),
                    entry("MED//E05", bound_event="MED//D04"),
                    entry("MED//E05", bound_event=never_boundary),
                    entry(never_target, bound_event=never_boundary),
                    entry("MED//E05", 1),
                    entry("ICD//A01", 100),
                ]
            }
        )
    )

    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, extended_codes_fp, contexts_path=cohort_fp, sequences_path=specs_fp)

    for shard in SHARDS:
        labels = _labels(out_dir, shard)
        QuerySeqSchema.align(pq.read_table(out_dir / "eval" / SPLIT / f"{shard}.parquet"))
        assert (
            labels["queries"].to_list()
            == [["ICD//C03", "MED//E05", "MED//E05", never_target, "MED//E05", "ICD//A01"]] * labels.height
        )
        assert labels["durations"].to_list() == [[-1.0, -1.0, -1.0, -1.0, 1.0, 100.0]] * labels.height
        assert (
            labels["bound_events"].to_list()
            == [["MED//D04", "MED//D04", never_boundary, never_boundary, None, None]] * labels.height
        )
        assert labels["answers"].to_list() == [[True, False, True, False, False, True]] * labels.height


def test_supplied_cohort_with_a_subject_outside_the_split_raises_before_writing(
    tmp_path: Path, data_dir: Path, codes_yaml: Path, synthetic_events: pl.DataFrame
):
    cohort_fp = tmp_path / "cohort.parquet"
    pl.DataFrame(
        {"subject_id": [1, 999], "prediction_time": [synthetic_events["time"][5]] * 2}
    ).write_parquet(cohort_fp)

    out_dir = tmp_path / "grid"
    with pytest.raises(ValueError, match=r"1 of 2 supplied subjects have no events"):
        _run_seq(data_dir, out_dir, codes_yaml, contexts_path=cohort_fp)
    assert not out_dir.exists()


# ── skip / overwrite ─────────────────────────────────────────────────────


def test_existing_outputs_are_skipped_unless_overwrite(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml)
    fp = out_dir / "eval" / SPLIT / "0.parquet"
    inode = fp.stat().st_ino

    _run_seq(data_dir, out_dir, codes_yaml)
    assert fp.stat().st_ino == inode, "second run must skip, not rewrite"

    _run_seq(data_dir, out_dir, codes_yaml, overwrite="true")
    assert fp.stat().st_ino != inode, "overwrite=true must regenerate"


def _inodes(out_dir: Path) -> list[int]:
    return [(out_dir / "eval" / SPLIT / f"{s}.parquet").stat().st_ino for s in SHARDS]


def test_changed_specs_relabel_instead_of_skipping(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    """The skip rule is keyed on the specs, not just on the output existing.

    Re-running into the same ``out_dir`` with a different sequence axis used to log "already
    exists, skipping" and keep the previous grid — a well-formed parquet answering the wrong
    questions.
    """
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml)
    before = _inodes(out_dir)

    _run_seq(data_dir, out_dir, codes_yaml, num_evaluation_sequences=N_SEQ + 1)
    assert all(a != b for a, b in zip(_inodes(out_dir), before, strict=True)), "more sequences must relabel"
    assert all(_labels(out_dir, s).height == _unique(out_dir, s).height * (N_SEQ + 1) for s in SHARDS)

    designed = tmp_path / "designed.yaml"
    codes = yaml.safe_load(codes_yaml.read_text())
    code = codes[0]
    designed.write_text(yaml.safe_dump({"one": [entry(code, 1)]}))
    before = _inodes(out_dir)
    _run_seq(data_dir, out_dir, codes_yaml, sequences_path=designed)
    assert all(a != b for a, b in zip(_inodes(out_dir), before, strict=True)), "designed specs must relabel"
    assert all(
        _spec_keys(_labels(out_dir, s)) == [f"{code}#1.0"] * _unique(out_dir, s).height for s in SHARDS
    )

    # Same designed file again: nothing changed, so nothing is rewritten.
    before = _inodes(out_dir)
    _run_seq(data_dir, out_dir, codes_yaml, sequences_path=designed)
    assert _inodes(out_dir) == before

    # Changing only the boundary changes both the labels and the task consumed by the model, so
    # it must participate in the provenance fingerprint just like the query and duration do.
    designed.write_text(yaml.safe_dump({"one": [entry(code, bound_event=codes[1])]}))
    _run_seq(data_dir, out_dir, codes_yaml, sequences_path=designed)
    before = _inodes(out_dir)
    designed.write_text(yaml.safe_dump({"one": [entry(code, bound_event=codes[2])]}))
    _run_seq(data_dir, out_dir, codes_yaml, sequences_path=designed)
    assert all(a != b for a, b in zip(_inodes(out_dir), before, strict=True)), (
        "a boundary-only spec change must relabel"
    )


def test_changed_cohort_relabels_instead_of_skipping(
    tmp_path: Path, data_dir: Path, codes_yaml: Path, synthetic_events: pl.DataFrame
):
    """...and on the cohort: sampling knobs for a sampled cohort, the frame itself for a supplied one."""
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml, prediction_times_per_subject=3)
    before = _inodes(out_dir)

    _run_seq(data_dir, out_dir, codes_yaml, prediction_times_per_subject=1)
    assert all(a != b for a, b in zip(_inodes(out_dir), before, strict=True)), "fewer times must relabel"
    assert all(_unique(out_dir, s).height == _unique(out_dir, s)["subject_id"].n_unique() for s in SHARDS)

    before = _inodes(out_dir)
    _run_seq(data_dir, out_dir, codes_yaml, prediction_times_per_subject=1, seed=2)
    assert all(a != b for a, b in zip(_inodes(out_dir), before, strict=True)), "a new seed must relabel"

    cohort = tmp_path / "cohort.parquet"
    first_times = synthetic_events.group_by("subject_id").agg(pl.col("time").min().alias("prediction_time"))
    first_times.write_parquet(cohort)
    before = _inodes(out_dir)
    _run_seq(data_dir, out_dir, codes_yaml, contexts_path=cohort)
    assert all(a != b for a, b in zip(_inodes(out_dir), before, strict=True)), (
        "a supplied cohort must relabel"
    )
    assert sum(_unique(out_dir, s).height for s in SHARDS) == first_times.height

    before = _inodes(out_dir)
    _run_seq(data_dir, out_dir, codes_yaml, contexts_path=cohort)
    assert _inodes(out_dir) == before, "the same supplied cohort must skip"

    first_times.head(2).write_parquet(cohort)
    _run_seq(data_dir, out_dir, codes_yaml, contexts_path=cohort)
    assert sum(_unique(out_dir, s).height for s in SHARDS) == 2, "an edited cohort file must relabel"


def test_missing_provenance_relabels_only_that_shard(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    """No sidecar means nothing vouches for that shard's parquet (a crash between the two writes):

    that shard alone is relabeled, the others are reused, and the sidecar comes back.
    """
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml)
    sidecars = {
        s: eval_seq._provenance_path(out_dir, out_dir / "eval" / SPLIT / f"{s}.parquet") for s in SHARDS
    }
    assert all(fp.is_file() for fp in sidecars.values())
    sidecars["0"].unlink()
    before = dict(zip(SHARDS, _inodes(out_dir), strict=True))

    _run_seq(data_dir, out_dir, codes_yaml)
    after = dict(zip(SHARDS, _inodes(out_dir), strict=True))
    assert after["0"] != before["0"], "the shard without provenance must be relabeled"
    assert after["1"] == before["1"], "a shard whose provenance is intact must be reused"
    assert sidecars["0"].is_file()


def test_a_pre_fingerprint_sidecar_is_stale(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    """A sidecar from before specs/cohort were recorded vouches for nothing and is relabeled once."""
    out_dir = tmp_path / "grid"
    _run_seq(data_dir, out_dir, codes_yaml)
    before = _inodes(out_dir)
    for shard in SHARDS:
        fp = out_dir / "eval" / SPLIT / f"{shard}.parquet"
        eval_seq._provenance_path(out_dir, fp).write_text('{"ontology_fingerprint": null}')

    _run_seq(data_dir, out_dir, codes_yaml)
    assert all(a != b for a, b in zip(_inodes(out_dir), before, strict=True))
    before = _inodes(out_dir)
    _run_seq(data_dir, out_dir, codes_yaml)
    assert _inodes(out_dir) == before


# ── config parity ────────────────────────────────────────────────────────


def test_config_cohort_keys_mirror_the_flat_sampler_and_sampling_keys_mirror_training():
    configs = Path(eval_seq.CONFIGS)
    seq_cfg = yaml.safe_load((configs / "sample_evaluation_query_sequences_config.yaml").read_text())
    flat_cfg = yaml.safe_load((configs / "sample_evaluation_tasks_config.yaml").read_text())
    train_cfg = yaml.safe_load((configs / "sample_query_sequences_config.yaml").read_text())

    cohort_keys = [
        "prediction_times_per_subject",
        "min_context_per_subject",
        "subject_subsample_fraction",
        "write_unique_prediction_times",
        "overwrite",
        "split",
        "seed",
    ]
    assert {k: seq_cfg[k] for k in cohort_keys} == {k: flat_cfg[k] for k in cohort_keys}
    for gone in ("n_contexts", "min_prediction_times_per_subject", "per_spec_dirs"):
        assert gone not in seq_cfg, f"{gone} was removed from the eval-seq config"

    sampling_keys = [
        "duration_min",
        "duration_max",
        "duration_distribution",
        "eos_first_fraction",
        "duration_mode",
        "eventbound_fraction",
        "ontology_dir",
    ]
    assert {k: seq_cfg[k] for k in sampling_keys} == {k: train_cfg[k] for k in sampling_keys}
