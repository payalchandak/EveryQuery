"""Explicit window starts in the dense evaluation grid generator (issue #27): specs, sampling, output.

Covered, in pipeline order:

1. ``SequenceSpec`` — every valid and invalid start representation.
2. Designed specs — the YAML mapping-entry form, JSON, the bare list, and the long-format parquet
   with optional ``start_duration_days`` / ``start_event`` columns; unknown start codes rejected.
3. Sampled specs — the start component is drawn on its own three seed axes, so (a) the default
   knobs reproduce the pre-#27 specs exactly and (b) any start knob leaves the query / duration /
   end-bound draw untouched; validation mirrors the multitask sampler's.
4. The grid — default output carries no start columns (schema-compatible with today); an active
   start puts both columns in the index and the parquet; the provenance fingerprint sees starts so
   a start-knob change relabels rather than serving a stale shard.

Parity reasoning, stated once (the issue asks for it here): the query / duration / end draw of the
eval grid is parity-anchored to ``query_sequence_labeling.py`` and must not move; only the start
component mirrors ``sample_multitask_sequences.py``'s ``BoundaryDistribution``.  The tests therefore
pin *both* halves — the legacy draw byte-for-byte, and the start draw's independence from it —
without claiming the whole sequence equals a multitask draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml
from hydra import compose, initialize_config_dir

from every_query.data.query_seq_dataset import EVENT_BOUND_DURATION_SENTINEL as SENTINEL
from every_query.data.schema import QuerySeqSchema
from every_query.generate_tasks import sample_evaluation_query_sequences as eval_seq
from every_query.generate_tasks.query_sequence_labeling import (
    BOUND_COL,
    START_DURATION_COL,
    START_EVENT_COL,
)
from every_query.generate_tasks.sample_evaluation_query_sequences import (
    SequenceSpec,
    _sample_starts,
    _specs_fingerprint,
    build_dense_sequence_index_df,
    read_sequence_specs,
    sample_sequence_specs,
    validate_spec_codes,
)
from tests.designed_specs import entry

SPLIT = "held_out"
SHARDS = ["0", "1"]


# ---------------------------------------------------------------------------
# 1. SequenceSpec representations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start_durations", "start_events", "active"),
    [
        ((), (), False),
        ((0.0, 0.0), (None, None), False),
        ((0, 7), (None, None), True),
        ((SENTINEL, 0.0), ("ADMIT", None), True),
        ((SENTINEL, SENTINEL), ("ADMIT", "DISCHARGE"), True),
        ((0.5, SENTINEL), (None, "ADMIT"), True),
    ],
)
def test_valid_start_representations(start_durations, start_events, active):
    spec = SequenceSpec(
        "s", ("A", "B"), (1.0, 2.0), start_durations=start_durations, start_events=start_events
    )
    assert spec.has_active_starts is active
    for i in range(2):
        sd, se = spec.start_at(i)
        assert isinstance(sd, float)
        assert (se is None) == (sd != SENTINEL)


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"start_durations": (7.0, 0.0)}, ValueError, "give both or neither"),
        ({"start_events": (None, None)}, ValueError, "give both or neither"),
        ({"start_durations": (7.0,), "start_events": (None,)}, ValueError, "2 queries but 1 start"),
        (
            {"start_durations": (7.0, 0.0), "start_events": ("ADMIT", None)},
            ValueError,
            "must be the -1.0 sentinel",
        ),
        (
            {"start_durations": (SENTINEL, 0.0), "start_events": (None, None)},
            ValueError,
            "finite number >= 0",
        ),
        ({"start_durations": (-2.0, 0.0), "start_events": (None, None)}, ValueError, "finite number >= 0"),
        (
            {"start_durations": (float("inf"), 0.0), "start_events": (None, None)},
            ValueError,
            "finite number >= 0",
        ),
        (
            {"start_durations": (float("nan"), 0.0), "start_events": (None, None)},
            ValueError,
            "finite number >= 0",
        ),
        ({"start_durations": (True, 0.0), "start_events": (None, None)}, TypeError, "must be a number"),
        (
            {"start_durations": (SENTINEL, 0.0), "start_events": ("", None)},
            ValueError,
            "non-empty string or null",
        ),
        (
            {"start_durations": (SENTINEL, 0.0), "start_events": (3, None)},
            ValueError,
            "non-empty string or null",
        ),
    ],
)
def test_invalid_start_representations(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        SequenceSpec("bad", ("A", "B"), (1.0, 2.0), **kwargs)


def test_spec_name_sanitising_keeps_the_starts():
    spec = SequenceSpec("a/b", ("A",), (1.0,), start_durations=(7.0,), start_events=(None,))
    (safe,) = eval_seq._sanitise_names([spec])
    assert safe.name == "a_b" and safe.start_durations == (7.0,) and safe.start_events == (None,)


# ---------------------------------------------------------------------------
# 2. Designed specs
# ---------------------------------------------------------------------------

DESIGNED = {
    "post_admission": [entry("LAB//X", 30, start_event="HOSPITAL_ADMISSION")],
    "delayed": [entry("ICD//I10", 30, start_duration_days=7)],
    "between_events": [
        # The sentinels spelled out, the way a script-generated file would; ``null`` is equivalent.
        entry("PROCEDURE//X", -1, start_event="HOSPITAL_ADMISSION", bound_event="HOSPITAL_DISCHARGE")
        | {"start_duration_days": -1}
    ],
    "mixed": [
        entry("TIMELINE//END", 1, forced_answer=False),
        entry("SEPSIS", bound_event="HOSPITAL_DISCHARGE"),
        entry("LAB//X", 3),
    ],
}


def _expected_designed() -> dict[str, tuple]:
    return {
        "post_admission": (("LAB//X",), (30.0,), (), (SENTINEL,), ("HOSPITAL_ADMISSION",), ()),
        "delayed": (("ICD//I10",), (30.0,), (), (7.0,), (None,), ()),
        "between_events": (
            ("PROCEDURE//X",),
            (-1.0,),
            ("HOSPITAL_DISCHARGE",),
            (SENTINEL,),
            ("HOSPITAL_ADMISSION",),
            (),
        ),
        "mixed": (
            ("TIMELINE//END", "SEPSIS", "LAB//X"),
            (1.0, -1.0, 3.0),
            (None, "HOSPITAL_DISCHARGE", None),
            (),
            (),
            (False, None, None),
        ),
    }


def _view(s: SequenceSpec) -> tuple:
    return (s.queries, s.durations, s.bounds, s.start_durations, s.start_events, s.forced_answers)


def _check_designed(specs: list[SequenceSpec]) -> None:
    assert {s.name: _view(s) for s in specs} == _expected_designed()


def test_yaml_mapping_form(tmp_path: Path):
    fp = tmp_path / "designed.yaml"
    fp.write_text(yaml.safe_dump(DESIGNED))
    _check_designed(read_sequence_specs(fp))


def test_json_form(tmp_path: Path):
    fp = tmp_path / "designed.json"
    fp.write_text(json.dumps(DESIGNED))
    _check_designed(read_sequence_specs(fp))


def test_bare_list_form(tmp_path: Path):
    fp = tmp_path / "designed.yaml"
    fp.write_text(yaml.safe_dump(list(DESIGNED.values())))
    specs = read_sequence_specs(fp)
    assert [s.name for s in specs] == [f"seq_{i:04d}" for i in range(4)]
    assert [_view(s) for s in specs] == list(_expected_designed().values())


def _long_format_rows() -> list[dict]:
    """``DESIGNED`` as long-format rows, out of position order within ``mixed``."""
    rows = [
        {"seq_id": name, "position": i, **e}
        for name, entries in DESIGNED.items()
        for i, e in enumerate(entries)
    ]
    return rows[::-1]


def _write_long_format(fp: Path, rows: list[dict]) -> Path:
    pl.DataFrame(rows).with_columns(
        pl.col("start_duration_days", "duration_days").cast(pl.Float64),
        pl.col("forced_answer").cast(pl.Boolean),
    ).write_parquet(fp)
    return fp


def test_long_format_parquet(tmp_path: Path):
    fp = _write_long_format(tmp_path / "designed.parquet", _long_format_rows())
    _check_designed(read_sequence_specs(fp))


@pytest.mark.parametrize(
    "column", ["duration_days", "bound_event", "start_duration_days", "start_event", "forced_answer"]
)
def test_long_format_parquet_must_carry_every_column(tmp_path: Path, column: str):
    rows = [{k: v for k, v in r.items() if k != column} for r in _long_format_rows()]
    fp = tmp_path / "designed.parquet"
    pl.DataFrame(rows).write_parquet(fp)
    with pytest.raises(ValueError, match=rf"missing required column\(s\) \['{column}'\]"):
        read_sequence_specs(fp)


@pytest.mark.parametrize("key", sorted(eval_seq._MAPPING_ENTRY_KEYS))
def test_every_key_of_a_designed_entry_is_required(tmp_path: Path, key: str):
    """``null`` is a legal value; a key that is simply not there is not."""
    fp = tmp_path / "sparse.yaml"
    full = entry("A", 30)
    fp.write_text(yaml.safe_dump({"s": [{k: v for k, v in full.items() if k != key}]}))
    with pytest.raises(ValueError, match=rf"missing required key\(s\) \['{key}'\]"):
        read_sequence_specs(fp)


@pytest.mark.parametrize("shorthand", [["A", 30], ["A", -1, "B"], "A"])
def test_the_list_shorthand_is_rejected(tmp_path: Path, shorthand):
    fp = tmp_path / "shorthand.yaml"
    fp.write_text(yaml.safe_dump({"s": [shorthand]}))
    with pytest.raises(ValueError, match="entry 0 must be a mapping with all of"):
        read_sequence_specs(fp)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"typo": 1}, "unknown key"),
        ({"start_event": "ADMIT", "start_duration_days": 7}, "sentinel"),
        ({"start_duration_days": -1}, "finite number >= 0"),
        ({"bound_event": "X"}, "sentinel"),
        ({"duration_days": None}, "neither a duration_days nor a bound_event"),
    ],
)
def test_contradictory_mapping_entries_are_rejected(tmp_path: Path, override, match):
    fp = tmp_path / "bad.yaml"
    fp.write_text(yaml.safe_dump({"s": [entry("A", 30) | override]}))
    with pytest.raises(ValueError, match=match):
        read_sequence_specs(fp)


# ---------------------------------------------------------------------------
# 2b. Forced answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".yaml", ".json", ".parquet"])
@pytest.mark.parametrize(
    "entries",
    [
        [entry("A", 1), entry("B", 30, forced_answer=True)],
        [entry("A", 1, forced_answer=False), entry("B", 30, forced_answer=False)],
        [entry("A", 1, forced_answer=True)],
    ],
    ids=["final-of-two", "every-position", "only-query"],
)
def test_the_final_query_may_never_be_forced(tmp_path: Path, suffix: str, entries: list[dict]):
    """The final query is the scored one: its answer conditions nothing, so a value there is refused
    in every supplied shape rather than silently ignored."""
    fp = tmp_path / f"bad{suffix}"
    if suffix == ".parquet":
        _write_long_format(fp, [{"seq_id": "s", "position": i, **e} for i, e in enumerate(entries)])
    elif suffix == ".json":
        fp.write_text(json.dumps({"s": entries}))
    else:
        fp.write_text(yaml.safe_dump({"s": entries}))
    with pytest.raises(ValueError, match=r"forces the answer of its final query .* must be null"):
        read_sequence_specs(fp)


@pytest.mark.parametrize("bad", [1, 0, "yes", "true"])
def test_a_forced_answer_is_strictly_boolean(tmp_path: Path, bad):
    fp = tmp_path / "bad.yaml"
    fp.write_text(yaml.safe_dump({"s": [entry("A", 1) | {"forced_answer": bad}, entry("B", 30)]}))
    with pytest.raises(TypeError, match="must be true, false or null"):
        read_sequence_specs(fp)


def test_a_file_that_forces_nothing_is_the_unforced_spec(tmp_path: Path):
    """Spelling ``forced_answer: null`` everywhere — which the strict format requires — must not change the
    fingerprint or put a ``forced_answers`` column in the grid."""
    fp = tmp_path / "plain.yaml"
    fp.write_text(yaml.safe_dump({"s": [entry("A", 1), entry("B", 30)]}))
    (spec,) = read_sequence_specs(fp)
    assert spec == SequenceSpec("s", ("A", "B"), (1.0, 30.0))
    assert not spec.has_forced_answers


def test_spec_name_sanitising_keeps_the_forced_answers():
    spec = SequenceSpec("a/b", ("A", "B"), (1.0, 1.0), forced_answers=(True, None))
    (safe,) = eval_seq._sanitise_names([spec])
    assert safe.name == "a_b" and safe.forced_answers == (True, None)


def test_unknown_start_codes_are_rejected_against_the_vocabulary():
    specs = [SequenceSpec("s", ("A",), (1.0,), start_durations=(SENTINEL,), start_events=("NOPE",))]
    with pytest.raises(ValueError, match="NOPE"):
        validate_spec_codes(specs, {"A", "B"})
    validate_spec_codes(specs, {"A", "NOPE"})


# ---------------------------------------------------------------------------
# 3. Sampled specs
# ---------------------------------------------------------------------------

CODES = ["A", "B", "C", "TIMELINE//END"]


def _draw(**kw) -> list[SequenceSpec]:
    base = {
        "n_sequences": 12,
        "query_codes": CODES,
        "min_queries": 1,
        "max_queries": 4,
        "duration_low": 1,
        "duration_high": 365,
        "seed": 7,
        "eventbound_fraction": 0.5,
    }
    base.update(kw)
    return sample_sequence_specs(**base)


def _legacy_view(specs: list[SequenceSpec]) -> list[tuple]:
    return [(s.queries, s.durations, s.bounds) for s in specs]


def _start_view(specs: list[SequenceSpec]) -> list[tuple]:
    return [tuple(s.start_at(i) for i in range(len(s))) for s in specs]


def test_default_start_knobs_reproduce_the_previous_draw_exactly():
    """Every window opens at the prediction time, the specs carry no start tuples, and the legacy (queries,
    durations, bounds) draw is the one the pre-#27 signature produced."""
    plain = _draw()
    spelled = _draw(eventstart_fraction=0.0, prediction_time_start_fraction=1.0)
    assert plain == spelled
    assert all(not s.start_durations and not s.start_events and not s.has_active_starts for s in plain)


# Literal output of ``sample_sequence_specs(3, ["A", "B", "C", "TIMELINE//END"], 1, 4, 1, 365, 7,
# eventbound_fraction=0.5)`` on the pre-#27 function (commit 494d7f5 of
# ``feat/prevalence-weighted-boundaries``), recorded so the draw is pinned against a *fixed* value
# rather than against the new function itself: rewiring the start component onto a legacy RNG axis
# would perturb these numbers and fail here, where a self-comparison would still pass.
_GOLDEN_LEGACY_DRAW = [
    (("A",), (47.47083986883709,), ()),
    (("A",), (-1.0,), ("A",)),
    (
        ("TIMELINE//END", "B", "TIMELINE//END", "TIMELINE//END"),
        (1.1336861389346837, -1.0, 11.378642731393592, 359.8491603781334),
        (None, "TIMELINE//END", None, None),
    ),
]


@pytest.mark.parametrize(
    "start_kwargs",
    [
        {},
        {"eventstart_fraction": 0.4, "prediction_time_start_fraction": 0.3, "start_event_codes": ["A", "B"]},
        {"eventstart_fraction": 0.0, "prediction_time_start_fraction": 0.0, "start_duration_max": 10.0},
    ],
    ids=["default-starts", "mixed-starts", "all-duration-starts"],
)
def test_legacy_draw_matches_the_pre_start_golden(start_kwargs):
    """The (queries, durations, bounds) draw is byte-identical to the pre-#27 sampler's, with the start knobs
    at their defaults and with every start form switched on."""
    specs = sample_sequence_specs(
        3, ["A", "B", "C", "TIMELINE//END"], 1, 4, 1, 365, 7, eventbound_fraction=0.5, **start_kwargs
    )
    assert _legacy_view(specs) == _GOLDEN_LEGACY_DRAW


def test_start_knobs_do_not_perturb_the_legacy_draw():
    plain = _draw()
    started = _draw(eventstart_fraction=0.3, prediction_time_start_fraction=0.3, start_event_codes=["A", "B"])
    assert _legacy_view(started) == _legacy_view(plain)
    assert [len(s) for s in started] == [len(s) for s in plain]
    assert any(s.has_active_starts for s in started)
    forms = {
        ("event" if e is not None else "pt" if d == 0 else "delay")
        for s in started
        for d, e in _start_view([s])[0]
    }
    assert forms == {"event", "pt", "delay"}


def test_legacy_knobs_do_not_perturb_the_start_component():
    """Changing an end-side knob changes the end draw only; the per-slot start forms/values stay put slot for
    slot (the start streams are drawn over the same total query count, so lengths must agree — which they do
    because lengths come from the untouched structure axis)."""
    a = _draw(eventstart_fraction=0.3, prediction_time_start_fraction=0.3, eventbound_fraction=0.0)
    b = _draw(eventstart_fraction=0.3, prediction_time_start_fraction=0.3, eventbound_fraction=1.0)
    assert _start_view(a) == _start_view(b)
    assert _legacy_view(a) != _legacy_view(b)


def test_each_start_axis_is_independent():
    base = {
        "eventstart_fraction": 0.4,
        "prediction_time_start_fraction": 0.2,
        "start_event_codes": ["A", "B"],
    }
    ref = _draw(**base)
    other_pool = _draw(**{**base, "start_event_codes": ["C"]})
    other_range = _draw(**{**base, "start_duration_min": 100, "start_duration_max": 200})

    def forms(specs):
        return [
            [("event" if e is not None else "pt" if d == 0 else "delay") for d, e in row]
            for row in _start_view(specs)
        ]

    # The pool only moves which event code is drawn; forms and delay durations are untouched.
    assert forms(other_pool) == forms(ref)
    assert [[d for d, e in row if e is None] for row in _start_view(other_pool)] == [
        [d for d, e in row if e is None] for row in _start_view(ref)
    ]
    assert {e for row in _start_view(other_pool) for _, e in row if e is not None} == {"C"}
    # The duration range only moves the delays; forms and event codes are untouched.
    assert forms(other_range) == forms(ref)
    assert [[e for _, e in row] for row in _start_view(other_range)] == [
        [e for _, e in row] for row in _start_view(ref)
    ]
    delays = [d for row in _start_view(other_range) for d, e in row if e is None and d > 0]
    assert delays and all(100 <= d <= 200 for d in delays)


def test_sampled_start_draw_is_deterministic():
    kw = {"eventstart_fraction": 0.5, "prediction_time_start_fraction": 0.25}
    assert _draw(**kw) == _draw(**kw)
    assert _draw(**kw) != _draw(**kw, seed=8)


def test_null_start_pool_is_the_query_universe():
    specs = _draw(eventstart_fraction=1.0, prediction_time_start_fraction=0.0, n_sequences=60)
    codes = {e for s in specs for e in s.start_events}
    assert codes == set(CODES)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"eventstart_fraction": 1.2}, "eventstart_fraction must be in"),
        ({"eventstart_fraction": -0.1}, "eventstart_fraction must be in"),
        ({"prediction_time_start_fraction": 1.5}, "prediction_time_start_fraction must be in"),
        ({"eventstart_fraction": 0.6, "prediction_time_start_fraction": 0.6}, "must be <= 1"),
        (
            {"eventstart_fraction": 0.5, "prediction_time_start_fraction": 0.5, "start_event_codes": []},
            "non-empty start_event_codes",
        ),
        ({"start_duration_min": 0}, "start_duration_min must be > 0"),
        ({"start_duration_min": 10, "start_duration_max": 5}, "start_duration_max"),
        ({"start_duration_distribution": "normal"}, "start_duration_distribution"),
        ({"start_event_codes": ["A", "NOPE"]}, "outside the query universe"),
    ],
)
def test_start_sampling_validation_mirrors_the_multitask_sampler(kw, match):
    with pytest.raises(ValueError, match=match):
        _draw(**kw)


def test_sample_starts_cumulative_split_matches_the_multitask_form_rule():
    """Same ``u`` thresholds as ``BoundaryDistribution.sample``: event below ``e``, prediction time below ``e
    + p``, delay above."""
    rng_u = np.random.default_rng(3)
    u = rng_u.random(500)
    d, e = _sample_starts(
        500,
        ["X"],
        0.2,
        0.3,
        1.0,
        5.0,
        "uniform",
        np.random.default_rng(3),
        np.random.default_rng(4),
        np.random.default_rng(5),
    )
    assert ((e != None) == (u < 0.2)).all()  # noqa: E711
    assert ((d == 0.0) == ((u >= 0.2) & (u < 0.5))).all()
    assert (d[(u >= 0.5)] >= 1.0).all() and (d[(u >= 0.5)] <= 5.0).all()


# ---------------------------------------------------------------------------
# 4. The grid
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path, synthetic_events: pl.DataFrame, write_split_shards) -> Path:
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


def _run(data_dir: Path, out_dir: Path, codes_yaml: Path, **overrides) -> None:
    kwargs = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "split": SPLIT,
        "num_evaluation_sequences": 4,
        "min_queries": 2,
        "max_queries": 2,
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


def _labels(out_dir: Path, shard: str) -> pl.DataFrame:
    return pl.read_parquet(out_dir / "eval" / SPLIT / f"{shard}.parquet")


# The start knobs that open every sampled window at the prediction time — the pre-#27 grid.  The
# shipped defaults draw active starts instead, so tests about that grid ask for it explicitly.
PREDICTION_TIME_STARTS = {"eventstart_fraction": 0.0, "prediction_time_start_fraction": 1.0}


def test_config_ships_the_start_keys():
    cfg = yaml.safe_load(
        (Path(eval_seq.CONFIGS) / "sample_evaluation_query_sequences_config.yaml").read_text()
    )
    assert cfg["eventstart_fraction"] == 0.2
    assert cfg["prediction_time_start_fraction"] == 0.4
    assert cfg["start_duration_min"] == 1 and cfg["start_duration_max"] == 180
    assert cfg["start_duration_distribution"] == "log-uniform"
    assert cfg["start_event_codes"] is None


def test_prediction_time_start_grid_carries_no_start_columns(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    """Schema-compatible with pre-#27: with every window opening at the prediction time the grid carries
    exactly the old columns — no start columns, and no ``forced_answers`` either."""
    out_dir = tmp_path / "grid"
    _run(data_dir, out_dir, codes_yaml, **PREDICTION_TIME_STARTS)
    for shard in SHARDS:
        df = _labels(out_dir, shard)
        assert df.columns == [
            "subject_id",
            "prediction_time",
            "queries",
            "durations",
            "answers",
            "bound_events",
        ]
        QuerySeqSchema.align(pq.read_table(out_dir / "eval" / SPLIT / f"{shard}.parquet"))


def test_default_grid_specs_are_the_previous_draw(
    tmp_path: Path, data_dir: Path, codes_yaml: Path, synthetic_query_codes
):
    """The rows the default config writes are the specs the pre-#27 sampler drew on this seed."""
    from every_query.utils.seeds import derive_seed

    out_dir = tmp_path / "grid"
    _run(data_dir, out_dir, codes_yaml)
    # The duration bounds come from the shipped config so this test tracks its defaults.
    shipped = yaml.safe_load(
        (Path(eval_seq.CONFIGS) / "sample_evaluation_query_sequences_config.yaml").read_text()
    )
    expected = sample_sequence_specs(
        4,
        synthetic_query_codes,
        2,
        2,
        float(shipped["duration_min"]),
        float(shipped["duration_max"]),
        derive_seed(1, "eval_seq_specs", SPLIT),
        eventbound_fraction=float(shipped["eventbound_fraction"]),
    )
    df = _labels(out_dir, "0").head(4)
    assert [tuple(q) for q in df["queries"].to_list()] == [s.queries for s in expected]
    # The parquet stores float32 horizons; the spec holds the float64 draw.
    got = [d for row in df["durations"].to_list() for d in row]
    assert got == [float(np.float32(d)) for s in expected for d in s.durations]
    assert [tuple(b) for b in df["bound_events"].to_list()] == [s.bounds for s in expected]


def test_sampled_starts_reach_the_output_and_label_through_the_start_path(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    out_dir = tmp_path / "grid"
    _run(
        data_dir,
        out_dir,
        codes_yaml,
        eventstart_fraction=0.5,
        prediction_time_start_fraction=0.0,
        num_evaluation_sequences=8,
    )
    for shard in SHARDS:
        df = _labels(out_dir, shard)
        assert {"start_durations", "start_events"} <= set(df.columns)
        QuerySeqSchema.align(pq.read_table(out_dir / "eval" / SPLIT / f"{shard}.parquet"))
        flat = df.explode("start_durations", "start_events")
        assert ((flat["start_durations"] == SENTINEL) == flat["start_events"].is_not_null()).all()
        assert (flat.filter(pl.col("start_events").is_null())["start_durations"] > 0).all()
        assert df["answers"].explode().null_count() == 0


def test_designed_starts_are_labeled_per_the_rule(
    tmp_path: Path, data_dir: Path, codes_yaml: Path, synthetic_events: pl.DataFrame
):
    """End to end through ``main`` on the synthetic cohort, checked row by row against the plain-Python oracle
    in ``tests/test_queryseq_starts.py`` (which shares no code with the labeler).

    The specs exercise every start/end form at once: an event start with an event end, an event
    start with a duration end, a duration start with an event end, and a start event that never
    occurs.  Comparing whole ``answers`` columns to the oracle — rather than asserting a few
    aggregate properties — is what makes a labeler that silently ignored the starts fail here.
    """
    from tests.test_queryseq_starts import _oracle

    # synthetic_events: per subject, codes cycle A01, B02, C03, D04, E05 every 10 days.
    specs = {
        # The forced answers ride along untouched by labeling: ``answers`` below is compared to
        # the oracle at *every* position, forced or not, so a labeler that wrote the forced value
        # into ``answers`` — and with it corrupted the truth — would fail here.
        "between": [
            entry("ICD//C03", start_event="ICD//B02", bound_event="ICD//C03", forced_answer=True),
            entry("MED//D04", start_event="ICD//B02", bound_event="MED//E05"),
        ],
        "after_b": [entry("ICD//C03", 15, start_event="ICD//B02")],
        "delayed": [
            entry("ICD//A01", start_duration_days=25, bound_event="MED//E05", forced_answer=False),
            entry("ICD//B02", 30, start_duration_days=25),
        ],
        "never": [entry("ICD//A01", start_event="MED//E05", bound_event="NOPE//X")],
    }
    fp = tmp_path / "specs.yaml"
    fp.write_text(yaml.safe_dump(specs))
    out_dir = tmp_path / "grid"
    with pytest.raises(ValueError, match="absent from the query vocabulary"):
        _run(data_dir, out_dir, codes_yaml, sequences_path=fp)
    specs["never"][0]["bound_event"] = "MED//D04"
    fp.write_text(yaml.safe_dump(specs))
    _run(data_dir, out_dir, codes_yaml, sequences_path=fp)
    df = pl.concat([_labels(out_dir, s) for s in SHARDS])
    # A spec that forces nothing keeps every context; a forced spec keeps only the contexts whose
    # truth agrees with it, and here that really does drop rows.
    n_contexts = df.select("subject_id", "prediction_time").n_unique()
    per_spec = dict(df.group_by(pl.col("queries").list.join("|")).len().rows())
    assert per_spec["ICD//C03"] == per_spec["ICD//A01"] == n_contexts
    assert df.height < n_contexts * len(specs)

    events = [tuple(r) for r in synthetic_events.select("subject_id", "time", "code").rows()]
    seen_forms: set[tuple] = set()
    for row in df.iter_rows(named=True):
        positions = zip(
            row["queries"],
            row["durations"],
            row["bound_events"],
            row["start_durations"],
            row["start_events"],
            row["answers"],
            strict=True,
        )
        for query, duration, bound, start_duration, start_event, answer in positions:
            spec = (query, float(duration), bound, float(start_duration), start_event)
            expected = _oracle(events, row["subject_id"], row["prediction_time"], spec)
            assert answer == expected, (row["subject_id"], row["prediction_time"], spec)
            seen_forms.add((start_event is not None, bound is not None))
    # Every start/end form combination was actually exercised.
    assert seen_forms == {(True, True), (True, False), (False, True), (False, False)}
    # Each row carries its own spec's forced answers, null-padded for the specs that force nothing.
    forced_by_queries = {
        tuple(e["query"] for e in entries): [e["forced_answer"] for e in entries]
        for entries in specs.values()
    }
    assert all(
        row["forced_answers"] == forced_by_queries[tuple(row["queries"])] for row in df.iter_rows(named=True)
    )
    # ...and every surviving row's truth agrees with each answer its spec forces.
    assert all(
        forced is None or forced == answer
        for row in df.iter_rows(named=True)
        for forced, answer in zip(row["forced_answers"], row["answers"], strict=True)
    )
    # The oracle is not vacuous: both answers occur across the grid.
    flat = df.explode("answers")
    assert flat["answers"].any() and not flat["answers"].all()


def test_yaml_null_start_keys_read_as_absent(tmp_path: Path):
    """An explicit ``start_duration_days: null`` is "absent", exactly as the parquet reader treats a
    null cell: an event start gets the sentinel, a duration-less entry the prediction time."""
    fp = tmp_path / "designed.yaml"
    fp.write_text(
        yaml.safe_dump(
            {
                "s": [
                    entry("A", 3, start_event="ADMIT"),
                    entry("B", 3, start_duration_days=None),
                ]
            }
        )
    )
    (spec,) = read_sequence_specs(fp)
    assert spec.start_durations == (SENTINEL, 0.0) and spec.start_events == ("ADMIT", None)


def test_flipping_a_forced_answer_relabels_and_an_unforced_file_writes_no_column(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    """A forced answer keeps the contexts whose truth matches it, so the two values partition the unforced
    grid; the fingerprint stands between a flipped ``forced_answer`` and the shard written under the other."""
    out_dir = tmp_path / "grid"
    fp = tmp_path / "specs.yaml"
    files = [out_dir / "eval" / SPLIT / f"{s}.parquet" for s in SHARDS]

    def run(forced: bool | None) -> list[list]:
        fp.write_text(
            yaml.safe_dump({"s": [entry("ICD//A01", 30, forced_answer=forced), entry("ICD//B02", 30)]})
        )
        _run(data_dir, out_dir, codes_yaml, sequences_path=fp)
        return [pl.read_parquet(f) for f in files]

    unforced = run(None)
    assert all("forced_answers" not in df.columns for df in unforced)

    yes = run(True)
    assert all(df["forced_answers"].to_list() == [[True, None]] * df.height for df in yes)
    no = run(False)
    assert all(df["forced_answers"].to_list() == [[False, None]] * df.height for df in no)
    # Forcing selects contexts, never rewrites what happened: YES and NO split the unforced rows.
    first_answer = pl.col("answers").list.first()
    for a, b, c in zip(unforced, yes, no, strict=True):
        assert b["answers"].to_list() == a.filter(first_answer)["answers"].to_list()
        assert c["answers"].to_list() == a.filter(~first_answer)["answers"].to_list()
    assert sum(df.height for df in yes) and sum(df.height for df in no)

    # Same file again: current, so untouched.
    before = [f.stat().st_ino for f in files]
    run(False)
    assert [f.stat().st_ino for f in files] == before


def test_start_knob_change_relabels_instead_of_serving_stale_shards(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    out_dir = tmp_path / "grid"
    _run(data_dir, out_dir, codes_yaml)
    files = [out_dir / "eval" / SPLIT / f"{s}.parquet" for s in SHARDS]
    stamps = {fp: fp.stat().st_mtime_ns for fp in files}
    _run(data_dir, out_dir, codes_yaml)  # same knobs: reused
    assert {fp: fp.stat().st_mtime_ns for fp in files} == stamps
    _run(data_dir, out_dir, codes_yaml, eventstart_fraction=0.5, prediction_time_start_fraction=0.5)
    assert all(fp.stat().st_mtime_ns != t for fp, t in stamps.items())
    assert all("start_events" in _labels(out_dir, s).columns for s in SHARDS)


def test_fingerprint_is_unchanged_by_spelled_out_default_starts_and_changed_by_active_ones():
    plain = [SequenceSpec("x", ("A", "B"), (1.0, 30.0))]
    spelled = [
        SequenceSpec("x", ("A", "B"), (1.0, 30.0), start_durations=(0.0, 0.0), start_events=(None, None))
    ]
    delayed = [
        SequenceSpec("x", ("A", "B"), (1.0, 30.0), start_durations=(0.0, 7.0), start_events=(None, None))
    ]
    event = [
        SequenceSpec("x", ("A", "B"), (1.0, 30.0), start_durations=(0.0, SENTINEL), start_events=(None, "A"))
    ]
    assert _specs_fingerprint(plain) == _specs_fingerprint(spelled)
    assert len({_specs_fingerprint(plain), _specs_fingerprint(delayed), _specs_fingerprint(event)}) == 3


def test_dense_index_carries_start_columns_only_for_active_starts():
    from datetime import datetime

    ctx = pl.DataFrame(
        {"subject_id": [1], "prediction_time": [datetime(2024, 1, 1)]},
        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")},
    )
    plain = build_dense_sequence_index_df(ctx, [SequenceSpec("a", ("X", "Y"), (1.0, 2.0))])
    assert START_DURATION_COL not in plain.columns and START_EVENT_COL not in plain.columns
    spelled = build_dense_sequence_index_df(
        ctx,
        [SequenceSpec("a", ("X", "Y"), (1.0, 2.0), start_durations=(0.0, 0.0), start_events=(None, None))],
    )
    assert spelled.equals(plain)
    started = build_dense_sequence_index_df(
        ctx,
        [
            SequenceSpec(
                "a", ("X", "Y"), (1.0, 2.0), start_durations=(7.0, SENTINEL), start_events=(None, "Z")
            ),
            SequenceSpec("b", ("X",), (-1.0,), bounds=("W",)),
        ],
    )
    assert started[START_DURATION_COL].to_list() == [7.0, SENTINEL, 0.0]
    assert started[START_EVENT_COL].to_list() == [None, "Z", None]
    assert started[BOUND_COL].to_list() == [None, None, "W"]
    assert started.schema[START_DURATION_COL] == pl.Float32
    empty = build_dense_sequence_index_df(
        ctx.head(0), [SequenceSpec("a", ("X",), (1.0,), start_durations=(7.0,), start_events=(None,))]
    )
    assert empty.height == 0 and START_EVENT_COL in empty.columns
