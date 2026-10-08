"""Semantic + differential tests for the multi-bound packed labeling kernel (``label_multitask_index``).

Every window rule from issue #20 is pinned directly, and the whole kernel is then compared - for every
``(context, boundary, code)`` - against the scalar ``label_with_event_bounds`` oracle on randomized
inputs.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from every_query.generate_tasks import sample_multitask_sequences as sms
from every_query.generate_tasks.interval_table import INF
from every_query.generate_tasks.sample_multitask_sequences import (
    TargetVocabulary,
    label_multitask_index,
    validate_index,
)
from tests.multitask.conftest import CODES, condition_answers_oracle, make_events, make_index, scalar_oracle
from tests.ontology_suite.golden import DECLARED_PARENTS, EVENTS, LEAVES, ONTOLOGY
from tests.ontology_suite.golden import T0 as GOLDEN_T0
from tests.ontology_suite.oracle import label_event_bounded
from tests.ontology_suite.production import events_to_frame

T0 = datetime(2024, 1, 1)
VOCAB = TargetVocabulary.from_pairs(["A", "B", "DISCHARGE", "TIMELINE//END"], [1, 2, 3, 4])


def _write_ontology(root: Path, codes: list[str], parents: dict[str, list[str]] | None = None) -> Path:
    """The three ``EQ_build_ontology`` artifacts for ``codes`` at leaf ids ``1..len(codes)``.

    Replicated from ``tests/test_multitask_ontology_targets.py`` rather than imported, so the two
    files never have to be edited together.
    """
    from every_query.data.ontology import (
        EMBEDDING_MIX_FILE,
        EVENT_TO_QUERY_NODES_FILE,
        ONTOLOGY_VOCAB_FILE,
        build_event_to_query_nodes,
        build_ontology,
    )
    from tests.ontology_suite.production import codes_frame

    nodes, mix = build_ontology(codes_frame(codes, parents or {}))
    root.mkdir(parents=True, exist_ok=True)
    nodes.write_parquet(root / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(root / EMBEDDING_MIX_FILE)
    build_event_to_query_nodes(nodes, mix).write_parquet(root / EVENT_TO_QUERY_NODES_FILE)
    return root


def _ancestor_names(onto: Path) -> list[str]:
    from every_query.data.ontology import load_nodes

    return load_nodes(onto).filter(~pl.col("is_observed_code"))["node_name"].to_list()


def _golden_vocab(tmp_path: Path, mode: str) -> tuple[Path, TargetVocabulary, TargetVocabulary]:
    """``(ontology_dir, leaf-only vocabulary, the same vocabulary widened by the ontology)``.

    The golden cohort's leaves at ids ``1..len(LEAVES)``, which is exactly what
    :func:`_write_ontology` numbers them as, so the cohort-identity check inside
    :func:`~every_query.generate_tasks.sample_multitask_sequences.attach_ontology` passes.
    """
    onto = _write_ontology(tmp_path / "onto", LEAVES, DECLARED_PARENTS)
    leaf_vocab = TargetVocabulary.from_pairs(LEAVES, list(range(1, len(LEAVES) + 1)))
    return onto, leaf_vocab, sms.attach_ontology(leaf_vocab, onto, mode)


def _golden_dense(packed: np.ndarray, vocab: TargetVocabulary) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)


def _events(rows: list[tuple[int, datetime | None, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
            "code": pl.Series([r[2] for r in rows], dtype=pl.Utf8),
        }
    )


def _dense(
    index_df: pl.DataFrame, events_df: pl.DataFrame, vocab: TargetVocabulary = VOCAB, k: int = 1, **kw
):
    meta, packed, _ = label_multitask_index(index_df, events_df, vocab, k, **kw)
    return meta, np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)


def _row(code: str, dense: np.ndarray, vocab: TargetVocabulary = VOCAB, k: int = 0) -> bool:
    return bool(dense[0, k, vocab.code_to_index()[code]])


# --- semantic tests -------------------------------------------------------------------------------


def test_occurrence_at_prediction_time_is_false_and_just_after_is_true() -> None:
    ev = _events([(1, T0, "A"), (1, T0 + timedelta(microseconds=1), "B")])
    _, d = _dense(make_index([(1, T0)], [[(10.0, None)]]), ev)
    assert _row("A", d) is False
    assert _row("B", d) is True


def test_occurrence_at_duration_endpoint_is_false_and_just_before_is_true() -> None:
    ev = _events([(1, T0 + timedelta(days=10), "A"), (1, T0 + timedelta(days=10, microseconds=-1), "B")])
    _, d = _dense(make_index([(1, T0)], [[(10.0, None)]]), ev)
    assert _row("A", d) is False
    assert _row("B", d) is True


def test_fractional_day_duration() -> None:
    ev = _events([(1, T0 + timedelta(hours=11), "A"), (1, T0 + timedelta(hours=13), "B")])
    _, d = _dense(make_index([(1, T0)], [[(0.5, None)]]), ev)
    assert _row("A", d) is True
    assert _row("B", d) is False


def test_event_bound_excludes_boundary_instant_and_boundary_code_itself() -> None:
    ev = _events(
        [
            (1, T0 + timedelta(days=1), "A"),
            (1, T0 + timedelta(days=3), "DISCHARGE"),
            (1, T0 + timedelta(days=3), "B"),  # co-timestamped with the boundary
            (1, T0 + timedelta(days=5), "TIMELINE//END"),
        ]
    )
    _, d = _dense(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]]), ev)
    assert _row("A", d) is True
    assert _row("B", d) is False
    assert _row("DISCHARGE", d) is False
    assert _row("TIMELINE//END", d) is False


def test_every_code_co_timestamped_with_boundary_is_excluded() -> None:
    t = T0 + timedelta(days=2)
    ev = _events([(1, t, "A"), (1, t, "B"), (1, t, "DISCHARGE"), (1, t, "TIMELINE//END")])
    _, d = _dense(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]]), ev)
    assert d.sum() == 0


def test_missing_boundary_event_means_infinity() -> None:
    ev = _events([(1, T0 + timedelta(days=900), "A"), (1, T0 + timedelta(days=901), "TIMELINE//END")])
    _, d = _dense(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]]), ev)
    assert _row("A", d) is True
    assert _row("TIMELINE//END", d) is True
    assert _row("DISCHARGE", d) is False


def test_boundary_occurrence_at_prediction_time_does_not_close_window() -> None:
    ev = _events(
        [(1, T0, "DISCHARGE"), (1, T0 + timedelta(days=1), "A"), (1, T0 + timedelta(days=4), "DISCHARGE")]
    )
    _, d = _dense(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]]), ev)
    assert _row("A", d) is True


def test_no_later_occurrence_is_false() -> None:
    ev = _events([(1, T0 - timedelta(days=1), "A"), (1, T0 + timedelta(days=3), "B")])
    _, d = _dense(make_index([(1, T0)], [[(30.0, None)]]), ev)
    assert _row("A", d) is False
    assert _row("B", d) is True


def test_duplicate_events_and_null_time_events() -> None:
    ev = _events(
        [
            (1, T0 + timedelta(days=1), "A"),
            (1, T0 + timedelta(days=1), "A"),
            (1, None, "B"),  # static: ignored
            (1, None, "DISCHARGE"),
        ]
    )
    _, d = _dense(make_index([(1, T0)], [[(10.0, None), (-1.0, "DISCHARGE")]]), ev, k=2)
    assert d[0, 0].tolist() == [False, True, False, False, False]
    assert d[0, 1].tolist() == [False, True, False, False, False]  # boundary never occurs in time => inf


def test_timeline_end_is_an_ordinary_target() -> None:
    ev = _events([(1, T0 + timedelta(days=2), "TIMELINE//END")])
    _, d = _dense(make_index([(1, T0), (1, T0)], [[(3.0, None)], [(1.0, None)]]), ev)
    assert d[:, 0, VOCAB.code_to_index()["TIMELINE//END"]].tolist() == [True, False]


def test_pad_bit_stays_false_and_unknown_event_codes_are_ignored() -> None:
    vocab = TargetVocabulary.from_pairs(["PAD", "A"], [0, 1])
    ev = _events(
        [
            (1, T0 + timedelta(days=1), "PAD"),
            (1, T0 + timedelta(days=1), "A"),
            (1, T0 + timedelta(days=1), "ZZZ"),
        ]
    )
    _, packed, stats = label_multitask_index(make_index([(1, T0)], [[(5.0, None)]]), ev, vocab, 1)
    d = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
    assert d[0, 0].tolist() == [False, True]  # an observed PAD event never sets bit 0
    assert stats.n_unknown_code_events == 1
    assert vocab.boundary_candidates() == ["A"]


def test_all_unknown_event_codes_is_a_hard_error() -> None:
    """A shard where *every* event is out-of-vocabulary is the wrong input (string-coded events),
    not a sparse shard: it must raise instead of labeling everything false."""
    vocab = TargetVocabulary.from_pairs(["PAD", "A"], [0, 1])
    ev = _events([(1, T0 + timedelta(days=1), "ZZZ"), (1, T0 + timedelta(days=2), "YYY")])
    with pytest.raises(ValueError, match="outside the target vocabulary"):
        label_multitask_index(make_index([(1, T0)], [[(5.0, None)]]), ev, vocab, 1)


def test_unknown_boundary_code_is_a_hard_error() -> None:
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    with pytest.raises(ValueError, match="not in the base vocabulary"):
        label_multitask_index(make_index([(1, T0)], [[(-1.0, "NOPE")]]), ev, VOCAB, 1)


def test_pad_boundary_code_is_a_hard_error() -> None:
    vocab = TargetVocabulary.from_pairs(["PAD", "A"], [0, 1])
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    with pytest.raises(ValueError, match="PAD"):
        label_multitask_index(make_index([(1, T0)], [[(-1.0, "PAD")]]), ev, vocab, 1)


def test_validate_index_rejects_wrong_arity_and_mixed_representation() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        validate_index(make_index([(1, T0)], [[(1.0, None)]]), 2)
    with pytest.raises(ValueError, match="either duration"):
        validate_index(make_index([(1, T0)], [[(5.0, "A")]]), 1)
    with pytest.raises(ValueError, match="either duration"):
        validate_index(make_index([(1, T0)], [[(-1.0, None)]]), 1)
    with pytest.raises(ValueError, match="missing required column"):
        validate_index(make_index([(1, T0)], [[(1.0, None)]]).drop("durations"), 1)


def test_ontology_dir_and_vocab_mode_must_agree(tmp_path: Path) -> None:
    """An ``ontology_dir`` is no longer refused outright - it must *agree* with the vocabulary it is
    labeling against, in both directions, because a leaf-only vocabulary cannot encode an ancestor and
    an ontology-bearing one was drawn against an event universe the caller must supply.

    Also pins the seam-2 contract: without a directory the event stream is returned untouched (the
    identity that keeps leaf labeling byte-identical), and with one it is closure-expanded.
    """
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    with pytest.raises(ValueError, match="disagrees with the vocabulary's ontology_mode"):
        label_multitask_index(make_index([(1, T0)], [[(1.0, None)]]), ev, VOCAB, 1, ontology_dir="x")

    codes = ["A//1", "A//2", "TIMELINE//END"]
    onto = _write_ontology(tmp_path / "onto", codes)
    vocab = sms.attach_ontology(TargetVocabulary.from_pairs(codes, [1, 2, 3]), onto, "boundaries")
    assert vocab.ontology_mode == "boundaries", "attach_ontology carries the mode onto the vocabulary"
    assert vocab.boundary_size > vocab.size, "the ancestors widen the *event* vocabulary only"
    ev2 = _events([(1, T0 + timedelta(days=1), "A//1")])
    idx = make_index([(1, T0)], [[(5.0, None)]])
    with pytest.raises(ValueError, match="disagrees with the vocabulary's ontology_mode"):
        label_multitask_index(idx, ev2, vocab, 1, ontology_dir=None)
    _, packed, _ = label_multitask_index(idx, ev2, vocab, 1, ontology_dir=str(onto))
    assert packed.any(), "with the matching directory the same index labels normally"

    assert sms.prepare_events_for_labeling(ev2, None) is ev2, "no ontology: the very same frame back"
    expanded = sms.prepare_events_for_labeling(ev2, str(onto))
    assert expanded.height > ev2.height, "the closure repeats each event under its ancestors"
    assert set(expanded["code"].to_list()) == {"A//1", "A"}, "the ancestor node name is carried as a code"


def test_vocab_size_not_divisible_by_eight_and_absent_codes_keep_bits_false() -> None:
    vocab = TargetVocabulary.from_pairs(["A", "B", "Z"], [1, 2, 12])  # V = 13 -> 2 packed bytes
    ev = _events([(1, T0 + timedelta(days=1), "A"), (1, T0 + timedelta(days=2), "B")])
    _, packed, _ = label_multitask_index(make_index([(1, T0)], [[(5.0, None)]]), ev, vocab, 1)
    assert packed.shape == (1, 1, 2)
    d = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
    assert d.shape == (1, 1, 13)
    assert d[0, 0, [1, 2]].tolist() == [True, True]
    assert not d[0, 0, 12] and not d[0, 0, 0]


# --- ordering / chunking / supplied-index contract -------------------------------------------------


def test_output_identical_for_different_initial_context_order_and_chunk_sizes() -> None:
    rng = np.random.default_rng(7)
    events = make_events(rng, [1, 2, 3], codes=CODES)
    vocab = TargetVocabulary.from_pairs(CODES, list(range(1, len(CODES) + 1)))
    contexts = [(int(rng.integers(1, 4)), T0 + timedelta(days=int(rng.integers(0, 400)))) for _ in range(30)]
    bounds = [
        [
            (-1.0, CODES[int(rng.integers(0, 11))])
            if rng.random() < 0.5
            else (float(rng.uniform(1, 60)), None)
            for _ in range(3)
        ]
        for _ in contexts
    ]
    conds = [[CODES[int(rng.integers(0, len(CODES)))] for _ in range(2)] for _ in contexts]
    idx = make_index(contexts, bounds, conds)
    meta_a, packed_a, _ = label_multitask_index(idx, events, vocab, 3, chunk_rows=4)
    perm = rng.permutation(idx.height)
    shuffled = idx.with_row_index("_ctx_id").with_columns(pl.col("_ctx_id").cast(pl.Int64))[perm.tolist()]
    meta_b, packed_b, _ = label_multitask_index(shuffled, events, vocab, 3, chunk_rows=1000)
    assert meta_a.equals(meta_b)
    assert np.array_equal(packed_a, packed_b)
    assert meta_a.sort("subject_id", "prediction_time").equals(meta_a)


def test_supplied_out_memmap_is_filled_in_place(tmp_path) -> None:
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    idx = make_index([(1, T0)], [[(5.0, None)]])
    mm = np.lib.format.open_memmap(
        tmp_path / "x.npy", mode="w+", dtype=np.uint8, shape=(1, 1, VOCAB.packed_width)
    )
    _, out, _ = label_multitask_index(idx, ev, VOCAB, 1, out=mm)
    assert out is mm
    mm.flush()
    del mm
    assert np.load(tmp_path / "x.npy")[0, 0, 0] == 0b10
    with pytest.raises(ValueError, match="out must be uint8"):
        label_multitask_index(idx, ev, VOCAB, 1, out=np.zeros((2, 1, 1), np.uint8))


# --- differential oracle ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_differential_against_scalar_label_with_event_bounds(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k = 5
    subjects = [1, 2, 3, 4]
    events = make_events(rng, subjects, n_times=(5, 25), horizon_days=60, codes=CODES)
    vocab = TargetVocabulary.from_pairs(CODES, list(range(1, len(CODES) + 1)))
    obs = CODES[:11]
    n = 24
    contexts = []
    for _ in range(n):
        sid = int(rng.integers(1, 6))  # 5 is an unknown subject: every bit false
        # Prediction times deliberately land on event instants often, to exercise the strict bounds.
        sub = events.filter(pl.col("subject_id") == sid)["time"].drop_nulls()
        if sub.len() and rng.random() < 0.7:
            pt = sub[int(rng.integers(0, sub.len()))]
        else:
            pt = T0 + timedelta(days=int(rng.integers(0, 60)))
        contexts.append((sid, pt))

    def pick_bound():
        if rng.random() < 0.5:
            return float(rng.uniform(0.1, 40)), None
        pool = CODES if rng.random() < 0.2 else obs  # TIMELINE//END is an eligible boundary too
        return -1.0, pool[int(rng.integers(0, len(pool)))]

    bounds = [[pick_bound() for _ in range(k)] for _ in range(n)]
    conds = [[CODES[int(rng.integers(0, len(CODES)))] for _ in range(k - 1)] for _ in range(n)]
    idx = make_index(contexts, bounds, conds)
    meta, packed, stats = label_multitask_index(idx, events, vocab, k, chunk_rows=5)
    dense = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
    got = dense[:, :, vocab.indices]
    expect = scalar_oracle(meta, events, list(vocab.codes), k)
    assert np.array_equal(got, expect)
    assert stats.n_contexts == n
    # condition_answers[i, j] == targets[i, j, index(condition_codes[i, j])], with repeats allowed.
    answers = np.array(meta["condition_answers"].to_list(), dtype=bool)
    assert answers.shape == (n, k - 1)
    assert np.array_equal(answers, condition_answers_oracle(meta, dense, vocab))
    assert answers.any() and not answers.all()


# --- conditioning codes (issue #22) ---------------------------------------------------------------


def test_condition_answer_is_the_target_bit_at_the_matching_boundary() -> None:
    ev = _events([(1, T0 + timedelta(days=2), "A"), (1, T0 + timedelta(days=8), "B")])
    idx = make_index([(1, T0)], [[(5.0, None), (10.0, None), (1.0, None)]], [["A", "A"]])
    meta, _ = _dense(idx, ev, k=3)
    assert meta["condition_answers"].to_list() == [[True, True]]
    idx = make_index([(1, T0)], [[(5.0, None), (10.0, None), (1.0, None)]], [["B", "B"]])
    meta, _ = _dense(idx, ev, k=3)
    assert meta["condition_answers"].to_list() == [[False, True]]  # B at day 8: outside 5d, inside 10d
    assert meta["condition_codes"].to_list() == [["B", "B"]]  # repeats are allowed and carried through


def test_single_bound_has_no_condition_slots() -> None:
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    meta, _ = _dense(make_index([(1, T0)], [[(5.0, None)]]), ev)
    assert meta["condition_codes"].to_list() == [[]] and meta["condition_answers"].to_list() == [[]]


def test_bad_condition_codes_are_hard_errors() -> None:
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    two = [[(5.0, None), (6.0, None)]]
    with pytest.raises(ValueError, match="exactly 1 condition_codes"):
        validate_index(make_index([(1, T0)], two, [["A", "B"]]), 2)
    with pytest.raises(ValueError, match="exactly 1 condition_codes"):
        validate_index(make_index([(1, T0)], two, [[]]), 2)
    with pytest.raises(ValueError, match="missing required column 'condition_codes'"):
        validate_index(make_index([(1, T0)], two).drop("condition_codes"), 2)
    with pytest.raises(ValueError, match="must not contain nulls"):
        validate_index(make_index([(1, T0)], two, [[None]]), 2)
    with pytest.raises(ValueError, match="condition code\\(s\\) are not in the base vocabulary"):
        label_multitask_index(make_index([(1, T0)], two, [["NOPE"]]), ev, VOCAB, 2)
    vocab = TargetVocabulary.from_pairs(["PAD", "A"], [0, 1])
    with pytest.raises(ValueError, match="condition code\\(s\\) at vocab index 0"):
        label_multitask_index(make_index([(1, T0)], two, [["PAD"]]), ev, vocab, 2)


# --- issue #24: explicit window starts -------------------------------------------------------------

VOCAB24 = TargetVocabulary.from_pairs(["A", "B", "X", "Y", "Z", "DISCHARGE"], [1, 2, 3, 4, 5, 6])


def _day(n: float) -> datetime:
    return T0 + timedelta(days=n)


def _dense24(index_df, events_df, k: int = 1, **kw):
    return _dense(index_df, events_df, VOCAB24, k, **kw)


def _labels(dense: np.ndarray, k: int = 0, vocab: TargetVocabulary = VOCAB24) -> dict[str, bool]:
    return {c: bool(dense[0, k, i]) for c, i in vocab.code_to_index().items()}


def test_issue_fixture_event_start_event_end_only_y_is_true() -> None:
    """The issue's day-3/5/5/6/8/8 fixture: start A, end B -> window (day 5, day 8) -> only Y."""
    ev = _events(
        [
            (1, _day(3), "B"),
            (1, _day(5), "A"),
            (1, _day(5), "X"),
            (1, _day(6), "Y"),
            (1, _day(8), "B"),
            (1, _day(8), "Z"),
        ]
    )
    idx = make_index([(1, T0)], [[(-1.0, "B")]], starts=[[(-1.0, "A")]])
    meta, d = _dense24(idx, ev)
    got = _labels(d)
    assert got == {"A": False, "X": False, "Y": True, "B": False, "Z": False, "DISCHARGE": False}
    assert meta["start_durations"].to_list() == [[-1.0]] and meta["start_events"].to_list() == [["A"]]


def test_zero_start_duration_reproduces_existing_labels_byte_identically() -> None:
    rng = np.random.default_rng(11)
    events = make_events(rng, [1, 2, 3], codes=CODES)
    vocab = TargetVocabulary.from_pairs(CODES, list(range(1, len(CODES) + 1)))
    contexts = [(int(rng.integers(1, 4)), T0 + timedelta(days=int(rng.integers(0, 400)))) for _ in range(30)]
    bounds = [
        [
            (-1.0, CODES[int(rng.integers(0, 11))])
            if rng.random() < 0.5
            else (float(rng.uniform(1, 60)), None)
            for _ in range(3)
        ]
        for _ in contexts
    ]
    conds = [[CODES[int(rng.integers(0, len(CODES)))] for _ in range(2)] for _ in contexts]
    legacy = make_index(contexts, bounds, conds)  # no start columns at all
    explicit = make_index(contexts, bounds, conds, starts=[[(0.0, None)] * 3 for _ in contexts])
    meta_a, packed_a, _ = label_multitask_index(legacy, events, vocab, 3, chunk_rows=4)
    meta_b, packed_b, _ = label_multitask_index(explicit, events, vocab, 3, chunk_rows=4)
    assert packed_a.tobytes() == packed_b.tobytes()
    assert meta_a.equals(meta_b)
    assert meta_a["start_durations"].to_list() == [[0.0, 0.0, 0.0]] * 30
    assert meta_a["start_events"].to_list() == [[None, None, None]] * 30
    # ... and both equal the scalar (prediction-time-start) oracle bit for bit.
    dense = np.unpackbits(packed_a, axis=-1, count=vocab.size, bitorder="little").astype(bool)
    assert np.array_equal(dense[:, :, vocab.indices], scalar_oracle(meta_a, events, list(vocab.codes), 3))


def test_positive_duration_start_and_duration_end_relative_to_start() -> None:
    """Pt = day 0, start 7 days, duration 30 -> window (day 7, day 37), never (day 7, day 30)."""
    ev = _events(
        [(1, _day(5), "A"), (1, _day(10), "B"), (1, _day(36), "X"), (1, _day(37), "Y"), (1, _day(38), "Z")]
    )
    _, d = _dense24(make_index([(1, T0)], [[(30.0, None)]], starts=[[(7.0, None)]]), ev)
    assert _labels(d) == {"A": False, "B": True, "X": True, "Y": False, "Z": False, "DISCHARGE": False}
    # Exactly at the start instant is excluded, just after is included.
    ev = _events([(1, _day(7), "A"), (1, _day(7) + timedelta(microseconds=1), "B")])
    _, d = _dense24(make_index([(1, T0)], [[(30.0, None)]], starts=[[(7.0, None)]]), ev)
    assert _labels(d)["A"] is False and _labels(d)["B"] is True


def test_duration_end_is_relative_to_an_event_start() -> None:
    ev = _events(
        [(1, _day(3), "X"), (1, _day(10), "A"), (1, _day(12), "B"), (1, _day(15), "Y"), (1, _day(16), "Z")]
    )
    _, d = _dense24(make_index([(1, T0)], [[(5.0, None)]], starts=[[(-1.0, "A")]]), ev)
    # (day 10, day 15): X at day 3 is before the start, Y sits on the endpoint.
    assert _labels(d) == {"A": False, "B": True, "X": False, "Y": False, "Z": False, "DISCHARGE": False}


def test_event_end_is_strictly_after_the_resolved_start_not_the_prediction_time() -> None:
    """Start A at day 10; B at days 8 and 20 -> (day 10, day 20): the day-8 boundary is ignored."""
    ev = _events(
        [
            (1, _day(8), "B"),
            (1, _day(10), "A"),
            (1, _day(15), "X"),
            (1, _day(20), "B"),
            (1, _day(20), "Y"),
            (1, _day(25), "Z"),
        ]
    )
    _, d = _dense24(make_index([(1, T0)], [[(-1.0, "B")]], starts=[[(-1.0, "A")]]), ev)
    assert _labels(d) == {"A": False, "B": False, "X": True, "Y": False, "Z": False, "DISCHARGE": False}


def test_start_event_exactly_at_prediction_time_is_skipped() -> None:
    ev = _events([(1, T0, "A"), (1, _day(2), "X"), (1, _day(3), "A"), (1, _day(5), "Y")])
    _, d = _dense24(make_index([(1, T0)], [[(10.0, None)]], starts=[[(-1.0, "A")]]), ev)
    # The window opens at the day-3 occurrence: X at day 2 is outside, Y at day 5 inside.
    assert _labels(d) == {"A": False, "B": False, "X": False, "Y": True, "Z": False, "DISCHARGE": False}


def test_co_timestamped_start_and_end_events_are_excluded() -> None:
    ev = _events(
        [(1, _day(5), "A"), (1, _day(5), "X"), (1, _day(6), "Y"), (1, _day(8), "B"), (1, _day(8), "Z")]
    )
    _, d = _dense24(make_index([(1, T0)], [[(-1.0, "B")]], starts=[[(-1.0, "A")]]), ev)
    assert _labels(d)["X"] is False and _labels(d)["Z"] is False and _labels(d)["Y"] is True
    # Same instants under a duration end landing exactly on day 8.
    _, d = _dense24(make_index([(1, T0)], [[(3.0, None)]], starts=[[(-1.0, "A")]]), ev)
    assert _labels(d) == {"A": False, "B": False, "X": False, "Y": True, "Z": False, "DISCHARGE": False}


def test_missing_start_event_gives_an_empty_all_false_window() -> None:
    """DISCHARGE never occurs after pt: no fallback to the prediction time, both end forms empty."""
    ev = _events([(1, _day(-1), "DISCHARGE"), (1, _day(1), "A"), (1, _day(2), "B"), (1, _day(900), "X")])
    idx = make_index(
        [(1, T0)],
        [[(30.0, None), (-1.0, "B"), (1826.0, None)]],
        [["A", "A"]],
        starts=[[(-1.0, "DISCHARGE")] * 3],
    )
    meta, packed, stats = label_multitask_index(idx, ev, VOCAB24, 3)
    assert packed.sum() == 0
    assert meta["condition_answers"].to_list() == [[False, False]]
    assert stats.n_event_starts == 3 and stats.frac_event_starts_unresolved == 1.0
    assert stats.frac_empty_windows == 1.0 and stats.mean_positives_per_window == 0.0
    assert stats.n_event_bounds == 1 and stats.frac_event_bounds_inf == 0.0  # only over resolved starts


def test_missing_end_event_after_an_event_start_is_unbounded() -> None:
    ev = _events([(1, _day(2), "X"), (1, _day(5), "A"), (1, _day(900), "Y"), (1, _day(901), "Z")])
    _, d = _dense24(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]], starts=[[(-1.0, "A")]]), ev)
    assert _labels(d) == {"A": False, "B": False, "X": False, "Y": True, "Z": True, "DISCHARGE": False}


def test_finite_start_after_every_event_is_an_empty_window() -> None:
    ev = _events([(1, _day(1), "A"), (1, _day(2), "B")])
    idx = make_index([(1, T0)], [[(30.0, None), (-1.0, "B")]], [["A"]], starts=[[(500.0, None)] * 2])
    _, packed, stats = label_multitask_index(idx, ev, VOCAB24, 2)
    assert packed.sum() == 0
    assert stats.n_event_starts == 0 and stats.frac_event_starts_unresolved == 0.0
    assert stats.frac_empty_windows == 1.0
    assert stats.frac_event_bounds_inf == 1.0  # the start resolved (day 500) but B never recurs after it


def test_equal_start_and_end_codes_select_consecutive_occurrences() -> None:
    ev = _events(
        [(1, _day(2), "A"), (1, _day(4), "X"), (1, _day(6), "A"), (1, _day(7), "Y"), (1, _day(9), "A")]
    )
    _, d = _dense24(make_index([(1, T0)], [[(-1.0, "A")]], starts=[[(-1.0, "A")]]), ev)
    # (day 2, day 6): X inside, Y after the second A, A itself never inside.
    assert _labels(d) == {"A": False, "B": False, "X": True, "Y": False, "Z": False, "DISCHARGE": False}
    # From a prediction time between the first two A's the pair shifts to (day 6, day 9).
    _, d = _dense24(make_index([(1, _day(3))], [[(-1.0, "A")]], starts=[[(-1.0, "A")]]), ev)
    assert _labels(d) == {"A": False, "B": False, "X": False, "Y": True, "Z": False, "DISCHARGE": False}


def test_all_six_start_end_combinations_in_one_context() -> None:
    from every_query.generate_tasks.interval_table import US_PER_DAY, naive_window_labels

    rows = [
        (1, _day(1), "X"),
        (1, _day(3), "A"),
        (1, _day(4), "Y"),
        (1, _day(6), "B"),
        (1, _day(8), "Z"),
        (1, _day(10), "A"),
        (1, _day(12), "B"),
        (1, _day(20), "DISCHARGE"),
    ]
    ev = _events(rows)
    starts = [(0.0, None), (0.0, None), (2.0, None), (2.0, None), (-1.0, "A"), (-1.0, "A")]
    bounds = [(5.0, None), (-1.0, "B"), (5.0, None), (-1.0, "B"), (5.0, None), (-1.0, "B")]
    idx = make_index([(1, T0)], [bounds], [["X"] * 5], starts=[starts])
    meta, d = _dense24(idx, ev, k=6, chunk_rows=1)
    c2i = VOCAB24.code_to_index()
    events = [(s, int((t - datetime(1970, 1, 1)).total_seconds() * 1_000_000), c2i[c]) for s, t, c in rows]
    windows = [
        [
            ((sd, None) if se is None else (None, c2i[se]), (ed, None) if be is None else (None, c2i[be]))
            for (sd, se), (ed, be) in zip(starts, bounds, strict=True)
        ]
    ]
    pt_us = int((T0 - datetime(1970, 1, 1)).total_seconds() * 1_000_000)
    st, en, expect = naive_window_labels(events, [(1, pt_us)], windows, VOCAB24.size, 6)
    assert np.array_equal(d, expect)
    day = US_PER_DAY
    assert ((st - pt_us) // day).tolist() == [[0, 0, 2, 2, 3, 3]]
    assert ((en - pt_us) // day).tolist() == [[5, 6, 7, 6, 8, 6]]
    # Spot checks: (0,5) X,A,Y; (0,6) X,A,Y; (2,7) A,Y,B; (2,6) A,Y; (3,8) Y,B; (3,6) Y.
    names = [sorted(c for c, i in c2i.items() if d[0, k, i]) for k in range(6)]
    assert names == [["A", "X", "Y"], ["A", "X", "Y"], ["A", "B", "Y"], ["A", "Y"], ["B", "Y"], ["Y"]]
    # Conditioning answers come from the resolved windows: X (day 1) is only inside the two
    # prediction-time-start windows.
    assert meta["condition_answers"].to_list() == [[True, True, False, False, False]]


def test_conditioning_answers_use_the_resolved_window() -> None:
    ev = _events([(1, _day(3), "X"), (1, _day(10), "A"), (1, _day(12), "Y")])
    # Windows: (pt, 30d) and (A -> +5d).  X at day 3 is inside the first but not the second.
    idx = make_index([(1, T0)], [[(30.0, None), (5.0, None)]], [["X"]], starts=[[(0.0, None), (-1.0, "A")]])
    meta, d = _dense24(idx, ev, k=2)
    assert meta["condition_answers"].to_list() == [[True]]
    idx = make_index([(1, T0)], [[(5.0, None), (30.0, None)]], [["X"]], starts=[[(-1.0, "A"), (0.0, None)]])
    meta, d = _dense24(idx, ev, k=2)
    assert meta["condition_answers"].to_list() == [[False]]
    assert bool(d[0, 0, VOCAB24.code_to_index()["Y"]]) is True


def test_identical_resolved_starts_for_two_contexts_of_one_subject() -> None:
    ev = _events([(1, _day(5), "A"), (1, _day(7), "X"), (1, _day(9), "B"), (1, _day(11), "Y")])
    idx = make_index(
        [(1, T0), (1, _day(1))],
        [[(-1.0, "B"), (10.0, None)], [(10.0, None), (-1.0, "B")]],
        [["X"], ["Y"]],
        starts=[[(-1.0, "A")] * 2, [(-1.0, "A")] * 2],
    )
    meta, packed, _ = label_multitask_index(idx, ev, VOCAB24, 2, chunk_rows=1)
    d = np.unpackbits(packed, axis=-1, count=VOCAB24.size, bitorder="little").astype(bool)
    x, y = VOCAB24.code_to_index()["X"], VOCAB24.code_to_index()["Y"]
    # Both contexts open at day 5; (5, 9) holds X only, (5, 15) holds X and Y.
    assert d[0, 0, [x, y]].tolist() == [True, False] and d[0, 1, [x, y]].tolist() == [True, True]
    assert d[1, 0, [x, y]].tolist() == [True, True] and d[1, 1, [x, y]].tolist() == [True, False]
    assert meta["condition_answers"].to_list() == [[True], [True]]


def test_resolved_start_order_differs_from_prediction_time_order() -> None:
    ev = _events([(1, _day(2), "X"), (1, _day(15), "Y"), (1, _day(25), "Z")])
    # ctx 0 (pt day 0) opens at day 20; ctx 1 (pt day 1) opens at day 1: start order is reversed.
    idx = make_index(
        [(1, T0), (1, _day(1))], [[(10.0, None)], [(10.0, None)]], starts=[[(20.0, None)], [(0.0, None)]]
    )
    meta, packed, _ = label_multitask_index(idx, ev, VOCAB24, 1, chunk_rows=1)
    d = np.unpackbits(packed, axis=-1, count=VOCAB24.size, bitorder="little").astype(bool)
    assert meta["prediction_time"].to_list() == [T0, _day(1)]  # metadata stays in prediction-time order
    assert _labels(d[:1]) == {"A": False, "B": False, "X": False, "Y": False, "Z": True, "DISCHARGE": False}
    assert _labels(d[1:]) == {"A": False, "B": False, "X": True, "Y": False, "Z": False, "DISCHARGE": False}


def test_validate_index_start_pair_rules() -> None:
    two = [[(5.0, None), (6.0, None)]]
    with pytest.raises(ValueError, match="each start slot"):
        validate_index(make_index([(1, T0)], two, starts=[[(5.0, "A"), (0.0, None)]]), 2)
    with pytest.raises(ValueError, match="each start slot"):
        validate_index(make_index([(1, T0)], two, starts=[[(-1.0, None), (0.0, None)]]), 2)
    with pytest.raises(ValueError, match="exactly 2 start_durations"):
        validate_index(make_index([(1, T0)], two, starts=[[(0.0, None)]]), 2)
    with pytest.raises(ValueError, match="not all of"):
        validate_index(make_index([(1, T0)], two, starts=[[(0.0, None)] * 2]).drop("start_events"), 2)
    validate_index(make_index([(1, T0)], two), 2)  # no start columns: legacy prediction-time starts
    validate_index(make_index([(1, T0)], two, starts=[[(0.0, None), (-1.0, "A")]]), 2)


def test_unknown_and_pad_start_codes_are_hard_errors() -> None:
    ev = _events([(1, _day(1), "A")])
    with pytest.raises(ValueError, match="start_event code\\(s\\) are not in the base vocabulary"):
        label_multitask_index(
            make_index([(1, T0)], [[(5.0, None)]], starts=[[(-1.0, "NOPE")]]), ev, VOCAB24, 1
        )
    vocab = TargetVocabulary.from_pairs(["PAD", "A"], [0, 1])
    with pytest.raises(ValueError, match="start_event code\\(s\\) at vocab index 0"):
        label_multitask_index(make_index([(1, T0)], [[(5.0, None)]], starts=[[(-1.0, "PAD")]]), ev, vocab, 1)


def _random_window_index(rng: np.random.Generator, n: int, k: int, subjects: list[int], horizon: int):
    obs = CODES[:11]
    contexts = [
        (int(rng.integers(subjects[0], subjects[-1] + 2)), T0 + timedelta(days=int(rng.integers(0, horizon))))
        for _ in range(n)
    ]

    def pick_start():
        u = rng.random()
        if u < 0.3:
            return -1.0, obs[int(rng.integers(0, len(obs)))]
        if u < 0.6:
            return 0.0, None
        return float(rng.uniform(0.1, 30)), None

    def pick_end():
        if rng.random() < 0.5:
            return float(rng.uniform(0.1, 40)), None
        pool = CODES if rng.random() < 0.2 else obs
        return -1.0, pool[int(rng.integers(0, len(pool)))]

    starts = [[pick_start() for _ in range(k)] for _ in range(n)]
    bounds = [[pick_end() for _ in range(k)] for _ in range(n)]
    conds = [[CODES[int(rng.integers(0, len(CODES)))] for _ in range(k - 1)] for _ in range(n)]
    return make_index(contexts, bounds, conds, starts=starts)


def _naive_from_frames(index_df: pl.DataFrame, events_df: pl.DataFrame, vocab: TargetVocabulary, k: int):
    from every_query.generate_tasks.interval_table import naive_window_labels

    c2i = vocab.code_to_index()
    ev = events_df.filter(pl.col("time").is_not_null()).with_columns(pl.col("time").cast(pl.Int64))
    events = [(int(s), int(t), c2i[c]) for s, t, c in ev.select("subject_id", "time", "code").iter_rows()]
    pts = index_df["prediction_time"].cast(pl.Int64).to_list()
    contexts = list(zip(index_df["subject_id"].to_list(), pts, strict=True))
    windows = []
    for r in index_df.iter_rows(named=True):
        row = []
        for j in range(k):
            sd, se = r["start_durations"][j], r["start_events"][j]
            ed, be = r["durations"][j], r["bound_events"][j]
            row.append(
                ((sd, None) if se is None else (None, c2i[se]), (ed, None) if be is None else (None, c2i[be]))
            )
        windows.append(row)
    return naive_window_labels(events, contexts, windows, vocab.size, k)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 1000])
def test_fuzz_window_starts_against_naive_and_scalar_oracles(seed: int, chunk_rows: int) -> None:
    from every_query.generate_tasks.interval_table import INF
    from tests.multitask.conftest import resolved_start_scalar_oracle

    rng = np.random.default_rng(seed)
    k = 4
    events = make_events(rng, [1, 2, 3], n_times=(5, 25), horizon_days=60, codes=CODES)
    vocab = TargetVocabulary.from_pairs(CODES, list(range(1, len(CODES) + 1)))
    idx = _random_window_index(rng, 20, k, [1, 3], 60)
    meta, packed, stats = label_multitask_index(idx, events, vocab, k, chunk_rows=chunk_rows)
    dense = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)

    st, _, expect = _naive_from_frames(meta, events, vocab, k)
    assert np.array_equal(dense, expect)
    assert not dense[:, :, 0].any()
    # Scalar differential: label_with_event_bounds fed the resolved start as its prediction time.
    scalar = resolved_start_scalar_oracle(meta, events, list(vocab.codes), k, st)
    assert np.array_equal(dense[:, :, vocab.indices], scalar)
    assert (dense[st == INF] == 0).all()
    # Condition answers are read per flattened row off the packed chunk.
    answers = np.array(meta["condition_answers"].to_list(), dtype=bool)
    assert np.array_equal(answers, condition_answers_oracle(meta, dense, vocab))
    assert stats.n_event_starts == int((meta["start_events"].explode().is_not_null()).sum())
    assert 0.0 <= stats.frac_empty_windows <= 1.0 and stats.mean_positives_per_window >= 0


def test_chunked_equals_unchunked_and_supplied_order_invariance_with_starts() -> None:
    rng = np.random.default_rng(21)
    events = make_events(rng, [1, 2, 3], codes=CODES)
    vocab = TargetVocabulary.from_pairs(CODES, list(range(1, len(CODES) + 1)))
    idx = _random_window_index(rng, 30, 3, [1, 3], 400)
    meta_a, packed_a, _ = label_multitask_index(idx, events, vocab, 3, chunk_rows=1)
    perm = rng.permutation(idx.height)
    shuffled = idx.with_row_index("_ctx_id").with_columns(pl.col("_ctx_id").cast(pl.Int64))[perm.tolist()]
    meta_b, packed_b, _ = label_multitask_index(shuffled, events, vocab, 3, chunk_rows=1000)
    assert meta_a.equals(meta_b)
    assert np.array_equal(packed_a, packed_b)
    assert meta_a.sort("subject_id", "prediction_time").equals(meta_a)
    assert list(meta_a.columns) == [
        "subject_id",
        "prediction_time",
        "start_durations",
        "start_events",
        "durations",
        "bound_events",
        "condition_codes",
        "condition_answers",
    ]


def test_many_windows_do_not_wrap_the_window_position_index() -> None:
    """Regression: a uint8 window position wrapped at ``num_bounds >= 256``, scattering a window's
    packed bits onto slot ``k % 256`` of the same context."""
    k = 260
    ev = _events([(1, _day(d), "X") for d in range(1, 40)])
    # Window k opens at day k/10 and closes 0.05 days later, so exactly the windows whose open/close
    # straddle an integer day hold X.  Slots 0..3 (which a uint8 cast would collide with 256..259)
    # deliberately differ from those late slots.
    starts = [[(round(0.1 * j, 4), None) for j in range(k)]]
    bounds = [[(0.95, None) for _ in range(k)]]
    idx = make_index([(1, T0)], bounds, [["X"] * (k - 1)], starts=starts)
    _, packed, _ = label_multitask_index(idx, ev, VOCAB24, k, chunk_rows=3)
    d = np.unpackbits(packed, axis=-1, count=VOCAB24.size, bitorder="little").astype(bool)
    x = VOCAB24.code_to_index()["X"]
    expected = [any(0.1 * j < day < 0.1 * j + 0.95 for day in range(1, 40)) for j in range(k)]
    assert d[0, :, x].tolist() == expected


# --- ancestors as window furniture (PRs C + D) -----------------------------------------------------
#
# Targets stay leaf-only in every mode; what an ontology changes here is the other half of a window.
# The golden cohort (``tests/ontology_suite``) is used throughout, so every expectation comes from the
# independently written oracle rather than from a second run of the code under test.

#: Pure ancestor nodes of the golden DAG, each with at least one descendant event in the fixture.
_GOLDEN_ANCESTOR_BOUNDS = ["READMISSION", "DX", "LAB", "TIMELINE"]
_GOLDEN_SUBJECTS = [1, 2, 3, 4, 5]


def test_ancestor_bound_event_matches_the_scalar_oracle(tmp_path: Path) -> None:
    """An ancestor-valued ``bound_event`` closes the window at the first occurrence of *any* descendant.

    Every leaf target of every golden subject is compared against ``oracle.label_event_bounded``, which
    resolves the same boundary by walking the DAG.  The failure this guards is silent: an ancestor id
    that never reaches the interval table resolves to ``INF``, which merely leaves the window open, so
    the run is also pinned to be genuinely bounded rather than degenerating to "ever again".
    """
    onto, _, vocab = _golden_vocab(tmp_path, "boundaries")
    assert set(_GOLDEN_ANCESTOR_BOUNDS) <= set(_ancestor_names(onto)), (
        "the fixture boundaries must be pure ancestor nodes, not leaves"
    )
    events = events_to_frame(EVENTS)
    k = len(_GOLDEN_ANCESTOR_BOUNDS)
    idx = make_index(
        [(s, GOLDEN_T0) for s in _GOLDEN_SUBJECTS],
        [[(-1.0, b) for b in _GOLDEN_ANCESTOR_BOUNDS]] * len(_GOLDEN_SUBJECTS),
        fill_condition="LAB//GLU",
    )
    times: dict[str, np.ndarray] = {}
    meta, packed, _ = label_multitask_index(
        idx, events, vocab, k, ontology_dir=str(onto), window_times_out=times
    )
    dense = _golden_dense(packed, vocab)
    c2i = vocab.code_to_index()
    for row, subject in enumerate(meta["subject_id"].to_list()):
        for j, boundary in enumerate(_GOLDEN_ANCESTOR_BOUNDS):
            for leaf in LEAVES:
                got = bool(dense[row, j, c2i[leaf]])
                want = label_event_bounded(EVENTS, subject, ONTOLOGY, GOLDEN_T0, leaf, boundary)
                assert got == want, f"subject {subject}, boundary {boundary!r}, leaf {leaf!r}"

    # The ancestor boundary really bounded something: an id that never reached the table would have
    # resolved to INF everywhere, leaving every window open to the end of the record.
    assert dense.any(), "every label false would pass the comparison only if the oracle agreed by luck"
    assert (times["end_times"] != INF).any(), "at least one ancestor boundary must actually occur"
    unbounded = make_index(
        [(s, GOLDEN_T0) for s in _GOLDEN_SUBJECTS],
        [[(3650.0, None)] * k] * len(_GOLDEN_SUBJECTS),
        fill_condition="LAB//GLU",
    )
    _, packed_open, _ = label_multitask_index(unbounded, events, vocab, k, ontology_dir=str(onto))
    assert not np.array_equal(packed, packed_open), (
        "an ancestor boundary resolving to INF would reproduce the unbounded ten-year window exactly"
    )


def test_ancestor_start_event_opens_the_window_at_the_first_descendant(tmp_path: Path) -> None:
    """An ancestor-valued ``start_event`` opens the window at the first descendant occurrence - the same
    window the responsible leaf would have opened.

    Subject 5's only READMISSION descendant is ``READMISSION//CHILD_B`` on 03-05, so the ancestor start
    and that leaf start must resolve to the identical instant.  ``next_occurrence_after`` returns ``INF``
    without error for a code wider than the table's ``code_bits``, and an unresolved start yields an
    all-false window, so the resolved times themselves are asserted, not just the bits.
    """
    onto, _, vocab = _golden_vocab(tmp_path, "boundaries")
    events = events_to_frame(EVENTS)
    ancestor_times: dict[str, np.ndarray] = {}
    leaf_times: dict[str, np.ndarray] = {}
    _, packed_a, _ = label_multitask_index(
        make_index([(5, GOLDEN_T0)], [[(10.0, None)]], starts=[[(-1.0, "READMISSION")]]),
        events,
        vocab,
        1,
        ontology_dir=str(onto),
        window_times_out=ancestor_times,
    )
    _, packed_l, _ = label_multitask_index(
        make_index([(5, GOLDEN_T0)], [[(10.0, None)]], starts=[[(-1.0, "READMISSION//CHILD_B")]]),
        events,
        vocab,
        1,
        ontology_dir=str(onto),
        window_times_out=leaf_times,
    )
    opened = int(ancestor_times["start_times"][0, 0])
    assert opened != INF, "the ancestor start must resolve; INF is next_occurrence_after's silent failure"
    expected_us = int((datetime(2024, 3, 5) - datetime(1970, 1, 1)).total_seconds() * 1_000_000)
    assert opened == expected_us, "the window opens at CHILD_B on 03-05, the first descendant"
    assert opened == int(leaf_times["start_times"][0, 0])
    assert np.array_equal(packed_a, packed_l), "ancestor start == its responsible leaf start, bit for bit"

    labels = {c: bool(_golden_dense(packed_a, vocab)[0, 0, i]) for c, i in vocab.code_to_index().items()}
    assert labels["DISCHARGE"] is True, "03-09 is strictly inside (03-05, 03-15)"
    assert labels["ADMISSION"] is False, "03-02 is before the resolved start"
    assert labels["READMISSION//CHILD_B"] is False, "the start instant itself is excluded"
    assert labels["TIMELINE//END"] is False, "03-28 is past the 10-day horizon"


_COND_WINDOWS = [(7.0, None), (30.0, None), (-1.0, "DISCHARGE")]


def test_ancestor_condition_answer_is_the_or_over_its_descendants(tmp_path: Path) -> None:
    """An ancestor conditioning code's answer is the OR over its closure descendants (PR D).

    The ancestor has no packed column, so the pre-PR-D path - reading bit ``index >> 3`` off the packed
    row - would land on a padding bit that is always 0.  The all-False result that would produce is
    refused explicitly, and a leaf conditioning code is checked alongside to show the ordinary
    packed-bit path is untouched.
    """
    onto, _, vocab = _golden_vocab(tmp_path, "conditions")
    events = events_to_frame(EVENTS)
    assert "READMISSION" in _ancestor_names(onto), "the fixture conditioning code must be an ancestor"
    k = len(_COND_WINDOWS)
    contexts = [(s, GOLDEN_T0) for s in _GOLDEN_SUBJECTS]
    windows = [_COND_WINDOWS] * len(_GOLDEN_SUBJECTS)

    meta, packed, stats = label_multitask_index(
        make_index(contexts, windows, fill_condition="READMISSION"),
        events,
        vocab,
        k,
        ontology_dir=str(onto),
    )
    dense = _golden_dense(packed, vocab)
    answers = np.array(meta["condition_answers"].to_list(), dtype=bool)
    assert answers.shape == (len(_GOLDEN_SUBJECTS), k - 1)
    assert np.array_equal(answers, condition_answers_oracle(meta, dense, vocab, ontology_dir=onto))
    assert answers.any() and not answers.all(), (
        "a padding-bit read would make every ancestor answer False; a constant-True bug would make "
        "them all True"
    )
    assert stats.n_ancestor_condition_slots == len(_GOLDEN_SUBJECTS) * (k - 1)

    leaf_meta, leaf_packed, leaf_stats = label_multitask_index(
        make_index(contexts, windows, fill_condition="LAB//GLU"),
        events,
        vocab,
        k,
        ontology_dir=str(onto),
    )
    leaf_answers = np.array(leaf_meta["condition_answers"].to_list(), dtype=bool)
    leaf_dense = _golden_dense(leaf_packed, vocab)
    assert np.array_equal(leaf_answers, condition_answers_oracle(leaf_meta, leaf_dense, vocab))
    assert leaf_stats.n_ancestor_condition_slots == 0
    assert np.array_equal(leaf_packed, packed), "the conditioning code never moves a target bit"


def test_leaf_bits_are_identical_with_and_without_the_ontology(tmp_path: Path) -> None:
    """The byte-identity invariant the whole design rests on: with leaf-only starts, bounds and
    conditioning codes, attaching an ontology changes nothing about the packed leaf targets.

    The closure pairs every leaf with itself, so the ``code_index < V`` rows of the expanded stream are
    exactly the unexpanded stream - and the leaf interval table built from them is the only one that
    ever sets a bit.  If the expansion leaked into labeling, an ancestor's occurrences would start
    setting descendant bits and this array comparison would fail.
    """
    onto, leaf_vocab, vocab = _golden_vocab(tmp_path, "boundaries+conditions")
    events = events_to_frame(EVENTS)
    windows = [(7.0, None), (40.0, None), (-1.0, "DISCHARGE"), (-1.0, "ADMISSION")]
    starts = [(0.0, None), (2.0, None), (-1.0, "ADMISSION"), (0.0, None)]
    conditions = ["LAB//GLU", "DX//CARDIO//MI", "TIMELINE//END"]
    n = len(_GOLDEN_SUBJECTS)
    idx = make_index(
        [(s, GOLDEN_T0) for s in _GOLDEN_SUBJECTS],
        [windows] * n,
        [conditions] * n,
        starts=[starts] * n,
    )
    meta_leaf, packed_leaf, stats_leaf = label_multitask_index(idx, events, leaf_vocab, len(windows))
    meta_onto, packed_onto, stats_onto = label_multitask_index(
        idx, events, vocab, len(windows), ontology_dir=str(onto)
    )
    assert np.array_equal(packed_leaf, packed_onto), "the ontology must not touch a single leaf bit"
    assert meta_leaf.equals(meta_onto), "metadata, conditioning answers included, is unchanged"
    assert packed_leaf.any(), "an all-zero comparison would be vacuous"
    assert stats_onto.n_query_node_events > stats_leaf.n_events, "the stream really was expanded"
    assert stats_onto.n_events == stats_leaf.n_events, "but the leaf table saw exactly the same events"
