"""Eval-path ontology plumbing: the knob the eval config documents must actually do something.

``sample_evaluation_query_sequences`` shipped ``ontology_dir`` (and, at the time, an ancestor-share
knob) in its config — comment block, rationale and all — while ``grep -n 'ontology' `` over the
module returned nothing.  Hydra accepted the overrides and discarded them, which made every
ancestor claim unmeasurable in two independent ways:

- the sampled query **universe** never contained an ancestor node, so an "ancestor" eval grid
  scored leaf skill under an ancestor headline, and
- the event stream was never exploded through the **closure**, so a *designed* ancestor query was
  labeled ``False`` at every context — an ancestor's own name occurs in no event stream, only its
  descendants' do — producing a well-formed ``QuerySeqSchema`` parquet full of wrong answers.

Wiring the two knobs up opened a third way to be silently wrong, which sections 5 and 5b cover:
the output path encodes only the shard, never the ontology, so a leaf-only grid and an
ancestor grid land on the **same output path**.  ``overwrite: false`` then skips the second run and
leaves the first one's labels in place — the correct-looking file is simply the wrong file.  The
provenance sidecar exists to make "already exists" mean "already correct", and it has to be keyed
on *both* ontology-derived inputs (the closure **and** the query universe, with its slot order and
multiplicities) and on the output's *full* path below ``out_dir``, or some pair of runs it cannot
tell apart reuses the other's answers.

None of these failures raises, and none changes a single shape or dtype.  Every test here
therefore asserts on **label values, universe membership and what is on disk**; a shape assertion
is exactly what let this ship green.  Where a test's subject is a skip decision, it is written so
that the *wrong* decision leaves an observably wrong grid behind, not merely a missing rewrite.

The fixtures are a seven-code synthetic vocabulary, the ontology it induces (plus a second one
built from a vocabulary short one code, so the closure can be varied on its own) and a two-patient
event stream, all built in this file.  Nothing here reads real data.
"""

import inspect
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

from every_query.data.ontology import (
    EMBEDDING_MIX_FILE,
    EVENT_TO_QUERY_NODES_FILE,
    ONTOLOGY_VOCAB_FILE,
    build_event_to_query_nodes,
    build_ontology,
)
from every_query.generate_tasks import query_sequence_labeling as train_seq
from every_query.generate_tasks import sample_evaluation_query_sequences as eval_seq
from every_query.generate_tasks import sample_multitask_sequences as sms
from every_query.generate_tasks.sample_tasks import LABELED_DIRNAME
from tests.designed_specs import entry


@pytest.fixture(autouse=True)
def _setup_doctest_namespace():
    """Override the repo-root autouse fixture: nothing here needs the HuggingFace demo model.

    Same no-op override ``tests/sampler/conftest.py`` uses, for the same reason — this file is a
    pure sampling/labeling test and must stay offline.
    """
    yield


# ── the synthetic world ─────────────────────────────────────────────────

SPLIT = "held_out"
PT = datetime(2020, 6, 1)
HORIZON = 30.0

#: The cohort vocabulary.  ``//`` is the MEDS hierarchy separator, so these codes also induce the
#: interior nodes ``ICD``, ``MED``, ``MED//STATIN`` and ``TIMELINE``.
#:
#: ``ICD//CIRC`` and ``ICD//RESP`` are deliberately **dual-role**: real codes AND the parent
#: prefixes of other codes — the ordinary situation in a MEDS ``codes.parquet`` (399 of MIMIC-IV's
#: 13,908 codes look like this) and the one that makes the labeling defect silent rather than
#: loud.
#:
#: Because one string cannot mean both "exactly this code" and "this code or any descendant",
#: ``build_ontology`` mints a separate subtree node beside each of them.  The leaf keeps its exact
#: meaning; the ``//ANY`` node is what rolls descendants up.  So the ancestor under test here is
#: ``ICD//CIRC//ANY``, not ``ICD//CIRC``.
#:
#: Note ``ICD//CIRC//ANY`` is in no ``codes.parquet``, so a designed spec naming it can only
#: validate if addressability is resolved against the ontology rather than the sampling universe
#: — which is exactly what ``model_query_vocab`` restores.
LEAVES = [
    "ICD//CIRC",
    "ICD//CIRC//I21",
    "ICD//CIRC//I50",
    "ICD//RESP",
    "ICD//RESP//J44",
    "MED//STATIN//ATORVA",
    "TIMELINE//END",
]
ANCESTOR = "ICD//CIRC//ANY"
CONTROL_ANCESTOR = "ICD//RESP//ANY"
#: A node that is *only* a prefix, never a code: addressable as a query solely through the
#: ontology.  ``MED//STATIN`` occurs (via ``ATORVA``) for subject 1 and never for 2.
ANCESTOR_ONLY_NODE = "MED//STATIN"
#: What ``build_query_universe`` may draw from: every non-leaf node except the ``TIMELINE``
#: namespace, which it drops as tautological.
USABLE_ANCESTORS = ["ICD", "ICD//CIRC//ANY", "ICD//RESP//ANY", "MED", "MED//STATIN"]

#: Subject 1 has an ``ICD//CIRC`` descendant inside the horizon and no ``ICD//RESP`` event at all;
#: subject 2 is the mirror image, and its one ``ICD//CIRC`` descendant sits *before* the
#: prediction time.  So each ancestor is True for exactly one subject: an implementation that
#: answers everything True, everything False, or ignores the window fails on some cell.
EVENTS = [
    (1, datetime(2020, 1, 10), "ICD//CIRC//I21"),  # 143d before PT
    (1, datetime(2020, 6, 6), "ICD//CIRC//I50"),  # PT + 5d  -> ICD//CIRC True
    (1, datetime(2020, 6, 10), "MED//STATIN//ATORVA"),
    (1, datetime(2020, 9, 1), "TIMELINE//END"),
    (2, datetime(2020, 2, 1), "ICD//CIRC//I21"),  # 121d before PT -> ICD//CIRC False
    (2, datetime(2020, 6, 4), "ICD//RESP//J44"),  # PT + 3d  -> ICD//RESP True
    (2, datetime(2020, 9, 1), "TIMELINE//END"),
]

#: The answers a correct labeler must produce for ``(subject, ancestor)`` at ``PT`` over 30 days.
#: Derived by hand from ``EVENTS`` above, not from either code path.
TRUTH = {
    (1, ANCESTOR): True,
    (1, CONTROL_ANCESTOR): False,
    (2, ANCESTOR): False,
    (2, CONTROL_ANCESTOR): True,
}


def _write_ontology(out: Path, codes: list[str]) -> Path:
    """Write the three ``EQ_build_ontology`` artifacts for ``codes`` into ``out`` (created if absent).

    Overwrites in place when ``out`` already holds an ontology, which is how the real thing behaves
    when ``EQ_build_ontology`` is re-run after the cohort's ``codes.parquet`` changed — the case
    :func:`test_rebuilding_the_ontology_in_place_relabels_the_grid` needs and no fixture provides,
    since a guard keyed on the ontology *path* rather than its contents survives every test that
    switches between two differently-named directories.
    """
    out.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({"code": codes, "code/vocab_index": list(range(1, len(codes) + 1))})
    nodes, mix = build_ontology(frame)
    nodes.write_parquet(out / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(out / EMBEDDING_MIX_FILE)
    build_event_to_query_nodes(nodes, mix).write_parquet(out / EVENT_TO_QUERY_NODES_FILE)
    return out


#: ``LEAVES`` minus the one code that links subject 1's in-window event to ``ICD//CIRC``.
STALE_LEAVES = [c for c in LEAVES if c != "ICD//CIRC//I50"]


@pytest.fixture
def ontology_dir(tmp_path: Path) -> Path:
    """A three-artifact ontology directory built from ``LEAVES``, as ``EQ_build_ontology`` writes it."""
    return _write_ontology(tmp_path / "ontology", LEAVES)


@pytest.fixture
def stale_ontology_dir(tmp_path: Path) -> Path:
    """A *second* ontology, built from a vocabulary missing ``ICD//CIRC//I50``.

    Its node set — and therefore the query universe it induces — is identical to
    :func:`ontology_dir`'s; only its **closure** differs, by the one row that links subject 1's
    in-window event to ``ICD//CIRC``.  That isolates the staleness question to the closure alone:
    a guard that merely hashed the universe would see two identical runs.
    """
    return _write_ontology(tmp_path / "stale_ontology", STALE_LEAVES)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """MEDS root whose ``held_out`` split holds the two subjects on *separate* shards."""
    split_dir = tmp_path / "meds" / "data" / SPLIT
    split_dir.mkdir(parents=True)
    events = pl.DataFrame(
        {
            "subject_id": [e[0] for e in EVENTS],
            "time": [e[1] for e in EVENTS],
            "code": [e[2] for e in EVENTS],
        }
    ).with_columns(pl.col("time").cast(pl.Datetime("us")))
    for shard, subject in (("0", 1), ("1", 2)):
        events.filter(pl.col("subject_id") == subject).write_parquet(split_dir / f"{shard}.parquet")
    return tmp_path / "meds"


@pytest.fixture
def cohort(tmp_path: Path) -> Path:
    """The supplied ``(subject_id, prediction_time)`` cohort: both subjects, one context each."""
    fp = tmp_path / "cohort.parquet"
    pl.DataFrame({"subject_id": [1, 2], "prediction_time": [PT, PT]}).with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us"))
    ).write_parquet(fp)
    return fp


@pytest.fixture
def codes_yaml(tmp_path: Path) -> Path:
    """``query_codes`` as a YAML list — the leaf vocabulary, with no ancestor in it."""
    fp = tmp_path / "codes.yaml"
    fp.write_text(yaml.safe_dump(LEAVES))
    return fp


def _specs_yaml(tmp_path: Path, **name_to_query: str) -> Path:
    """Write a designed-spec YAML of one-query sequences: ``{name: [entry(code, HORIZON)]}``."""
    fp = tmp_path / "specs.yaml"
    fp.write_text(yaml.safe_dump({n: [entry(q, HORIZON)] for n, q in name_to_query.items()}))
    return fp


# ── driving the two Hydra entry points ──────────────────────────────────


def _run_eval(**overrides) -> None:
    """Compose the *real* eval config and run ``main`` on it, the way the CLI does.

    Going through Hydra rather than calling ``run_worker`` directly is the point: defect #3 was
    that Hydra accepted ``ontology_dir=`` and ``main`` dropped it on the floor, which no call made
    straight to ``run_worker`` would ever notice.
    """
    # The production default (40 positives) would drop every task in these tiny fixtures.
    overrides = {"min_task_positives": "null", **overrides}
    with initialize_config_dir(config_dir=eval_seq.CONFIGS, version_base=None):
        cfg = compose(
            config_name="sample_evaluation_query_sequences_config",
            overrides=[f"{k}={v}" for k, v in overrides.items()],
        )
    eval_seq.main.__wrapped__(cfg)


def _run_train(**overrides) -> None:
    """Same, for the training sampler (which only samples: its contexts come from the split)."""
    with initialize_config_dir(config_dir=train_seq.CONFIGS, version_base=None):
        cfg = compose(
            config_name="sample_query_sequences_config",
            overrides=[f"{k}={v}" for k, v in overrides.items()],
        )
    train_seq.main.__wrapped__(cfg)


def _grid_fps(out_dir: Path) -> list[Path]:
    """The per-shard output files of one run: both subjects live on separate shards."""
    return [out_dir / "eval" / SPLIT / f"{shard}.parquet" for shard in ("0", "1")]


def _grid(src: Path) -> pl.DataFrame:
    """Read a whole grid — every shard under ``{out_dir}/eval/{SPLIT}`` — or one shard file."""
    if src.is_dir():
        return pl.concat([pl.read_parquet(fp) for fp in _grid_fps(src)])
    return pl.read_parquet(src)


def _answers_by_subject_and_query(src: Path) -> dict[tuple[int, str], bool]:
    """Flatten a ``QuerySeqSchema`` grid of one-query sequences to ``(subject, query) -> answer``."""
    df = _grid(src)
    out: dict[tuple[int, str], bool] = {}
    for row in df.iter_rows(named=True):
        for q, a in zip(row["queries"], row["answers"], strict=True):
            out[(int(row["subject_id"]), q)] = bool(a)
    return out


def _answers_by_context_and_query(df: pl.DataFrame) -> dict[tuple[int, datetime, str], bool]:
    """Flatten ``QuerySeqSchema`` rows to ``(subject, prediction_time, query) -> answer``."""
    out: dict[tuple[int, datetime, str], bool] = {}
    for row in df.iter_rows(named=True):
        for q, a in zip(row["queries"], row["answers"], strict=True):
            out[(int(row["subject_id"]), row["prediction_time"], q)] = bool(a)
    return out


def _all_queries(src: Path) -> list[str]:
    """Every query string emitted in a grid, in row-major order."""
    return [q for row in _grid(src).iter_rows(named=True) for q in row["queries"]]


# ── 1. defect #3: the universe ──────────────────────────────────────────


def test_an_ontology_puts_every_ancestor_node_in_the_sampled_query_universe(
    tmp_path: Path, data_dir: Path, cohort: Path, codes_yaml: Path, ontology_dir: Path
):
    """``ontology_dir`` must extend the pool the eval grid draws its queries from.

    Red before the fix: ``main`` never called ``build_query_universe``, so the universe was the
    five leaf codes and no ancestor node could ever be drawn — an "ancestor" eval grid that
    measured leaf skill, with nothing in the output to say so.

    The share is asserted, not merely the presence of one ancestor: the contract is that every
    node — leaf or ancestor — is one equally likely slot of the pool, which a mirror that appended
    a single ancestor node would also satisfy at "presence" strength.
    """
    out_dir = tmp_path / "grid"
    _run_eval(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=codes_yaml,
        contexts_path=cohort,
        split=SPLIT,
        num_evaluation_sequences=96,
        ontology_dir=ontology_dir,
    )

    queries = _all_queries(out_dir)
    drawn_ancestors = [q for q in queries if q in USABLE_ANCESTORS]

    assert drawn_ancestors, (
        "an ontology was configured but no ancestor node was drawn; the eval query universe is "
        f"still leaf-only. Queries drawn: {sorted(set(queries))}"
    )
    # 7 leaves + 5 usable ancestors, each one slot: an ancestor share of 5/12, drawn
    # 32 sequences x 3 positions = 96 times.
    share = len(drawn_ancestors) / len(queries)
    assert 0.25 <= share <= 0.6, f"ancestor share {share:.3f} is nowhere near 5/12"
    assert len(set(drawn_ancestors)) >= 3, (
        f"only {sorted(set(drawn_ancestors))} were drawable; the whole ancestor set should be in play"
    )
    # The TIMELINE namespace is tautological as a query and must stay out, exactly as in training.
    assert "TIMELINE" not in queries


def test_no_ontology_keeps_the_universe_leaf_only(
    tmp_path: Path, data_dir: Path, cohort: Path, codes_yaml: Path
):
    """Without an ontology nothing but the cohort's own codes can be drawn."""
    out_dir = tmp_path / "grid"
    _run_eval(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=codes_yaml,
        contexts_path=cohort,
        split=SPLIT,
        num_evaluation_sequences=96,
    )

    queries = set(_all_queries(out_dir))
    assert queries <= set(LEAVES), (
        f"non-leaf codes {sorted(queries - set(LEAVES))} were drawn without an ontology"
    )


# ── 2. defect #4: the label ─────────────────────────────────────────────


def test_designed_ancestor_query_is_true_when_only_a_descendant_occurred(
    tmp_path: Path, data_dir: Path, cohort: Path, codes_yaml: Path, ontology_dir: Path
):
    """``ICD//CIRC`` must label ``True`` for a patient whose stream only holds ``ICD//CIRC//I50``.

    No event in either subject's stream carries the string ``ICD//CIRC``; the ancestor is only
    ever *implied* by its descendants.  Without the closure explosion the asof join finds nothing
    and every answer comes back ``False`` — a full, schema-valid parquet of wrong labels, which is
    why this asserts the four booleans themselves rather than a row count.

    The two subjects sit on separate shards, so both per-shard workers have to explode.

    Both queried codes are in ``query_codes`` already, so nothing rejects the spec and nothing
    warns.  The pre-fix module ran this exact grid to completion and wrote ``False`` in all four
    cells.
    """
    out_dir = tmp_path / "grid"
    specs = _specs_yaml(tmp_path, circ_30d=ANCESTOR, resp_30d=CONTROL_ANCESTOR)
    _run_eval(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=codes_yaml,
        contexts_path=cohort,
        sequences_path=specs,
        split=SPLIT,
        ontology_dir=ontology_dir,
    )

    for fp in _grid_fps(out_dir):
        assert fp.is_file(), f"expected output at {fp}"
    answers = _answers_by_subject_and_query(out_dir)

    assert answers == TRUTH, (
        "ancestor queries are mislabeled: the event stream was not exploded through the closure. "
        f"got {answers}, expected {TRUTH}"
    )


def test_a_node_that_is_only_a_prefix_is_both_addressable_and_correctly_labeled(
    tmp_path: Path, data_dir: Path, cohort: Path, codes_yaml: Path, ontology_dir: Path
):
    """The other ancestor shape: a node that exists *only* as a prefix, never as a code.

    ``MED//STATIN`` is in no ``codes.parquet`` and in no event stream; it reaches the eval grid
    only if the ontology put it in the universe (or ``validate_spec_codes`` rejects the spec
    outright), and it labels correctly only if the closure explosion ran.  So this needs both
    halves of the plumbing at once, and pins the label rather than the mere absence of an error.
    """
    out_dir = tmp_path / "grid"
    specs = _specs_yaml(tmp_path, statin_30d=ANCESTOR_ONLY_NODE)
    _run_eval(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=codes_yaml,
        contexts_path=cohort,
        sequences_path=specs,
        split=SPLIT,
        ontology_dir=ontology_dir,
    )

    # Subject 1 takes atorvastatin 9 days after the prediction time; subject 2 never does.
    assert _answers_by_subject_and_query(out_dir) == {
        (1, ANCESTOR_ONLY_NODE): True,
        (2, ANCESTOR_ONLY_NODE): False,
    }


def test_leaf_labels_are_untouched_by_the_explosion(
    tmp_path: Path, data_dir: Path, cohort: Path, codes_yaml: Path, ontology_dir: Path
):
    """Turning the ontology on must not change what a *leaf* query answers.

    The explosion repeats every event under its ancestors' names; if it dropped, duplicated or renamed the
    original leaf rows instead of adding to them, leaf answers would move and every leaf metric in an ontology
    run would shift for a reason nobody would look for.
    """
    specs = _specs_yaml(tmp_path, i50="ICD//CIRC//I50", j44="ICD//RESP//J44")
    got = {}
    for tag, ont in (("off", None), ("on", ontology_dir)):
        out_dir = tmp_path / f"grid_{tag}"
        kwargs = {
            "data_dir": data_dir,
            "out_dir": out_dir,
            "query_codes": codes_yaml,
            "contexts_path": cohort,
            "sequences_path": specs,
            "split": SPLIT,
        }
        if ont is not None:
            kwargs["ontology_dir"] = ont
        _run_eval(**kwargs)
        got[tag] = _answers_by_subject_and_query(out_dir)

    # Anchored, so "both runs are equally wrong" cannot pass: I50 is subject 1's only, J44 is
    # subject 2's only, both inside the 30-day horizon.
    assert got["off"] == {
        (1, "ICD//CIRC//I50"): True,
        (1, "ICD//RESP//J44"): False,
        (2, "ICD//CIRC//I50"): False,
        (2, "ICD//RESP//J44"): True,
    }
    assert got["on"] == got["off"], "the closure explosion changed a leaf query's answer"


# ── 3. the differential: eval vs training on the same question ──────────


def test_eval_and_training_paths_agree_on_the_same_ancestor_query(
    tmp_path: Path, data_dir: Path, ontology_dir: Path
):
    """One split, one ontology, one ancestor query, one horizon — two samplers, one answer.

    The training sampler is driven with ``MED//STATIN`` as its only leaf code and a fixed horizon
    (``duration_min == duration_max``) — the ontology adds the other ancestor nodes to its universe,
    so only its ``MED//STATIN`` draws are compared; the contexts it sampled and the horizon it
    emitted are then handed back to the eval grid as ``contexts_path`` / a designed spec, so the two
    paths answer a literally identical question about literally identical contexts.

    Divergence here **is** the defect: before the fix the training path exploded the events and
    the eval path did not.  The agreed answers are pinned to the hand-derived truth as well, so the
    two paths agreeing on a wrong label is not a pass either.  Sampled contexts are a subject's
    non-first event times, so the query is the prefix-only node ``MED//STATIN``: subject 1's
    ``ATORVA`` (2020-06-10) lies inside the 30-day window only from its 2020-06-06 context, and
    subject 2 has no statin at all.
    """
    ancestor_codes = tmp_path / "ancestor_only.yaml"
    ancestor_codes.write_text(yaml.safe_dump([ANCESTOR_ONLY_NODE]))
    horizon = HORIZON

    train_out = tmp_path / "train_tasks"
    _run_train(
        data_dir=data_dir,
        out_dir=train_out,
        query_codes=ancestor_codes,
        split=SPLIT,
        num_training_sequence_examples=64,
        min_prediction_times_per_subject=1,
        min_queries=1,
        max_queries=1,
        duration_min=horizon,
        duration_max=horizon,
        # The comparison is against a fixed-horizon designed spec, so the training draw must be
        # horizon-only regardless of the config's default event-bound share.
        eventbound_fraction=0,
        ontology_dir=ontology_dir,
    )
    train_df = pl.concat([pl.read_parquet(fp) for fp in sorted((train_out / SPLIT).glob("*.parquet"))])
    train_answers = {
        k: v for k, v in _answers_by_context_and_query(train_df).items() if k[2] == ANCESTOR_ONLY_NODE
    }
    key = (1, datetime(2020, 6, 6), ANCESTOR_ONLY_NODE)
    assert key in train_answers, (
        "the one context whose answer is True was not sampled; raise num_training_sequence_examples"
    )

    # Take the horizon the training path actually emitted (post float32 round-trip) so the eval
    # spec cannot differ from it by an epsilon at the window boundary.
    horizons = {float(d) for row in train_df["durations"].to_list() for d in row}
    assert len(horizons) == 1, f"expected one fixed horizon from the training draw, got {horizons}"
    horizon = horizons.pop()

    cohort = tmp_path / "sampled_cohort.parquet"
    train_df.select("subject_id", "prediction_time").unique().write_parquet(cohort)
    specs = tmp_path / "ancestor_spec.yaml"
    specs.write_text(yaml.safe_dump({"statin": [entry(ANCESTOR_ONLY_NODE, horizon)]}))
    eval_out = tmp_path / "eval_grid"
    _run_eval(
        data_dir=data_dir,
        out_dir=eval_out,
        query_codes=ancestor_codes,
        contexts_path=cohort,
        sequences_path=specs,
        split=SPLIT,
        ontology_dir=ontology_dir,
    )
    eval_answers = _answers_by_context_and_query(_grid(eval_out))

    assert train_answers, "no MED//STATIN query was drawn at all; raise num_training_sequence_examples"
    assert {k: eval_answers[k] for k in train_answers} == train_answers, (
        "the two samplers disagree about the same ancestor query on the same contexts: "
        f"eval={eval_answers}, training={train_answers}"
    )
    assert eval_answers == {k: k == key for k in eval_answers}


# ── 4. the ontology-off path writes nothing but its parquets ─────────────


def test_ontology_off_records_a_null_ontology_and_keeps_the_output_tree_clean(
    tmp_path: Path, data_dir: Path, cohort: Path, codes_yaml: Path
):
    """With ``ontology_dir: null`` — the default — the sidecar records ``ontology_fingerprint: null``.

    The sidecar is always written (it also carries the specs and cohort fingerprints), but only ever
    to the artifacts sibling: the output tree itself is exactly the per-shard grid and its
    ``eval_unique`` mirror.
    """
    out_dir = tmp_path / "grid"
    _run_eval(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=codes_yaml,
        contexts_path=cohort,
        split=SPLIT,
        num_evaluation_sequences=8,
        seed=7,
    )
    for fp in _grid_fps(out_dir):
        recorded = eval_seq._recorded_fingerprint(out_dir, fp)
        assert recorded is not None and recorded["ontology_fingerprint"] is None
    on_disk = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())
    assert on_disk == [
        f"eval/{SPLIT}/0.parquet",
        f"eval/{SPLIT}/1.parquet",
        f"eval_unique/{SPLIT}/0.parquet",
        f"eval_unique/{SPLIT}/1.parquet",
    ]


# ── 5. staleness: an existing output is not automatically a current one ─

#: The four labels a *correct* run of the one-query ``ICD//CIRC`` grid must produce, and the four a
#: run through the ``stale_ontology_dir`` closure must produce.  Anchored rather than merely
#: "different from each other", so a guard that relabels into the wrong answer is not a pass.
CIRC_TRUTH = {(1, ANCESTOR): True, (2, ANCESTOR): False}
CIRC_UNDER_STALE_CLOSURE = {(1, ANCESTOR): False, (2, ANCESTOR): False}


def test_a_grid_labeled_under_a_different_ontology_is_relabeled_not_skipped(
    tmp_path: Path,
    data_dir: Path,
    cohort: Path,
    codes_yaml: Path,
    ontology_dir: Path,
    stale_ontology_dir: Path,
):
    """Neither output tag encodes the ontology, so ``overwrite=false`` must not trust mere existence.

    Both runs below write to exactly the same path; they differ in the closure they label
    through — the "ontology rebuilt from a different ``codes.parquet``" case.  A guard that
    hashed only the universe could call these two runs identical.  With an
    existence-only skip the second run keeps the first's ``False`` and the grid reports the
    ancestor feature as dead while the config says it is on.
    """
    out_dir = tmp_path / "grid"
    specs = _specs_yaml(tmp_path, circ_30d=ANCESTOR)
    common = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "contexts_path": cohort,
        "sequences_path": specs,
        "split": SPLIT,
    }
    fps = _grid_fps(out_dir)

    _run_eval(**common, ontology_dir=stale_ontology_dir)
    assert _answers_by_subject_and_query(out_dir) == CIRC_UNDER_STALE_CLOSURE, (
        "the stale ontology has no closure row for subject 1's in-window event, so False is right here"
    )

    _run_eval(**common, ontology_dir=ontology_dir)
    assert _answers_by_subject_and_query(out_dir) == CIRC_TRUTH, (
        "labels from the previous ontology survived because the output file already existed"
    )

    # ...and the guard keys on the ontology rather than relabeling unconditionally: re-running the
    # same configuration leaves the files untouched (a fresh write means a fresh inode).
    provenances = [eval_seq._provenance_path(out_dir, fp) for fp in fps]
    recorded = [json.loads(p.read_text())["ontology_fingerprint"] for p in provenances]
    inodes = [fp.stat().st_ino for fp in fps]
    _run_eval(**common, ontology_dir=ontology_dir)
    assert [fp.stat().st_ino for fp in fps] == inodes, "an unchanged ontology should have been skipped"
    assert [json.loads(p.read_text())["ontology_fingerprint"] for p in provenances] == recorded


def test_turning_the_ontology_off_relabels_and_nulls_the_ontology_provenance(
    tmp_path: Path,
    data_dir: Path,
    cohort: Path,
    codes_yaml: Path,
    ontology_dir: Path,
):
    """The other direction: an output built *with* an ontology is stale for a run without one.

    A leaf query answers the same either way, so what is at stake is provenance rather than these
    particular labels — but a sidecar still carrying the old ontology fingerprint would claim a
    never-exploded parquet was exploded, and the next ontology run would trust it and skip.
    """
    out_dir = tmp_path / "grid"
    specs = _specs_yaml(tmp_path, i50="ICD//CIRC//I50")
    common = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "contexts_path": cohort,
        "sequences_path": specs,
        "split": SPLIT,
    }
    fps = _grid_fps(out_dir)
    leaf_truth = {(1, "ICD//CIRC//I50"): True, (2, "ICD//CIRC//I50"): False}

    _run_eval(**common, ontology_dir=ontology_dir)
    assert _answers_by_subject_and_query(out_dir) == leaf_truth
    assert all(eval_seq._provenance_path(out_dir, fp).is_file() for fp in fps)
    inodes = [fp.stat().st_ino for fp in fps]

    _run_eval(**common)
    assert all(fp.stat().st_ino != i for fp, i in zip(fps, inodes, strict=True)), (
        "dropping the ontology must relabel, not skip"
    )
    assert _answers_by_subject_and_query(out_dir) == leaf_truth
    assert all(eval_seq._recorded_fingerprint(out_dir, fp)["ontology_fingerprint"] is None for fp in fps)


def test_a_sidecar_less_grid_is_relabeled_when_an_ontology_is_turned_on(
    tmp_path: Path,
    data_dir: Path,
    cohort: Path,
    codes_yaml: Path,
    ontology_dir: Path,
    stale_ontology_dir: Path,
):
    """A sidecar-less output is the *pre-fix* on-disk state, and it is stale for an ontology run.

    "No sidecar" is not "unknown, assume fine": it means precisely "labeled under some ontology we
    can no longer identify".  Every grid written before this branch existed is in that state, so a
    guard that read a missing sidecar as current would skip exactly the outputs the branch was
    written to correct — turning the ontology on would leave the wrong labels in place and change
    nothing but the log line.

    The sidecar-less state is produced here by labeling under the *stale* ontology and then
    deleting the sidecar.  It used to be produced by labeling with no ontology at all, which is no
    longer reachable for this spec: the ancestor under test is now the minted subtree node
    ``ICD//CIRC//ANY``, which exists only in an ontology, so an ontology-less run rejects the spec
    outright rather than labeling it all-``False``.  That rejection is the intended behaviour — a
    grid cannot silently measure an ancestor the run knows nothing about — so the staleness claim
    is exercised through the reachable path instead.
    """
    out_dir = tmp_path / "grid"
    specs = _specs_yaml(tmp_path, circ_30d=ANCESTOR)
    common = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "contexts_path": cohort,
        "sequences_path": specs,
        "split": SPLIT,
    }
    fps = _grid_fps(out_dir)

    _run_eval(**common, ontology_dir=stale_ontology_dir)
    assert _answers_by_subject_and_query(out_dir) == CIRC_UNDER_STALE_CLOSURE, (
        "the stale closure is missing the one code that links subject 1's in-window event"
    )
    for fp in fps:
        eval_seq._provenance_path(out_dir, fp).unlink()

    _run_eval(**common, ontology_dir=ontology_dir)
    assert _answers_by_subject_and_query(out_dir) == CIRC_TRUTH, (
        "a sidecar-less (pre-fix) output was treated as current, so turning the ontology on was a no-op"
    )


def test_rebuilding_the_ontology_in_place_relabels_the_grid(
    tmp_path: Path,
    data_dir: Path,
    cohort: Path,
    codes_yaml: Path,
):
    """``ontology_dir`` is unchanged between these two runs; only what is *inside* it changes.

    This is what re-running ``EQ_build_ontology`` after the cohort's ``codes.parquet`` grew looks
    like from the eval sampler's side: same configured path, same output path, a different closure.
    A guard that recorded the ontology's *path* (or its mtime, or its name) instead of a digest of
    its contents would call the second run current and keep the first's labels — and every other
    staleness test here switches between two differently-named directories, so none of them can
    tell a content digest from a path.
    """
    out_dir = tmp_path / "grid"
    ontology = _write_ontology(tmp_path / "ontology_in_place", STALE_LEAVES)
    specs = _specs_yaml(tmp_path, circ_30d=ANCESTOR)
    common = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "contexts_path": cohort,
        "sequences_path": specs,
        "split": SPLIT,
        "ontology_dir": ontology,
    }

    _run_eval(**common)
    assert _answers_by_subject_and_query(out_dir) == CIRC_UNDER_STALE_CLOSURE

    _write_ontology(ontology, LEAVES)  # same directory, rebuilt from the full vocabulary
    _run_eval(**common)
    assert _answers_by_subject_and_query(out_dir) == CIRC_TRUTH, (
        "the ontology was rebuilt in place and the grid kept the old closure's labels; the guard "
        "is keyed on the ontology's location rather than its contents"
    )


def test_each_shard_output_carries_its_own_provenance(
    tmp_path: Path,
    data_dir: Path,
    cohort: Path,
    codes_yaml: Path,
    ontology_dir: Path,
    stale_ontology_dir: Path,
):
    """Two shards, one run: neither shard's sidecar may answer for the other.

    Every shard's output is ``{shard}.parquet`` under the same ``eval/{split}`` dir, so a sidecar
    keyed on anything less than the file's path below ``out_dir`` collapses the whole split onto
    one JSON file.  The failure that follows is order-dependent and completely silent: the loop
    relabels the first shard and rewrites the shared sidecar with the *new* fingerprint, and every
    later shard then reads that fresh fingerprint, calls itself current and keeps the previous
    ontology's labels.

    Subject 1 (the one whose ancestor answer flips) is on shard ``0``, the first labeled; subject
    2 is on shard ``1``.  Both specs ask the same ancestor question at different horizons, so the
    whole grid must flip from all-``False`` to subject 1's ``True`` on both shards.
    """
    out_dir = tmp_path / "grid"
    specs = tmp_path / "two_specs.yaml"
    specs.write_text(
        yaml.safe_dump({"circ_30d": [entry(ANCESTOR, 30.0)], "circ_60d": [entry(ANCESTOR, 60.0)]})
    )
    common = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "contexts_path": cohort,
        "sequences_path": specs,
        "split": SPLIT,
    }
    fps = _grid_fps(out_dir)

    _run_eval(**common, ontology_dir=stale_ontology_dir)
    assert _answers_by_subject_and_query(out_dir) == CIRC_UNDER_STALE_CLOSURE
    assert [_grid(fp).height for fp in fps] == [2, 2], "one context x two specs per shard"

    _run_eval(**common, ontology_dir=ontology_dir)
    assert _answers_by_subject_and_query(out_dir) == CIRC_TRUTH, (
        "one of the two shards kept the previous ontology's labels: the sidecars collided, so "
        "relabeling the first shard made the second one look current"
    )

    # The mechanism, pinned directly: distinct outputs get distinct sidecars.
    sidecars = [eval_seq._provenance_path(out_dir, fp) for fp in fps]
    assert len(set(sidecars)) == 2, f"the two outputs share one provenance file: {sidecars}"
    assert all(sc.is_file() for sc in sidecars)


def test_the_output_tree_holds_only_parquets_when_a_sidecar_is_written(
    tmp_path: Path,
    data_dir: Path,
    cohort: Path,
    codes_yaml: Path,
    ontology_dir: Path,
):
    """The provenance sidecar must land in the artifacts sibling, never in the output root.

    ``EQ_predict_multitask`` consumes an eval grid by rglobbing its output directory, so invariant
    7 — the final-output tree holds nothing but the parquets — is load-bearing, not tidiness.  A
    sidecar written next to (or under) the parquet is a well-formed JSON file that the consumer
    would try to read as a task frame.

    The suite's other rglob assertion lives in the ontology-**off** test, which by construction
    writes no sidecar at all; only a run that actually writes one can catch a regression in where
    it lands.
    """
    out_dir = tmp_path / "grid"
    specs = _specs_yaml(tmp_path, circ_30d=ANCESTOR)
    _run_eval(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=codes_yaml,
        contexts_path=cohort,
        sequences_path=specs,
        split=SPLIT,
        ontology_dir=ontology_dir,
    )
    fps = _grid_fps(out_dir)

    sidecars = [eval_seq._provenance_path(out_dir, fp) for fp in fps]
    assert all(sc.is_file() for sc in sidecars), "this run must write sidecars, or the invariant is vacuous"

    on_disk = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())
    assert on_disk == sorted(
        [fp.relative_to(out_dir).as_posix() for fp in fps]
        + [f"eval_unique/{SPLIT}/{shard}.parquet" for shard in ("0", "1")]
    ), f"the output tree must hold nothing but its parquets; found {on_disk}"
    assert not any(sc.is_relative_to(out_dir) for sc in sidecars), f"a sidecar is inside {out_dir}"
    # Invariant 7 also forbids ``_``-prefixed entries in the output root, directories included.
    assert not list(out_dir.rglob("_*")), f"stray private entries under {out_dir}"

    # The exact layout, mirroring ``sample_tasks.labeled_fingerprint_path``: provenance for the
    # output at ``{out_dir}/{rel}.parquet`` lives at ``{out_dir}_artifacts/_labeled/{rel}.json``.
    # Pinned so the sidecar tree stays recognisable to a human debugging a skipped run, and stays
    # segregated from whatever else the artifacts root accumulates.
    expected = [
        eval_seq.default_artifacts_dir(out_dir)
        / LABELED_DIRNAME
        / fp.relative_to(out_dir).with_suffix(".json")
        for fp in fps
    ]
    assert sidecars == expected


def test_a_failed_output_write_leaves_no_provenance_behind(
    tmp_path: Path, data_dir: Path, cohort: Path, codes_yaml: Path, ontology_dir: Path, monkeypatch
):
    """The sidecar is a commit marker, so it may only appear *after* its parquet is committed.

    Written first, it survives a crashed or failed parquet write and then describes whatever stale output
    happens to be sitting at that path — the next run reads a fingerprint that matches its configuration,
    skips, and ships the previous ontology's labels.  That is the same silent-wrong-label ending as an
    existence-only skip, reached through the write ordering instead, and only a failed write can distinguish
    the two orderings.
    """
    out_dir = tmp_path / "grid"
    specs = _specs_yaml(tmp_path, circ_30d=ANCESTOR)
    _run_eval(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=codes_yaml,
        contexts_path=cohort,
        sequences_path=specs,
        split=SPLIT,
        ontology_dir=ontology_dir,
    )
    labeled = _grid(out_dir)

    def boom(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(eval_seq, "_atomic_write_parquet", boom)
    target = out_dir / "eval" / SPLIT / "other.parquet"
    with pytest.raises(OSError):
        eval_seq._write(labeled, target, out_dir, "some-ontology-fingerprint")

    assert not eval_seq._provenance_path(out_dir, target).exists(), (
        "a provenance sidecar outlived the failed write of the parquet it describes; the next run "
        "will trust it and skip"
    )


# ── 5b. staleness, the universe half: the closure is only one input ──────


def test_reordering_the_query_universe_relabels_the_grid(
    tmp_path: Path, data_dir: Path, cohort: Path, ontology_dir: Path
):
    """The universe half must be digested in order: the slot index is what the draw indexes into.

    These two runs are handed the same seven codes as the same multiset, permuted.  The sampler
    draws slot *indices* from a seeded stream, so the permutation moves which code each draw
    lands on — the grids genuinely differ, and the second must be written.  An order-insensitive
    digest (a sorted, or set-valued, or count-valued universe hash) reports them as the same run
    and leaves the first grid in place.
    """
    out_dir = tmp_path / "grid"
    straight = tmp_path / "codes_straight.yaml"
    straight.write_text(yaml.safe_dump(LEAVES))
    swapped = tmp_path / "codes_swapped.yaml"
    swapped.write_text(yaml.safe_dump([LEAVES[1], LEAVES[0], *LEAVES[2:]]))
    common = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "contexts_path": cohort,
        "split": SPLIT,
        "num_evaluation_sequences": 96,
        "ontology_dir": ontology_dir,
    }
    _run_eval(**common, query_codes=straight)
    first = Counter(_all_queries(out_dir))
    _run_eval(**common, query_codes=swapped)
    second = Counter(_all_queries(out_dir))

    assert set(first) == set(second), "fixture drift: a permutation cannot change which codes exist"
    assert first != second, (
        "the permuted universe produced a byte-identical grid: the run was skipped, because the "
        f"universe is digested without its slot index. counts={dict(second)}"
    )
    # The permutation swapped the first two slots, so their draw counts swap with them.
    assert (second[LEAVES[0]], second[LEAVES[1]]) == (first[LEAVES[1]], first[LEAVES[0]])


def test_ontology_fingerprint_separates_both_of_its_halves(ontology_dir: Path, stale_ontology_dir: Path):
    """The digest itself, at unit range: which changes must move it and which must not.

    Every end-to-end staleness test above can only observe the fingerprint through a skip
    decision, which makes them silent about a digest that happens to collide for a reason no
    fixture reaches.  This pins the contract directly — same inputs, same digest; a different
    closure, a different universe *membership*, *order* or *multiplicity*, each a different digest
    — and it is the only place ``None`` (no ontology configured) is pinned as its own value rather
    than as the absence of a sidecar.
    """
    base = list(LEAVES)
    fingerprint = eval_seq._ontology_fingerprint
    digest = fingerprint(ontology_dir, base)

    assert fingerprint(ontology_dir, list(base)) == digest, "an identical run must digest the same"
    assert fingerprint(None, base) is None, "no ontology must digest to None, not to a hash of one"

    # The closure half.
    assert fingerprint(stale_ontology_dir, base) != digest, "a different closure must move the digest"

    # The universe half: membership, order and multiplicity each on their own.
    assert fingerprint(ontology_dir, [*base, "ICD"]) != digest, "an added ancestor slot must move it"
    assert fingerprint(ontology_dir, [base[1], base[0], *base[2:]]) != digest, (
        "the universe is digested without its slot index, so a permutation — which changes what "
        "every draw resolves to — reads as the same run"
    )
    assert fingerprint(ontology_dir, [*base, base[0]]) != digest, (
        "a repeated slot must move the digest: the digest is of the slot list, not of its set"
    )


# ── 6. config <-> code wiring ───────────────────────────────────────────


def test_eval_ontology_config_keys_match_training_and_are_actually_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    cohort: Path,
    codes_yaml: Path,
    ontology_dir: Path,
):
    """The eval config's ontology block must agree with training's *and* reach the code.

    The first half mirrors the existing key-parity tests
    (``test_conditional_queries.py::test_eval_sampling_defaults_stay_in_training_distribution``):
    a drifted default gives programmatic callers a different grid than the CLI produces.

    The second half is the one that would have caught this defect.  Hydra already rejects a
    *typo'd* key (asserted below), so the failure mode was never an unknown key being swallowed —
    it was a **known, documented** key that no line of code read.  Only watching the values arrive
    at ``build_query_universe`` and ``run_worker`` tells those two apart.

    The multitask sampler ships the same pair of knobs (``ontology_dir`` plus its own
    ``ontology_mode``) out of the same configs directory, and is exposed to exactly the same defect:
    a documented default that nothing reads leaves an ontology run silently leaf-only.  So its two
    defaults are pinned here too, against the functions that consume them — the event-side closure
    expansion and the mode resolution — rather than against a shape.
    """
    configs = Path(eval_seq.CONFIGS)
    eval_cfg = yaml.safe_load((configs / "sample_evaluation_query_sequences_config.yaml").read_text())
    train_cfg = yaml.safe_load((configs / "sample_query_sequences_config.yaml").read_text())
    multitask_cfg = yaml.safe_load((configs / "sample_multitask_sequences_config.yaml").read_text())

    assert eval_cfg["ontology_dir"] == train_cfg["ontology_dir"], "ontology_dir default differs"
    assert eval_cfg["ontology_dir"] is None
    assert inspect.signature(eval_seq.run_worker).parameters["ontology_dir"].default is None

    # The multitask sampler: off by default on both keys, and both keys read by the code.
    assert multitask_cfg["ontology_dir"] is None, "the multitask sampler's ontology_dir default drifted"
    assert multitask_cfg["ontology_mode"] is None, (
        "ontology_mode must default to null — the mode is derived from whether an ontology_dir was "
        "given, so a hardcoded default would silently override half of the pair"
    )
    assert inspect.signature(sms.prepare_events_for_labeling).parameters["ontology_dir"].default is None
    assert callable(getattr(sms, "resolve_ontology_mode", None)), (
        "sample_multitask_sequences must resolve the (ontology_dir, ontology_mode) pair itself; "
        "without it the config's ontology_mode reaches no code at all"
    )
    assert sms.resolve_ontology_mode(None, None) == "none", "the shipped defaults must mean no ontology"
    assert sms.resolve_ontology_mode(str(ontology_dir), None) == "boundaries+conditions", (
        "a configured ontology_dir with the default null mode must turn both ancestor halves on"
    )

    # A typo'd key is already an error in this repo's plain-YAML Hydra setup — there is no schema
    # to add, and appending one with `+` is the documented escape hatch.
    with (
        pytest.raises(ConfigCompositionException),
        initialize_config_dir(config_dir=eval_seq.CONFIGS, version_base=None),
    ):
        compose(
            config_name="sample_evaluation_query_sequences_config",
            overrides=["ontolgy_dir=nowhere"],
        )

    seen: dict[str, object] = {}

    # Stubs, not wrappers: the point is only that the configured values *arrive* here.  What the
    # real functions then do with them is what every other test in this file measures.  (The
    # ontology has to be a real one: ``main`` resolves the model vocabulary and the provenance
    # fingerprint from it before any worker runs.)
    def spy_universe(query_codes, **kwargs):
        seen["universe_kwargs"] = kwargs
        return list(query_codes)

    def spy_run_worker(**kwargs):
        seen["run_worker_kwargs"] = kwargs
        return []

    monkeypatch.setattr(eval_seq, "build_query_universe", spy_universe)
    monkeypatch.setattr(eval_seq, "run_worker", spy_run_worker)

    _run_eval(
        data_dir=data_dir,
        out_dir=tmp_path / "grid",
        query_codes=codes_yaml,
        contexts_path=cohort,
        split=SPLIT,
        ontology_dir=ontology_dir,
        seed=1,
    )

    assert seen["universe_kwargs"]["ontology_dir"] == str(ontology_dir)
    assert seen["run_worker_kwargs"]["ontology_dir"] == str(ontology_dir)
