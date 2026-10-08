"""Randomised differential testing: production labeller vs the independent oracle.

The golden truth table pins the cases a human thought of.  This pins the ones nobody thought of:
random small event streams, random small DAG ontologies, random queries of both forms, compared
row by row.  Any disagreement is reported with the seed and a minimised failing example so it can
be turned into a permanent regression case.

Deliberately small worlds.  Three subjects and a handful of codes on a ten-day grid means
timestamp collisions, empty windows, boundaries that never fire and ancestors with no events all
happen constantly, instead of once in ten thousand draws.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from tests.ontology_suite import production as prod
from tests.ontology_suite.oracle import (
    Event,
    Ontology,
    label_duration,
    label_event_bounded,
)

BASE = datetime(2024, 6, 1)
N_WORLDS = 120

#: Code-name atoms chosen so `//`-prefix collisions are common: `A//B` is both a code in its own
#: right and a prefix of `A//B//C`.
LEAF_POOL = [
    "A",
    "A//B",
    "A//B//C",
    "A//B//D",
    "A//E",
    "Z//Q",
    "Z//Q//R",
    "SOLO",
    "T//U//V",
    "ADM",
    "DIS",
]
GROUPER_POOL = ["GRP//ONE", "GRP//TWO", "OTHER", "DEEP//G//H"]


class World:
    """One random (vocabulary, ontology, event stream) triple."""

    def __init__(self, seed: int):
        self.seed = seed
        rng = random.Random(seed)
        self.rng = rng

        n_leaves = rng.randint(3, 8)
        self.leaves = sorted(rng.sample(LEAF_POOL, n_leaves))
        # TIMELINE//END is always in the vocabulary: censoring is expressed as a query on it.
        if "TIMELINE//END" not in self.leaves:
            self.leaves.append("TIMELINE//END")

        # Declared parent edges.  Parents are usually pure groupers, but sometimes another LEAF
        # (which exercises traversal *through* a leaf) and sometimes a deliberate cycle.
        self.declared: dict[str, list[str]] = {}
        for leaf in self.leaves:
            if leaf == "TIMELINE//END" or rng.random() > 0.45:
                continue
            roll = rng.random()
            if roll < 0.65:
                parents = [rng.choice(GROUPER_POOL)]
            elif roll < 0.9:
                others = [c for c in self.leaves if c != leaf and c != "TIMELINE//END"]
                parents = [rng.choice(others)] if others else [rng.choice(GROUPER_POOL)]
            else:
                parents = [rng.choice(GROUPER_POOL), rng.choice(GROUPER_POOL)]
            self.declared[leaf] = sorted(set(parents))

        self.onto = Ontology(self.leaves, self.declared)

        # Events: three subjects on a coarse day grid so collisions are frequent.
        self.events: list[Event] = []
        emitters = [c for c in self.leaves if c != "TIMELINE//END"]
        for sid in (1, 2, 3):
            for _ in range(rng.randint(0, 7)):
                self.events.append(
                    Event(sid, BASE + timedelta(days=rng.randint(0, 10)), rng.choice(emitters))
                )
            if rng.random() < 0.8:  # most, not all, records have an explicit end
                self.events.append(Event(sid, BASE + timedelta(days=rng.randint(0, 12)), "TIMELINE//END"))

        self.nodes = sorted(self.onto.nodes)

    def queries(self, n: int) -> list[tuple]:
        """`(subject, t, query, duration, boundary)` rows; boundary None for duration form."""
        rng = self.rng
        out = []
        for _ in range(n):
            sid = rng.randint(1, 3)
            t = BASE + timedelta(days=rng.randint(0, 10))
            q = rng.choice(self.nodes)
            if rng.random() < 0.5:
                out.append((sid, t, q, float(rng.choice([1, 2, 3, 5, 10])), None))
            else:
                out.append((sid, t, q, prod.EVENT_BOUND_SENTINEL, rng.choice(self.nodes)))
        return out


def _oracle(world: World, row: tuple) -> bool:
    sid, t, q, dur, bound = row
    if bound is None:
        return label_duration(world.events, sid, world.onto, t, q, dur)
    return label_event_bounded(world.events, sid, world.onto, t, q, bound)


def _describe(world: World, row: tuple, got: bool, want: bool) -> str:
    sid, t, q, dur, bound = row
    evs = sorted((e for e in world.events if e.subject_id == sid), key=lambda e: e.time)
    lines = [
        "",
        f"seed          : {world.seed}",
        f"leaves        : {world.leaves}",
        f"declared      : {world.declared}",
        f"query         : node={q!r} " + (f"duration={dur}" if bound is None else f"bound={bound!r}"),
        f"subject       : {sid}   prediction_time={t:%Y-%m-%d}",
        f"production    : {got}",
        f"oracle        : {want}",
        f"query is a    : {'leaf' if q in world.onto.leaves else 'ancestor node'}",
        "timeline      :",
    ]
    for e in evs:
        mark = "  <-- matches query" if world.onto.matches(q, e.code) else ""
        lines.append(f"    {e.time:%Y-%m-%d}  {e.code}{mark}")
    return "\n".join(lines)


def _minimise(world: World, row: tuple) -> tuple:
    """Drop events that do not change the disagreement, so the report is as small as possible."""
    sid = row[0]
    keep = list(world.events)
    i = 0
    while i < len(keep):
        trial = keep[:i] + keep[i + 1 :]
        saved, world.events = world.events, trial
        try:
            got = prod.label([row], trial, closure_df=world.closure, with_bounds=row[4] is not None)[0]
            want = _oracle(world, row)
            still_disagrees = got != want
        finally:
            world.events = saved
        if still_disagrees and any(e.subject_id == sid for e in trial):
            keep = trial
        else:
            i += 1
    return keep


@pytest.mark.parametrize("seed", range(N_WORLDS))
def test_production_agrees_with_oracle(seed: int):
    world = World(seed)
    _nodes_df, _mix, closure = prod.build_artifacts(world.leaves, world.declared)
    world.closure = closure

    # The two implementations must first agree on WHAT the vocabulary is; otherwise a label
    # comparison is meaningless.
    prod_nodes = set(_nodes_df["node_name"].to_list())
    assert prod_nodes == set(world.nodes), (
        f"seed {seed}: node sets disagree.\n"
        f"  only in production: {sorted(prod_nodes - set(world.nodes))}\n"
        f"  only in oracle    : {sorted(set(world.nodes) - prod_nodes)}\n"
        f"  leaves={world.leaves} declared={world.declared}"
    )

    rows = world.queries(24)
    dur_rows = [r for r in rows if r[4] is None]
    evt_rows = [r for r in rows if r[4] is not None]

    for subset, with_bounds in ((dur_rows, False), (evt_rows, True)):
        if not subset:
            continue
        got = prod.label(subset, world.events, closure_df=closure, with_bounds=with_bounds)
        for row, g in zip(subset, got, strict=True):
            want = _oracle(world, row)
            if g != want:
                world.events = _minimise(world, row)
                pytest.fail(_describe(world, row, g, want))


def test_ontology_off_matches_identity_ontology():
    """Leaf-only labeling must be identical with the ontology off and with a closure that only ever pairs a
    code with itself.

    This is the `ancestor_fraction=0` safety claim, stated as an equality rather than as a vibe.
    """
    import polars as pl

    for seed in range(30):
        world = World(seed)
        identity = pl.DataFrame({"event_code": world.leaves, "query_node": world.leaves})
        rows = [r for r in world.queries(20) if r[2] in world.onto.leaves]
        if not rows:
            continue
        dur = [r for r in rows if r[4] is None]
        if not dur:
            continue
        off = prod.label(dur, world.events, closure_df=None, with_bounds=False)
        ident = prod.label(dur, world.events, closure_df=identity, with_bounds=False)
        assert off == ident, f"seed {seed}: identity closure changed leaf answers"
