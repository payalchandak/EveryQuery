"""A tiny synthetic MEDS cohort whose every answer was worked out by hand.

The expected column in :data:`TRUTH_TABLE` was written by reading the contract, not by running
the labeler.  Tests assert it three ways:

1. the independent :mod:`oracle` reproduces it   (proves the oracle is right);
2. the production labeler reproduces it          (proves the production code is right);
3. ontology-disabled leaf answers are unchanged  (proves enabling the DAG is non-destructive).

``docs/ontology_truth_table.md`` is generated from this module, so the prose and the fixture
cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .oracle import Event, Ontology

# --------------------------------------------------------------------------------------------
# Vocabulary + DAG
# --------------------------------------------------------------------------------------------

#: Codes that really exist in the cohort's ``codes.parquet``.  Everything else is a pure
#: ancestor node, addressable as a subtree query but never emitted as a raw event.
LEAVES: list[str] = [
    # The clinically recognisable ontology-required family: there is deliberately NO bare
    # `READMISSION` code, so `READMISSION` can only ever be answered through descendants.
    "READMISSION//CHILD_A",
    "READMISSION//CHILD_B",
    # Two internal ancestors (`DX//CARDIO`, `DX//RESP`) under a root (`DX`), with siblings.
    "DX//CARDIO//MI",
    "DX//CARDIO//HF",
    "DX//RESP//COPD",
    # An unrelated subtree.
    "LAB//GLU",
    # A leaf that is ALSO a `//`-prefix of another leaf.  A query for `PROC//X` must keep its
    # exact meaning; widening it to its descendants is the prefix-absorption defect.
    "PROC//X",
    "PROC//X//SUB",
    # A two-hop DECLARED chain: WARFARIN_SODIUM -> MED//WARFARIN -> ATC//B01AA.
    # `MED//WARFARIN` is itself a leaf, which is what makes the second hop easy to lose.
    "MED//WARFARIN",
    "MED//WARFARIN_SODIUM",
    # Boundary codes for event-bounded queries.
    "ADMISSION",
    "DISCHARGE",
    "TIMELINE//END",
]

#: ``parent_codes``-style declared edges.  `DX//CARDIO//MI` therefore has TWO parents -- its
#: `//`-prefix parent `DX//CARDIO` and its declared grouper `GRP//ACS` -- which is the multiple
#: inheritance the real cohort exhibits (max 2 parents over 5,123 codes).
DECLARED_PARENTS: dict[str, list[str]] = {
    "DX//CARDIO//MI": ["GRP//ACS"],
    "DX//CARDIO//HF": ["GRP//HF_GRP"],
    "MED//WARFARIN_SODIUM": ["MED//WARFARIN"],
    "MED//WARFARIN": ["ATC//B01AA"],
}

ONTOLOGY = Ontology(LEAVES, DECLARED_PARENTS)

#: The prediction time every query in the truth table is anchored at.
T0 = datetime(2024, 3, 1, 0, 0, 0)  # noqa: DTZ001 — naive ts is fine for synthetic fixtures


def _d(month: int, day: int, hour: int = 0) -> datetime:
    return datetime(2024, month, day, hour, 0, 0)  # noqa: DTZ001


# --------------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------------

EVENTS: list[Event] = [
    # --- Subject 1: duration-window endpoint cases -------------------------------------------
    Event(1, _d(2, 25), "DX//CARDIO//MI"),  # strictly BEFORE t0
    Event(1, _d(3, 1), "LAB//GLU"),  # EXACTLY at t0 -> excluded (open below)
    Event(1, _d(3, 3), "DX//RESP//COPD"),  # strictly inside a 7-day window
    Event(1, _d(3, 8), "PROC//X//SUB"),  # EXACTLY at the 7-day endpoint -> excluded
    Event(1, _d(3, 9), "READMISSION//CHILD_A"),  # just AFTER the 7-day endpoint
    Event(1, _d(3, 31), "TIMELINE//END"),  # record extends far past the window
    # --- Subject 2: event-boundary cases -----------------------------------------------------
    Event(2, _d(3, 2), "ADMISSION"),
    Event(2, _d(3, 4), "DX//CARDIO//HF"),  # target BEFORE the boundary
    Event(2, _d(3, 6), "DISCHARGE"),  # boundary occurrence #1
    Event(2, _d(3, 6), "READMISSION//CHILD_B"),  # target SHARING the boundary instant
    Event(2, _d(3, 10), "DX//CARDIO//MI"),  # target AFTER the boundary
    Event(2, _d(3, 12), "DISCHARGE"),  # boundary occurrence #2 (must be ignored)
    Event(2, _d(3, 20), "TIMELINE//END"),
    # --- Subject 3: missing boundary, early record end, declared 2-hop chain -----------------
    Event(3, _d(3, 2), "LAB//GLU"),
    Event(3, _d(3, 4), "MED//WARFARIN_SODIUM"),
    Event(3, _d(3, 5), "TIMELINE//END"),  # record ends BEFORE a 7-day endpoint
    # (deliberately no DISCHARGE anywhere -> the boundary never occurs)
    # --- Subject 4: repeats, and no matching event at all ------------------------------------
    Event(4, _d(3, 2), "LAB//GLU"),
    Event(4, _d(3, 3), "LAB//GLU"),  # repeated
    Event(4, _d(3, 4), "LAB//GLU"),  # repeated
    Event(4, _d(3, 25), "TIMELINE//END"),
    # --- Subject 5: the ontology-required readmission case -----------------------------------
    Event(5, _d(3, 2), "ADMISSION"),
    Event(5, _d(3, 5), "READMISSION//CHILD_B"),  # only descendant; no raw READMISSION event
    Event(5, _d(3, 9), "DISCHARGE"),
    Event(5, _d(3, 28), "TIMELINE//END"),
]


# --------------------------------------------------------------------------------------------
# Truth table
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One hand-computed expected answer.

    Attributes:
        case_id: stable identifier, used in test ids and in the generated markdown.
        subject_id: which synthetic subject.
        query: the queried node (leaf or ancestor).
        duration_days: horizon for a duration-bounded query; ``None`` when event-bounded.
        bound_event: boundary code for an event-bounded query; ``None`` when duration-bounded.
        expected: the hand-computed occurrence answer.
        censored: whether the record ends inside the window (the ``TIMELINE//END`` answer).
        ontology_required: the answer can only be produced through descendant closure.
        why: the reasoning, reproduced verbatim in the generated truth table.
    """

    case_id: str
    subject_id: int
    query: str
    duration_days: float | None
    bound_event: str | None
    expected: bool
    censored: bool
    ontology_required: bool
    why: str

    @property
    def form(self) -> str:
        return "event" if self.bound_event is not None else "duration"

    @property
    def target_kind(self) -> str:
        return "leaf" if self.query in ONTOLOGY.leaves else "ancestor"

    @property
    def boundary_kind(self) -> str | None:
        if self.bound_event is None:
            return None
        return "leaf" if self.bound_event in ONTOLOGY.leaves else "ancestor"


# fmt: off
TRUTH_TABLE: list[Case] = [
    # ---- Subject 1: duration-window boundary semantics --------------------------------------
    Case("S1-before-t0", 1, "DX//CARDIO//MI", 7.0, None, False, False, False,
         "The only MI is on 02-25, strictly before t0; the window is open below."),
    Case("S1-exactly-at-t0", 1, "LAB//GLU", 7.0, None, False, False, False,
         "LAB//GLU lands exactly ON t0. The lower bound is OPEN, so it does not count."),
    Case("S1-strictly-inside", 1, "DX//RESP//COPD", 7.0, None, True, False, False,
         "COPD on 03-03 is strictly inside (03-01, 03-08)."),
    Case("S1-exactly-at-endpoint", 1, "PROC//X//SUB", 7.0, None, False, False, False,
         "PROC//X//SUB lands exactly ON the 7-day horizon 03-08. Upper bound is OPEN."),
    Case("S1-endpoint-included-at-d8", 1, "PROC//X//SUB", 8.0, None, True, False, False,
         "Same event, one day more horizon: 03-08 is now strictly inside (03-01, 03-09)."),
    Case("S1-just-after-endpoint", 1, "READMISSION//CHILD_A", 7.0, None, False, False, False,
         "CHILD_A on 03-09 is past the 03-08 horizon."),
    Case("S1-leaf-prefix-stays-narrow-d7", 1, "PROC//X", 7.0, None, False, False, False,
         "PROC//X is a REAL leaf. No raw PROC//X event exists, so the answer is False even "
         "though its name prefixes PROC//X//SUB. Widening it is the prefix-absorption defect."),
    Case("S1-leaf-prefix-stays-narrow-d30", 1, "PROC//X", 30.0, None, False, False, False,
         "Same, with a horizon that comfortably covers the PROC//X//SUB event on 03-08."),
    Case("S1-ancestor-rolls-up", 1, "PROC", 30.0, None, True, False, True,
         "CONTROL for the two cases above: PROC is a pure ancestor, so PROC//X//SUB does roll "
         "up. If this were False the fixture would pass by the closure simply being empty."),
    Case("S1-dual-role-subtree-node", 1, "PROC//X//ANY", 30.0, None, True, False, True,
         "PROC//X is dual-role -- a real code AND the prefix of PROC//X//SUB -- so its subtree "
         "meaning lives under the minted node PROC//X//ANY, which PROC//X//SUB rolls up to. "
         "This is the rung that would be missing if dual-role names were left purely exact."),
    Case("S1-dual-role-subtree-node-d7", 1, "PROC//X//ANY", 7.0, None, False, False, True,
         "Same node, but PROC//X//SUB lands exactly on the 7-day horizon, so the window rule "
         "still applies to the minted node like any other."),
    Case("S1-readmission-outside", 1, "READMISSION", 7.0, None, False, False, True,
         "The only descendant (CHILD_A, 03-09) is outside the 7-day window."),
    Case("S1-readmission-inside-d30", 1, "READMISSION", 30.0, None, True, False, True,
         "ONTOLOGY-REQUIRED: no raw READMISSION event exists anywhere in the fixture; the "
         "True can only come from CHILD_A on 03-09 via descendant closure."),
    Case("S1-internal-ancestor-negative", 1, "DX//CARDIO", 30.0, None, False, False, True,
         "DX//CARDIO's only descendant event (MI) is on 02-25, before t0."),
    Case("S1-root-ancestor-positive", 1, "DX", 30.0, None, True, False, True,
         "DX rolls up DX//RESP//COPD on 03-03."),
    Case("S1-declared-ancestor-negative", 1, "GRP//ACS", 30.0, None, False, False, True,
         "GRP//ACS is MI's declared parent; MI is before t0, so False."),
    Case("S1-uncensored", 1, "TIMELINE//END", 7.0, None, False, False, False,
         "The record ends 03-31, well past the 03-08 horizon: the window is fully observed."),
    Case("S1-censored-at-d40", 1, "TIMELINE//END", 40.0, None, True, True, False,
         "With a 40-day horizon the record's end (03-31) falls inside: censored."),
    # ---- Subject 2: event-boundary semantics ------------------------------------------------
    Case("S2-target-before-boundary", 2, "DX//CARDIO//HF", None, "DISCHARGE", True, False, False,
         "First DISCHARGE after t0 is 03-06. HF on 03-04 is strictly inside (03-01, 03-06)."),
    Case("S2-target-at-boundary-instant", 2, "READMISSION//CHILD_B", None, "DISCHARGE", False, False, False,
         "CHILD_B shares the boundary's exact instant (03-06). The upper bound is OPEN, so a "
         "co-timestamped event does NOT count."),
    Case("S2-ancestor-at-boundary-instant", 2, "READMISSION", None, "DISCHARGE", False, False, True,
         "Same instant, asked as an ancestor: still excluded."),
    Case("S2-target-after-first-boundary", 2, "DX//CARDIO//MI", None, "DISCHARGE", False, False, False,
         "MI is on 03-10, after the FIRST discharge (03-06). The second discharge (03-12) must "
         "not be the one selected."),
    Case("S2-same-target-inside-d30", 2, "DX//CARDIO//MI", 30.0, None, True, True, False,
         "CONTROL: that same MI IS in the record and inside a 30-day duration window. Censored "
         "too -- subject 2's record ends 03-20, inside (03-01, 03-31)."),
    Case("S2-ancestor-target-leaf-boundary", 2, "DX//CARDIO", None, "DISCHARGE", True, False, True,
         "Ancestor target, leaf boundary: HF on 03-04 rolls up to DX//CARDIO, inside the window."),
    Case("S2-declared-ancestor-event-bounded", 2, "GRP//HF_GRP", None, "DISCHARGE", True, False, True,
         "ONTOLOGY-REQUIRED, event-bounded: GRP//HF_GRP is HF's declared parent only."),
    Case("S2-earlier-boundary-excludes", 2, "DX//CARDIO//HF", None, "ADMISSION", False, False, False,
         "First ADMISSION after t0 is 03-02, so the window is (03-01, 03-02); HF on 03-04 is out."),
    Case("S2-boundary-equals-query", 2, "ADMISSION", None, "ADMISSION", False, False, False,
         "When the boundary code equals the query, the first occurrence after t0 is "
         "simultaneously the boundary and the earliest candidate; nothing is strictly before "
         "itself, so the answer is unconditionally False."),
    # ---- Subject 3: missing boundary, early end, declared 2-hop chain -----------------------
    Case("S3-simple-positive", 3, "LAB//GLU", 7.0, None, True, True, False,
         "GLU on 03-02 is inside (03-01, 03-08). Note the record ends 03-05, so this row is "
         "positive AND censored -- occurrence is decided by what was observed."),
    Case("S3-censored", 3, "TIMELINE//END", 7.0, None, True, True, False,
         "The record ends 03-05, inside the 7-day window: the window is not fully observed."),
    Case("S3-negative-but-censored", 3, "DX//CARDIO//MI", 7.0, None, False, True, False,
         "No MI, but the record also ends early -- a False that must not be read as a "
         "confident negative."),
    Case("S3-declared-leaf-stays-narrow", 3, "MED//WARFARIN", 7.0, None, False, True, False,
         "MED//WARFARIN is a REAL leaf and no raw MED//WARFARIN event occurs. The event is "
         "MED//WARFARIN_SODIUM, which merely DECLARES it as a parent."),
    Case("S3-declared-leaf-subtree-node", 3, "MED//WARFARIN//ANY", 7.0, None, True, True, True,
         "MED//WARFARIN is dual-role by the OTHER route -- a real code named as another code's "
         "declared parent -- so it too gets a subtree node, and WARFARIN_SODIUM rolls up to it."),
    Case("S3-two-hop-declared-ancestor", 3, "ATC//B01AA", 7.0, None, True, True, True,
         "TWO declared hops above the event: WARFARIN_SODIUM -> MED//WARFARIN -> ATC//B01AA. "
         "Losing the second hop is the transitive-closure defect."),
    Case("S3-two-hop-declared-root", 3, "ATC", 7.0, None, True, True, True,
         "One `//`-prefix hop above ATC//B01AA, so three hops from the event."),
    Case("S3-one-hop-string-ancestor", 3, "MED", 7.0, None, True, True, True,
         "CONTROL for the two cases above: MED is a plain one-hop `//`-prefix ancestor of "
         "MED//WARFARIN_SODIUM and must be True even with declared edges broken."),
    Case("S3-missing-boundary-degenerates", 3, "LAB//GLU", None, "DISCHARGE", True, True, False,
         "Subject 3 has NO discharge. The window is left open to the end of the record, so the "
         "query degenerates to 'does GLU ever occur again' -- and it does, on 03-02."),
    Case("S3-missing-boundary-still-negative", 3, "DX//CARDIO//MI", None, "DISCHARGE", False, True, False,
         "Same open window, but no MI ever occurs."),
    # ---- Subject 4: repeats, absent targets, unrelated subtrees -----------------------------
    Case("S4-repeated-events", 4, "LAB//GLU", 7.0, None, True, False, False,
         "Three GLU events inside the window still answer True exactly once."),
    Case("S4-no-matching-event", 4, "DX//CARDIO//MI", 7.0, None, False, False, False,
         "This subject has no MI at all."),
    Case("S4-unrelated-subtree", 4, "READMISSION", 30.0, None, False, True, True,
         "GLU events cannot make a READMISSION query positive -- an unrelated subtree. Censored: "
         "subject 4's record ends 03-25, inside (03-01, 03-31)."),
    Case("S4-lab-ancestor", 4, "LAB", 7.0, None, True, False, True,
         "LAB rolls up LAB//GLU."),
    Case("S4-uncensored", 4, "TIMELINE//END", 7.0, None, False, False, False,
         "Record ends 03-25, past the 03-08 horizon."),
    # ---- Subject 5: the headline ontology-required case --------------------------------------
    Case("S5-ontology-required-duration", 5, "READMISSION", 7.0, None, True, False, True,
         "THE headline case. No raw READMISSION event exists anywhere; CHILD_B on 03-05 is the "
         "only evidence, and it is reachable only through the closure."),
    Case("S5-sibling-not-positive", 5, "READMISSION//CHILD_A", 7.0, None, False, False, False,
         "CHILD_B occurring must NOT make its sibling CHILD_A positive."),
    Case("S5-descendant-itself", 5, "READMISSION//CHILD_B", 7.0, None, True, False, False,
         "The descendant's own leaf query."),
    Case("S5-ontology-required-event-bounded", 5, "READMISSION", None, "DISCHARGE", True, False, True,
         "Same ancestor, event-bounded: first DISCHARGE is 03-09, and CHILD_B on 03-05 is "
         "strictly inside (03-01, 03-09)."),
    Case("S5-earlier-boundary-excludes-ancestor", 5, "READMISSION", None, "ADMISSION", False, False, True,
         "First ADMISSION is 03-02, so the window (03-01, 03-02) excludes CHILD_B on 03-05."),
]
# fmt: on


def cases_by_id() -> dict[str, Case]:
    return {c.case_id: c for c in TRUTH_TABLE}


def _assert_unique_ids() -> None:
    ids = [c.case_id for c in TRUTH_TABLE]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise AssertionError(f"duplicate case ids in TRUTH_TABLE: {dupes}")


_assert_unique_ids()
