"""A deliberately slow, obvious, independent labeler for EveryQuery queries.

This module is the *reference* semantics.  It is written from the prose contract, not from the
production implementation, and it must never import one:

* no :mod:`every_query.data.ontology` (no ``build_event_to_query_nodes``, no
  ``expand_events_to_query_nodes``);
* no :mod:`every_query.generate_tasks.query_sequence_labeling` labelers.

Everything is plain Python over a list of ``Event`` tuples.  It is O(n) per query on purpose --
readability is the only thing being optimised, because a subtle oracle is worthless.

The semantics it encodes
------------------------

**Vocabulary split.**  A node is either a *leaf* (a code that really appears in the cohort's
``codes.parquet``) or an *ancestor node* (a name that exists only as some leaf's parent).  This
distinction is load-bearing:

* a **leaf** query means *exactly that code occurred*.  It is NOT widened to its descendants,
  even when the leaf's name happens to be a ``//``-prefix of some other leaf.  This is forced by
  the query-universe invariant -- ancestor query slots are drawn from non-leaf nodes only -- so a
  leaf that is also a prefix was never addressable as a subtree query, and widening it can only
  corrupt its ordinary meaning.
* an **ancestor node** query means *any descendant of that node occurred*.

**Ancestry** is the transitive closure of two kinds of edge, treated identically once built:

* ``//``-prefix edges: ``A//B//C`` -> ``A//B`` -> ``A``.  The separator is the two-character
  ``//``; a single ``/`` is not a separator, so ``ICD10CM/A04.72`` has no ancestors.
* declared ``parent_codes`` edges from the cohort metadata.

Closure is transitive over *both* kinds and over their mixtures: if ``X`` declares parent ``P``
and ``P`` declares parent ``GRP//G``, then ``GRP//G`` and ``GRP`` are both ancestors of ``X``.
Cycles are tolerated and never make a node its own strict ancestor.

**Duration-bounded query** ``(code, d)`` at prediction time ``t`` is True iff some event whose
code matches ``code`` occurs at a time strictly inside the OPEN interval ``(t, t + d)``.  Both
endpoints are excluded: an event exactly at ``t`` does not count, and neither does one exactly
at ``t + d``.

**Event-bounded query** ``(code, boundary)`` at ``t`` is True iff some matching event occurs
strictly inside ``(t, b)``, where ``b`` is the time of the *first* event matching ``boundary``
strictly after ``t``.  Both ends open again, so an event sharing the boundary's exact instant
does NOT count.  When no boundary event ever occurs after ``t``, the window is left open to the
end of the record -- the query degenerates to "does this code ever occur again".

**Censoring** is not a third label.  It is the answer to the ordinary query
``(TIMELINE//END, d)``: True means the record ends inside the window, i.e. the window is not
fully observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SEP = "//"
EOS_CODE = "TIMELINE//END"


@dataclass(frozen=True)
class Event:
    """One MEDS measurement: who, when, what."""

    subject_id: int
    time: datetime
    code: str


# --------------------------------------------------------------------------------------------
# Ontology
# --------------------------------------------------------------------------------------------


def prefix_parents(code: str) -> list[str]:
    """The single immediate ``//``-prefix parent of ``code``, or ``[]``.

    Written independently of ``every_query.data.ontology.string_ancestors``: that one returns
    *all* proper prefixes, this one returns only the immediate parent and lets the transitive
    closure below do the rest.  Two different shapes computing the same set is exactly the
    redundancy we want.

    >>> prefix_parents("A//B//C")
    ['A//B']
    >>> prefix_parents("A")
    []
    >>> prefix_parents("ICD10CM/A04.72")
    []
    """
    if SEP not in code:
        return []
    return [code.rsplit(SEP, 1)[0]]


class Ontology:
    """An explicit DAG over code names, built from leaves + declared parent edges.

    Args:
        leaves: every code that really occurs in the cohort vocabulary.
        declared_parents: optional ``code -> [parent, ...]`` edges from ``parent_codes``.

    Attributes:
        leaves: the set of real codes.
        nodes: leaves plus every ancestor name reachable from them.
        ancestors: non-leaf nodes -- the only names addressable as subtree queries.
    """

    def __init__(
        self,
        leaves: Iterable[str],
        declared_parents: Mapping[str, Sequence[str]] | None = None,
        subtree_suffix: str | None = "ANY",
    ):
        self.leaves: set[str] = set(leaves)
        self.subtree_suffix = subtree_suffix
        self._declared: dict[str, list[str]] = {k: list(v) for k, v in (declared_parents or {}).items()}

        # Direct-parent map over every name reachable from a leaf, closed to a fixed point so
        # that a declared grouper's own prefixes (and its own declared parents) join the DAG.
        self.direct: dict[str, set[str]] = {}
        frontier = list(self.leaves) + list(self._declared)
        seen: set[str] = set()
        while frontier:
            name = frontier.pop()
            if name in seen:
                continue
            seen.add(name)
            parents = set(prefix_parents(name)) | {p for p in self._declared.get(name, ()) if p and p != name}
            self.direct[name] = parents
            frontier.extend(parents)

        # Dual-role names -- a real code that is also somebody's ancestor -- get a distinct
        # subtree node, so "exactly this code" and "this code or any descendant" stay separate
        # questions.  Written as a rewrite of the direct-parent map: every edge that pointed at
        # the leaf now points at its subtree node, and the leaf itself gains an edge up to it.
        self.subtree_of: dict[str, str] = {}
        if subtree_suffix:
            dual = {p for parents in self.direct.values() for p in parents if p in self.leaves}
            self.subtree_of = {leaf: f"{leaf}{SEP}{subtree_suffix}" for leaf in dual}
            rewritten: dict[str, set[str]] = {}
            for name, parents in self.direct.items():
                rewritten[name] = {self.subtree_of.get(p, p) for p in parents}
            for leaf, sub in self.subtree_of.items():
                rewritten[sub] = set(rewritten[leaf])
                rewritten[leaf] = set(rewritten[leaf]) | {sub}
            self.direct = rewritten

        self.nodes: set[str] = set(self.direct)
        self.ancestors: set[str] = self.nodes - self.leaves

    def strict_ancestors(self, code: str) -> set[str]:
        """Every node strictly above ``code``.  Cycle-safe; never contains ``code`` itself.

        >>> o = Ontology(["A//B//C"])
        >>> sorted(o.strict_ancestors("A//B//C"))
        ['A', 'A//B']

        A declared chain closes transitively.  ``P`` is a real code, so what sits above ``X`` is
        its *subtree node* ``P//ANY`` rather than the leaf ``P`` itself:

        >>> o2 = Ontology(["X", "P"], {"X": ["P"], "P": ["GRP//G"]})
        >>> sorted(o2.strict_ancestors("X"))
        ['GRP', 'GRP//G', 'P//ANY']

        With subtree nodes switched off the leaf is used directly:

        >>> o2b = Ontology(["X", "P"], {"X": ["P"], "P": ["GRP//G"]}, subtree_suffix=None)
        >>> sorted(o2b.strict_ancestors("X"))
        ['GRP', 'GRP//G', 'P']

        A declared cycle terminates and does not make a node its own ancestor:

        >>> o3 = Ontology(["A", "B"], {"A": ["B"], "B": ["A"]}, subtree_suffix=None)
        >>> sorted(o3.strict_ancestors("A"))
        ['B']
        """
        out: set[str] = set()
        frontier = list(self.direct.get(code, ()))
        while frontier:
            n = frontier.pop()
            if n in out or n == code:
                continue
            out.add(n)
            frontier.extend(self.direct.get(n, ()))
        return out

    def descendants_or_self(self, node: str) -> set[str]:
        """Every leaf whose ancestry contains ``node``, plus ``node`` itself when it is a leaf.

        >>> o = Ontology(["A//B//C", "A//B//D", "Z"])
        >>> sorted(o.descendants_or_self("A//B"))
        ['A//B//C', 'A//B//D']
        """
        out = {node} if node in self.leaves else set()
        for leaf in self.leaves:
            if node in self.strict_ancestors(leaf):
                out.add(leaf)
        return out

    def matches(self, query: str, event_code: str) -> bool:
        """Does an event with ``event_code`` answer a query for ``query``?

        A **leaf** query is exact.  An **ancestor** query is satisfied by any descendant.

        >>> o = Ontology(["A//B", "A//B//C"])
        >>> o.matches("A//B", "A//B//C")   # leaf query stays narrow
        False
        >>> o.matches("A//B", "A//B")
        True
        >>> o.matches("A", "A//B//C")      # `A` is a pure ancestor node
        True

        ``A//B`` is dual-role, so its subtree meaning lives under a separate name and covers
        both the leaf itself and its descendants:

        >>> o.matches("A//B//ANY", "A//B//C"), o.matches("A//B//ANY", "A//B")
        (True, True)
        """
        if query in self.leaves:
            return event_code == query
        return query == event_code or query in self.strict_ancestors(event_code)


# --------------------------------------------------------------------------------------------
# Labeling
# --------------------------------------------------------------------------------------------


def subject_events(events: Sequence[Event], subject_id: int) -> list[Event]:
    """This subject's events, oldest first.

    Ties keep input order (never consulted below).
    """
    return sorted((e for e in events if e.subject_id == subject_id), key=lambda e: e.time)


def occurs_in_open_interval(
    events: Sequence[Event],
    subject_id: int,
    onto: Ontology,
    query: str,
    lo: datetime,
    hi: datetime | None,
) -> bool:
    """``True`` iff a matching event lands strictly inside ``(lo, hi)``.

    ``hi=None`` means "no upper bound" -- the window runs to the end of the record.
    """
    for e in subject_events(events, subject_id):
        if e.time <= lo:
            continue
        if hi is not None and e.time >= hi:
            continue
        if onto.matches(query, e.code):
            return True
    return False


def label_duration(
    events: Sequence[Event],
    subject_id: int,
    onto: Ontology,
    t: datetime,
    query: str,
    duration_days: float,
) -> bool:
    """Duration-bounded occurrence over the OPEN interval ``(t, t + duration_days)``."""
    return occurs_in_open_interval(events, subject_id, onto, query, t, t + timedelta(days=duration_days))


def first_boundary_after(
    events: Sequence[Event],
    subject_id: int,
    onto: Ontology,
    t: datetime,
    boundary: str,
) -> datetime | None:
    """Time of the first event matching ``boundary`` strictly after ``t``; ``None`` if never."""
    for e in subject_events(events, subject_id):
        if e.time > t and onto.matches(boundary, e.code):
            return e.time
    return None


def label_event_bounded(
    events: Sequence[Event],
    subject_id: int,
    onto: Ontology,
    t: datetime,
    query: str,
    boundary: str,
) -> bool:
    """Event-bounded occurrence over the OPEN interval ``(t, b)``.

    ``b`` is the first boundary occurrence strictly after ``t``.  With no such occurrence the
    window is unbounded above (the documented degenerate case).
    """
    b = first_boundary_after(events, subject_id, onto, t, boundary)
    return occurs_in_open_interval(events, subject_id, onto, query, t, b)


def label_censor(
    events: Sequence[Event],
    subject_id: int,
    onto: Ontology,
    t: datetime,
    duration_days: float,
) -> bool:
    """Does the record end inside ``(t, t + d)``?

    The answer to the ``TIMELINE//END`` query.
    """
    return label_duration(events, subject_id, onto, t, EOS_CODE, duration_days)


def record_end(events: Sequence[Event], subject_id: int) -> datetime | None:
    """Time of this subject's ``TIMELINE//END``, or their last event when absent."""
    evs = subject_events(events, subject_id)
    if not evs:
        return None
    for e in evs:
        if e.code == EOS_CODE:
            return e.time
    return evs[-1].time
