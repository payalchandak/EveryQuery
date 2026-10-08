"""Adapters that run the **production** pipeline over the golden fixture.

This is the only module in the package allowed to import production code.  :mod:`oracle` and
:mod:`golden` stay clean so the expected answers can never be produced by the code under test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from every_query.data.ontology import build_event_to_query_nodes, build_ontology, expand_events_to_query_nodes
from every_query.generate_tasks.query_sequence_labeling import label_query_sequences

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .oracle import Event

BOUND_COL = "bound_event"
#: The sequence sampler writes this in place of a horizon on event-bounded rows.
EVENT_BOUND_SENTINEL = -1.0


def events_to_frame(events: Sequence[Event]) -> pl.DataFrame:
    """Golden events -> the MEDS-shaped frame the labelers consume."""
    return pl.DataFrame(
        {
            "subject_id": [e.subject_id for e in events],
            "time": [e.time for e in events],
            "code": [e.code for e in events],
        }
    ).with_columns(pl.col("time").cast(pl.Datetime("us")))


def codes_frame(leaves: Sequence[str], declared_parents: dict[str, list[str]]) -> pl.DataFrame:
    """A `codes.parquet`-shaped frame for :func:`every_query.data.ontology.build_ontology`."""
    return pl.DataFrame(
        {
            "code": list(leaves),
            "code/vocab_index": list(range(1, len(leaves) + 1)),
            "parent_codes": [declared_parents.get(c) for c in leaves],
        }
    )


def build_artifacts(leaves: Sequence[str], declared_parents: dict[str, list[str]], decay: float = 0.5):
    """`(nodes_df, mix_df, closure_df)` straight from the production builder."""
    nodes_df, mix_df = build_ontology(codes_frame(leaves, declared_parents), decay=decay)
    return nodes_df, mix_df, build_event_to_query_nodes(nodes_df, mix_df)


def index_frame(rows: Sequence[tuple], *, with_bounds: bool) -> pl.DataFrame:
    """Build the flat Stage-4' index frame.

    Args:
        rows: ``(subject_id, prediction_time, query, duration_days, bound_event)`` tuples.
        with_bounds: emit the ``bound_event`` column, which is what makes
            :func:`label_query_sequences` dispatch to the event-bounded labeler.
    """
    df = pl.DataFrame(
        {
            "_ctx_id": list(range(len(rows))),
            "_position": [0] * len(rows),
            "subject_id": [r[0] for r in rows],
            "prediction_time": [r[1] for r in rows],
            "query": [r[2] for r in rows],
            "duration_days": [float(r[3]) for r in rows],
        }
    ).with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us")),
        pl.col("duration_days").cast(pl.Float32),
        pl.col("_ctx_id").cast(pl.UInt32),
    )
    if with_bounds:
        df = df.with_columns(pl.Series(BOUND_COL, [r[4] for r in rows], dtype=pl.Utf8))
    return df


def label(
    rows: Sequence[tuple],
    events: Sequence[Event],
    *,
    closure_df: pl.DataFrame | None,
    with_bounds: bool,
) -> list[bool]:
    """Run the production labeler and return one answer per row, in input order.

    Every row is given its own ``_ctx_id``, so the output has one single-element list per row
    and the group-by's ``maintain_order=True`` keeps them aligned with ``rows``.
    """
    if not rows:
        return []
    events_df = events_to_frame(events)
    if closure_df is not None:
        events_df = expand_events_to_query_nodes(events_df, closure_df)
    out = label_query_sequences(index_frame(rows, with_bounds=with_bounds), events_df)
    return [a[0] for a in out["answers"].to_list()]
