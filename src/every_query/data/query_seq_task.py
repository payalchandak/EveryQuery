"""The task identity shared by evaluation-grid filtering and AUROC reporting."""

import polars as pl

from every_query.data.query_seq_dataset import (
    ANSWERS_COL,
    BOUND_EVENTS_COL,
    DURATIONS_COL,
    FORCED_ANSWERS_COL,
    QUERIES_COL,
    START_DURATIONS_COL,
    START_EVENTS_COL,
)

SPEC_KEY = [QUERIES_COL, DURATIONS_COL, START_DURATIONS_COL, START_EVENTS_COL, BOUND_EVENTS_COL]
PRIOR_ANSWERS_COL = "prior_answers"
TASK_KEY = [*SPEC_KEY, FORCED_ANSWERS_COL, PRIOR_ANSWERS_COL]


def _with_prior_answers(rows: pl.DataFrame) -> pl.DataFrame:
    """Append the observed conditioning answers, excluding the final scored answer.

    >>> df = pl.DataFrame({"answers": [[True, False, True], [False]]})
    >>> _with_prior_answers(df)["prior_answers"].to_list()
    [[True, False], []]
    """
    answers = pl.col(ANSWERS_COL)
    return rows.with_columns(answers.list.head(answers.list.len() - 1).alias(PRIOR_ANSWERS_COL))


def normalize_task_columns(rows: pl.DataFrame) -> pl.DataFrame:
    """Fill optional grid columns as the prediction adapter does before task grouping."""
    defaults = {
        START_DURATIONS_COL: pl.lit(0.0, dtype=pl.Float32),
        START_EVENTS_COL: pl.lit(None, dtype=pl.Utf8),
        BOUND_EVENTS_COL: pl.lit(None, dtype=pl.Utf8),
        FORCED_ANSWERS_COL: pl.lit(None, dtype=pl.Boolean),
    }
    missing = [
        pl.col(QUERIES_COL).list.eval(value).alias(name)
        for name, value in defaults.items()
        if name not in rows.columns
    ]
    if missing:
        rows = rows.with_columns(missing)
    # Older shards (and sparse synthetic ones) may infer List(Null) when every element is null.
    # Cast even present columns so keys from different shards can be joined without schema drift.
    return rows.with_columns(
        pl.col(START_DURATIONS_COL).cast(pl.List(pl.Float32)),
        pl.col(START_EVENTS_COL).cast(pl.List(pl.Utf8)),
        pl.col(BOUND_EVENTS_COL).cast(pl.List(pl.Utf8)),
        pl.col(FORCED_ANSWERS_COL).cast(pl.List(pl.Boolean)),
    )
