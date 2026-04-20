"""Metrics computation over :class:`PredictionSchema`-conformant parquets.

Pure-numpy pipeline — no Lightning trainer, no datamodule.  Consumes what ``EQ_predict``
writes (a parquet of ``(subject_id, prediction_time, query, duration_days, boolean_value,
censor_prob, occurs_prob)``) and produces a per-``(query, duration_days)`` metrics table.

Computes two AUROCs per group, one per model head:

- ``occurs_auroc`` — ``occurs_prob`` vs. ``boolean_value``, the headline task metric.
  Censored rows (``boolean_value`` is null) are dropped before the AUROC computation
  since the label is undefined there and ``occurs_prob`` is loss-masked during training.
- ``censor_auroc`` — ``censor_prob`` vs. ``is_censored = boolean_value.is_null()``, a
  sanity check that the censor head agrees with the data's own censoring indicator.

Extends naturally to other metrics as the paper's needs solidify (calibration, Brier
score, etc.) — one-line additions in :func:`_metrics_for_group`.
"""

import logging

import polars as pl
from sklearn.metrics import roc_auc_score

from every_query.data.schema import TaskQuerySchema
from every_query.predict.schema import PredictionSchema

logger = logging.getLogger(__name__)


def _auroc_or_none(y_true: list[bool], y_score: list[float]) -> float | None:
    """Return AUROC, or ``None`` if the label is single-class (AUROC undefined)."""
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _metrics_for_group(group_df: pl.DataFrame) -> dict[str, float | int | None]:
    """Compute metrics for one ``(query, duration_days)`` slice of predictions.

    Input frame carries at least ``boolean_value``, ``censor_prob``, ``occurs_prob`` for a
    single ``(query, duration_days)`` pair.

    Returns a dict with:

    - ``n_rows`` — total rows in the group.
    - ``n_occurs_labeled`` — rows with non-null ``boolean_value`` (i.e. not censored).
    - ``n_positive`` — rows where ``boolean_value`` is ``True``.
    - ``occurs_auroc`` — AUROC of ``occurs_prob`` vs. ``boolean_value`` over not-censored rows
      (``None`` if all labels are the same class, or if every row is censored).
    - ``censor_auroc`` — AUROC of ``censor_prob`` vs. ``is_censored`` over all rows
      (``None`` if every row has the same censor status).
    """
    is_censored = group_df[TaskQuerySchema.boolean_value_name].is_null().to_list()
    censor_prob = group_df[PredictionSchema.censor_prob_name].to_list()
    censor_auroc = _auroc_or_none(is_censored, censor_prob)

    not_censored = group_df.filter(pl.col(TaskQuerySchema.boolean_value_name).is_not_null())
    if not_censored.is_empty():
        return {
            "n_rows": group_df.height,
            "n_occurs_labeled": 0,
            "n_positive": 0,
            "occurs_auroc": None,
            "censor_auroc": censor_auroc,
        }

    y_true = not_censored[TaskQuerySchema.boolean_value_name].to_list()
    y_score = not_censored[PredictionSchema.occurs_prob_name].to_list()
    return {
        "n_rows": group_df.height,
        "n_occurs_labeled": not_censored.height,
        "n_positive": int(sum(y_true)),
        "occurs_auroc": _auroc_or_none(y_true, y_score),
        "censor_auroc": censor_auroc,
    }


def compute_metrics(predictions: pl.DataFrame) -> pl.DataFrame:
    """Group :class:`PredictionSchema`-shaped rows by ``(query, duration_days)`` and compute metrics.

    Returns a dataframe with columns ``query``, ``duration_days``, ``n_rows``,
    ``n_occurs_labeled``, ``n_positive``, ``occurs_auroc``, ``censor_auroc``.

    Raises:
        ValueError: if ``predictions`` is missing any of the required input columns.

    Examples:
        Two queries over two durations, censored rows mixed in via null
        ``boolean_value``.  Query ``A`` at duration 30 has two labeled rows with
        differing labels (AUROC defined).  Query ``B`` at duration 30 has two labeled
        rows both positive (AUROC undefined, null).  Every row has a censor probability,
        so the censor AUROC is computed across the entire group.

        >>> import polars as pl
        >>> preds = pl.DataFrame({
        ...     "query": ["A", "A", "A", "B", "B", "B"],
        ...     "duration_days": pl.Series(
        ...         [30.0, 30.0, 30.0, 30.0, 30.0, 30.0], dtype=pl.Float32
        ...     ),
        ...     "boolean_value": [True, False, None, True, True, None],
        ...     "censor_prob": [0.05, 0.10, 0.90, 0.02, 0.08, 0.80],
        ...     "occurs_prob": [0.9, 0.1, 0.5, 0.8, 0.7, 0.5],
        ... })
        >>> out = compute_metrics(preds).sort("query")
        >>> for row in out.iter_rows(named=True):
        ...     print(f"{row['query']} n={row['n_rows']} pos={row['n_positive']} "
        ...           f"occurs_auroc={row['occurs_auroc']} censor_auroc={row['censor_auroc']}")
        A n=3 pos=1 occurs_auroc=1.0 censor_auroc=1.0
        B n=3 pos=2 occurs_auroc=None censor_auroc=1.0
        >>> out["duration_days"].dtype
        Float32

        Empty input yields an empty frame with the right shape:

        >>> empty = pl.DataFrame(schema={
        ...     "query": pl.Utf8, "duration_days": pl.Float32,
        ...     "boolean_value": pl.Boolean,
        ...     "censor_prob": pl.Float32, "occurs_prob": pl.Float32,
        ... })
        >>> compute_metrics(empty).shape
        (0, 7)
    """
    required = {
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
        TaskQuerySchema.boolean_value_name,
        PredictionSchema.censor_prob_name,
        PredictionSchema.occurs_prob_name,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"predictions is missing required column(s) {sorted(missing)} — compute_metrics "
            f"needs a PredictionSchema-conformant frame from EQ_predict."
        )

    rows = []
    for (query, duration_days), group_df in predictions.group_by(
        [TaskQuerySchema.query_name, TaskQuerySchema.duration_days_name]
    ):
        metrics = _metrics_for_group(group_df)
        rows.append(
            {
                TaskQuerySchema.query_name: query,
                TaskQuerySchema.duration_days_name: float(duration_days),
                **metrics,
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                TaskQuerySchema.query_name: pl.Utf8,
                TaskQuerySchema.duration_days_name: pl.Float32,
                "n_rows": pl.Int64,
                "n_occurs_labeled": pl.Int64,
                "n_positive": pl.Int64,
                "occurs_auroc": pl.Float64,
                "censor_auroc": pl.Float64,
            }
        )
    # Cast the AUROC columns to Float64 explicitly — when every group has null AUROCs
    # (e.g. all single-class, or every row censored), polars would otherwise infer
    # Null dtype for those columns, producing an unstable on-disk schema where parquet
    # files from different runs have different types for the same logical column.
    return pl.DataFrame(rows).with_columns(
        pl.col(TaskQuerySchema.duration_days_name).cast(pl.Float32),
        pl.col("occurs_auroc").cast(pl.Float64),
        pl.col("censor_auroc").cast(pl.Float64),
    )
