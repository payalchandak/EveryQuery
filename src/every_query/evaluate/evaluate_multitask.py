"""Per-task AUROCs for multitask conditional-query predictions — ``EQ_evaluate_multitask``.

Consumes the one-row-per-grid-row parquet written by ``EQ_predict_multitask`` (``subject_id``,
``prediction_time``, the five window list columns, ``answers``, ``forced_answers``, ``target_code``,
``label``, ``prob``) and writes ``<metrics_stem>.by_task.parquet``: one row per task with its AUROC
and a 95% bootstrap interval.  Nothing else — no macro, no cross-task intervals.

**A task is a unique query sequence under one set of conditioning answers**, :data:`TASK_KEY`: the
five window list columns (what was asked, over which windows) plus ``prior_answers``
(``answers[:-1]``, the teacher-forced answers the final query was conditioned on).  A task is
therefore one conditional question — ``P(A_K | patient, Q_1..Q_K, A_1..A_{K-1} = a)`` for one fixed
``a`` — and every ``(subject_id, prediction_time)`` row that asked it is one prediction.  The class
label is ``label`` (i.e. ``answers[-1]``, the final query's answer) and the score is ``prob``.

A ``K``-query sequence splits into up to ``2 ** (K - 1)`` tasks skewed hard toward all-``False``
(most codes are rare), and AUROC is undefined on a single-class task, so expect null ``auroc`` rows
and read ``n_rows`` / ``n_positive`` next to every estimate.

``forced_answers`` is in the key too.  A designed conditioning answer selects a cohort —
``EQ_generate_evaluation_query_sequences`` writes a forced spec only at the contexts whose truth
agrees with it — so a forced task holds the same rows as the matching ``prior_answers`` task of the
same spec left unforced.  Keying on it keeps the two apart when a grid carries both, instead of
pooling them and counting those rows twice.  A predictions parquet written before the column existed
reads as all-null (nothing forced) and groups exactly as it did.

**The interval is a row bootstrap within the task**: draw the task's rows with replacement,
recompute the AUROC, repeat ``n_resamples`` times, and take the 2.5th / 97.5th percentiles.  Rows
are treated as independent, so when a subject contributes several prediction times to one task
(``n_rows > n_subjects``) the interval is somewhat narrower than a subject-level one would be.
"""

import logging
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from every_query.data.query_seq_dataset import (
    ANSWERS_COL,
    FORCED_ANSWERS_COL,
    QUERIES_COL,
)
from every_query.data.query_seq_task import PRIOR_ANSWERS_COL, SPEC_KEY, TASK_KEY, _with_prior_answers
from every_query.evaluate.metrics import _auroc_or_none

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "evaluate" / "configs")

SUBJECT_ID_COL = "subject_id"
LABEL_COL = "label"
PROB_COL = "prob"

REQUIRED_COLUMNS = [*SPEC_KEY, ANSWERS_COL, SUBJECT_ID_COL, LABEL_COL, PROB_COL]

# Percentile method, 1000 resamples, 95%, seed 0 — the `upstream/task-auroc-ci` convention.
DEFAULT_N_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 0
CONFIDENCE_LEVEL = 0.95

# (upper-exclusive edge in days, bucket label) pairs.  Descriptive only — never a grouping key.
DURATION_BUCKETS = [
    (2, "1d"),
    (8, "2-7d"),
    (31, "8-30d"),
    (91, "31-90d"),
    (181, "91-180d"),
    (366, "181-365d"),
]

EVENT_BOUND_BUCKET = "event-bound"

# The non-key columns of ``by_task``, pinned so an empty or all-null run writes the same schema.
_METRIC_SCHEMA = {
    "target_code": pl.Utf8,
    "n_queries": pl.Int64,
    "duration_bucket": pl.Utf8,
    "n_rows": pl.Int64,
    "n_positive": pl.Int64,
    "prevalence": pl.Float64,
    "n_subjects": pl.Int64,
    "auroc": pl.Float64,
    "auroc_ci_lo": pl.Float64,
    "auroc_ci_hi": pl.Float64,
    "n_degenerate_replicates": pl.Int64,
}


def _duration_bucket(duration_days: float | None) -> str | None:
    """Map the final query's horizon to a human-readable bucket label.

    An **event-bounded** query has no horizon: its window ends at a boundary event and its duration
    is the negative event-bound sentinel.  It gets its own bucket rather than falling through the
    ``<`` ladder, where ``-1.0 < 2`` would file every such row under the shortest horizon and
    quietly mix two different questions into one label.

    Examples:
        >>> [_duration_bucket(d) for d in (0.5, 3.0, 20.0, 400.0, -1.0, None)]
        ['1d', '2-7d', '8-30d', '>365d', 'event-bound', None]
    """
    if duration_days is None:
        return None
    if duration_days < 0:
        return EVENT_BOUND_BUCKET
    for hi, label in DURATION_BUCKETS:
        if duration_days < hi:
            return label
    return ">365d"


def _bootstrap_aurocs(
    y: np.ndarray, score: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """AUROCs of ``n_resamples`` resamples of one task's rows, drawn with replacement.

    AUROC is the Mann-Whitney statistic, so a replicate is fully determined by how many positives
    and negatives it drew at each distinct score -- never by which rows they were.  Resampling the
    rows is therefore one multinomial draw over the (score, label) cells, and every replicate is
    scored at once by cumulative sums, instead of sorting each one through ``roc_auc_score``.
    Tied scores get midranks, which is what ``roc_auc_score``'s trapezoidal curve already does.

    Examples:
        A perfectly separable task scores 1.0 on every resample that holds both classes:

        >>> y, score = np.array([True, True, False, False]), np.array([0.9, 0.8, 0.2, 0.1])
        >>> replicates = _bootstrap_aurocs(y, score, 16, np.random.default_rng(0))
        >>> replicates.shape, set(replicates[~np.isnan(replicates)].tolist())
        ((16,), {1.0})

        Tied scores carrying both labels land on the midrank answer, not on 0.0 or 1.0:

        >>> y, score = np.array([True, False]), np.array([0.5, 0.5])
        >>> r = _bootstrap_aurocs(y, score, 8, np.random.default_rng(0))
        >>> set(r[~np.isnan(r)].tolist())
        {0.5}
    """
    n = y.size
    # inv indexes the ascending distinct scores, so cell counts are already in rank order.
    uniq, inv = np.unique(score, return_inverse=True)
    n_scores = uniq.size
    cells = np.concatenate(
        [np.bincount(inv[y], minlength=n_scores), np.bincount(inv[~y], minlength=n_scores)]
    )
    # ponytail: (n_resamples, 2 * n_scores) held at once -- fine while scores are coarse, as the
    # Monte Carlo probabilities are.  Chunk over replicates if continuous scores ever come through.
    counts = rng.multinomial(n, cells / n, size=n_resamples)
    n_pos_c, n_neg_c = counts[:, :n_scores], counts[:, n_scores:]

    tied = n_pos_c + n_neg_c
    midrank = np.cumsum(tied, axis=1) - tied + (tied + 1.0) / 2.0
    n_pos = n_pos_c.sum(axis=1)
    n_neg = n_neg_c.sum(axis=1)
    u = (n_pos_c * midrank).sum(axis=1) - n_pos * (n_pos + 1.0) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = u / (n_pos * n_neg)
    out[(n_pos == 0) | (n_neg == 0)] = np.nan  # a single-class replicate has no AUROC
    return out


def _percentile_bounds(values: np.ndarray) -> tuple[float | None, float | None]:
    """Two-sided ``CONFIDENCE_LEVEL`` percentile bounds, ignoring degenerate (``nan``) replicates.

    ``(None, None)`` when every replicate was degenerate — an interval resting on nothing is not an
    interval, and a null is easier to notice downstream than a silently wide one.

    Examples:
        >>> lo, hi = _percentile_bounds(np.array([0.0, 0.5, 1.0]))
        >>> round(lo, 6), round(hi, 6)
        (0.025, 0.975)
        >>> _percentile_bounds(np.array([0.4, np.nan, 0.4]))  # degenerate replicates skipped
        (0.4, 0.4)
        >>> _percentile_bounds(np.array([np.nan, np.nan]))
        (None, None)
    """
    if values.size == 0 or bool(np.isnan(values).all()):
        return None, None
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0 * 100.0
    lo, hi = np.nanpercentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def _validate_columns(predictions: pl.DataFrame) -> None:
    """Fail with the missing column names rather than a ``KeyError`` deep in the group loop."""
    missing = [c for c in REQUIRED_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(
            f"predictions is missing required column(s) {missing} — EQ_evaluate_multitask needs the "
            f"parquet written by EQ_predict_multitask."
        )
    if predictions[LABEL_COL].null_count():
        raise ValueError(
            f"{predictions[LABEL_COL].null_count()} row(s) have a null {LABEL_COL!r}. The evaluation "
            "grid's answers are binary and never null (censoring is carried by an explicit "
            "TIMELINE//END query), so a null label means the grid or the prediction run is malformed."
        )


def compute_multitask_metrics(
    predictions: pl.DataFrame,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pl.DataFrame:
    """One row per task: its key, counts, AUROC and 95% row-bootstrap interval.

    Args:
        predictions: One row per evaluation-grid row; see the module docstring for the schema.
        n_resamples: Bootstrap replicates per task.
        bootstrap_seed: Seed for the resampling, so a fixed prediction frame gives fixed intervals.

    Returns:
        ``by_task``, sorted by :data:`TASK_KEY`.  ``auroc`` and its interval are null for a
        single-class task.

    Raises:
        ValueError: if a required column is missing, a label is null, or ``n_resamples < 1``.

    Examples:
        Two tasks over four subjects.  ``A`` is perfectly separable; ``B`` is all-``True`` and so
        unscorable, and carries a null interval rather than a fabricated one.

        >>> labels = [True, True, False, False, True, True, True, True]
        >>> preds = pl.DataFrame({
        ...     "subject_id": [1, 2, 3, 4, 1, 2, 3, 4],
        ...     "queries": [["A"]] * 4 + [["B"]] * 4,
        ...     "durations": [[30.0]] * 8,
        ...     "start_durations": [[0.0]] * 8,
        ...     "start_events": [[None]] * 8,
        ...     "bound_events": [[None]] * 8,
        ...     "answers": [[a] for a in labels],
        ...     "label": labels,
        ...     "prob": [0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.5, 0.4],
        ... })
        >>> by_task = compute_multitask_metrics(preds, n_resamples=32)
        >>> by_task.select("target_code", "prior_answers", "n_rows", "auroc").rows()
        [('A', [], 4, 1.0), ('B', [], 4, None)]
        >>> by_task.select("auroc_ci_lo", "auroc_ci_hi").rows()
        [(1.0, 1.0), (None, None)]
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    _validate_columns(predictions)
    if FORCED_ANSWERS_COL not in predictions.columns:
        nothing_forced = pl.col(QUERIES_COL).list.eval(pl.lit(None, dtype=pl.Boolean))
        predictions = predictions.with_columns(nothing_forced.alias(FORCED_ANSWERS_COL))

    # A canonical row order, so that neither the task order nor which rows a resample index picks
    # depends on the order the predictions happened to be written in.
    predictions = _with_prior_answers(predictions).sort([*TASK_KEY, PROB_COL, LABEL_COL])
    rng = np.random.default_rng(bootstrap_seed)

    key_frames, rows = [], []
    for key, group in predictions.group_by(TASK_KEY, maintain_order=True):
        queries, durations = key[0], key[1]
        y = group[LABEL_COL].to_numpy().astype(bool)
        score = group[PROB_COL].to_numpy().astype(np.float64)

        auroc = _auroc_or_none(y.tolist(), score.tolist())
        lo = hi = n_degenerate = None
        if auroc is not None:
            replicates = _bootstrap_aurocs(y, score, n_resamples, rng)
            lo, hi = _percentile_bounds(replicates)
            n_degenerate = int(np.isnan(replicates).sum())

        # Carried from the group itself, not rebuilt from python values, so the key's list dtypes
        # survive to parquet unchanged and can never drift out of alignment with the metrics.
        key_frames.append(group.select(TASK_KEY).head(1))
        rows.append(
            {
                "target_code": queries[-1] if queries else None,
                "n_queries": len(queries),
                "duration_bucket": _duration_bucket(durations[-1] if durations else None),
                "n_rows": group.height,
                "n_positive": int(y.sum()),
                "prevalence": float(y.mean()),
                "n_subjects": group[SUBJECT_ID_COL].n_unique(),
                "auroc": auroc,
                "auroc_ci_lo": lo,
                "auroc_ci_hi": hi,
                "n_degenerate_replicates": n_degenerate,
            }
        )

    if key_frames:
        keys = pl.concat(key_frames)
    else:
        # Spelled out rather than `.clear()`ed: on an empty frame polars (1.40) types the derived
        # `prior_answers` as List(Null), which would write a different parquet schema.
        schema = {c: predictions.schema[c] for c in TASK_KEY[:-1]}
        keys = pl.DataFrame(schema={**schema, PRIOR_ANSWERS_COL: predictions.schema[ANSWERS_COL]})
    return keys.hstack(pl.DataFrame(rows, schema=_METRIC_SCHEMA))


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="evaluate_multitask")
def main(cfg: DictConfig) -> None:
    predictions_parquet = Path(cfg.predictions_parquet)
    out_fp = Path(cfg.metrics_stem).with_suffix(".by_task.parquet")

    predictions = pl.read_parquet(predictions_parquet)
    logger.info(f"Loaded {predictions.height} grid-row predictions from {predictions_parquet}")

    by_task = compute_multitask_metrics(
        predictions,
        n_resamples=int(cfg.n_resamples),
        bootstrap_seed=int(cfg.bootstrap_seed),
    )

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    by_task.write_parquet(out_fp)
    n_null = by_task["auroc"].null_count()
    logger.info(
        f"Wrote {by_task.height} task(s) to {out_fp}: {by_task.height - n_null} scored, {n_null} "
        f"single-class (null AUROC); 95% intervals from {cfg.n_resamples} row resamples per task, "
        f"seed {cfg.bootstrap_seed}"
    )


if __name__ == "__main__":
    main()
