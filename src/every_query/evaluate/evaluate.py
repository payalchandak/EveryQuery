"""Consolidated ``EQ_evaluate`` entry point.

Replaces the four-stage eval pipeline (``EQ_gen_eval_index`` → ``EQ_gen_eval_tasks`` →
the old ``EQ_evaluate`` → ``EQ_select_model``) with a single CLI that consumes a
:class:`PredictionSchema` parquet written by ``EQ_predict`` and emits a per-``(query,
duration_days)`` metrics parquet.

The CLI is intentionally narrow: no model instantiation, no trainer loop, no multi-model
orchestration.  Cross-model comparison (pairwise win-rate ranking, what the old
``EQ_select_model`` did) moves to ``paper_experiments/`` per the #82 inventory.

Entry-point rewiring to point ``EQ_evaluate`` at this module happens in a follow-up
consolidation wave after the #82 inventory is signed off; the current ``EQ_evaluate``
console script still points at the legacy ``evaluate/eval.py`` until then.
"""

import logging
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import pyarrow.parquet as pq
from omegaconf import DictConfig

from every_query.evaluate.metrics import compute_metrics
from every_query.predict.schema import PredictionSchema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "evaluate" / "configs")


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Compute metrics on a ``EQ_predict`` output parquet."""
    predictions_parquet = Path(cfg.predictions_parquet)
    metrics_parquet = Path(cfg.metrics_parquet)

    logger.info(f"Loading predictions from {predictions_parquet}")
    table = PredictionSchema.align(pq.read_table(predictions_parquet))
    predictions = pl.from_arrow(table)
    logger.info(
        f"Loaded {predictions.height} predictions across "
        f"{predictions['query'].n_unique()} queries x "
        f"{predictions['duration_days'].n_unique()} durations"
    )

    metrics = compute_metrics(predictions)

    metrics_parquet.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_parquet(metrics_parquet)
    logger.info(f"Wrote {metrics.height} metric rows to {metrics_parquet}")


if __name__ == "__main__":
    main()
