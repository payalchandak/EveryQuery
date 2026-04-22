"""Periodic per-(query, duration_days) AUROC monitoring during training.

Optional callback that runs inference on a pre-built fixed task set, computes
per-task AUROC, and logs one scalar per task to the trainer's logger.  Disabled
cleanly when ``tracking_dir`` is unset or the labels directory is missing — the
training path works identically with or without this callback.

The labels directory is produced once per dataset by ``EQ_generate_tracking_tasks``
and lives at ``<tracking_dir>/labels/`` (with a sibling ``tracking_tasks.parquet``
manifest).  A typical value of ``every_n_steps`` is large (1000s) — a full pass
over the labeled set is much heavier than the fast ``limit_val_batches=100``
validation hook that ``LightningModule`` already runs.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

import lightning as L
import polars as pl  # noqa: TC002 — used at runtime by `_log_metrics`
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from meds_torchdata.extensions.lightning_datamodule import Datamodule

from every_query.evaluate.metrics import compute_metrics
from every_query.predict.predict import _gather_probabilities, _identifiers_from_schema_df

logger = logging.getLogger(__name__)


def _slug_for_metric(query: str, duration_days: float) -> str:
    """Make a WandB-safe metric tag from a (query, duration_days) pair.

    Slashes inside ``query`` (common in MEDS codes like ``ICD//I10``) would
    otherwise create unintended nested groups in WandB.  Collapse any run of
    slashes to a single ``__``.
    """
    safe_query = re.sub(r"/+", "__", query)
    return f"{safe_query}@{int(duration_days)}d"


class PerTaskAurocCallback(L.Callback):
    """Run inference on a fixed labeled task set every ``every_n_steps`` and log per-task AUROC.

    Args:
        tracking_dir: Output dir of ``EQ_generate_tracking_tasks``.  Expected to contain
            a ``labels/`` subdirectory of TaskQuerySchema parquets.  ``None`` (or a
            missing / empty ``labels/`` dir) disables the callback — training proceeds
            unchanged.
        every_n_steps: How often (in optimizer steps) to fire the per-task pass.
            The first fire is at ``trainer.global_step == every_n_steps`` (not at
            step 0 — the model is still random there and the per-task AUROCs are
            uninformative).
        prefix: Logger metric prefix.  Per-task scalars are logged as
            ``{prefix}/{slug}/occurs_auroc`` and ``{prefix}/{slug}/censor_auroc``.
            Aggregate stats (``occurs_auroc_mean``, ``occurs_auroc_median``,
            ``n_groups_with_occurs_auroc``) are logged under ``{prefix}/`` directly.
    """

    def __init__(
        self,
        tracking_dir: str | None = None,
        every_n_steps: int = 2000,
        prefix: str = "tuning/per_task",
    ) -> None:
        super().__init__()
        if every_n_steps <= 0:
            raise ValueError(f"every_n_steps must be > 0 (got {every_n_steps})")
        self.tracking_dir = Path(tracking_dir) if tracking_dir else None
        self.every_n_steps = int(every_n_steps)
        self.prefix = prefix

        # Resolved at ``setup`` — None means the callback is disabled (no labels dir).
        self._labels_dir: Path | None = None
        # Track the last step we fired on, to avoid double-firing across multiple
        # ``on_train_batch_end`` invocations within one optimizer step (gradient
        # accumulation calls the hook once per micro-batch).
        self._last_fired_step: int = -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage != "fit":
            return
        if self.tracking_dir is None:
            logger.info("PerTaskAurocCallback: tracking_dir is unset; per-task AUROC disabled.")
            return
        labels_dir = self.tracking_dir / "labels"
        if not labels_dir.is_dir():
            logger.info(
                "PerTaskAurocCallback: %s does not exist; per-task AUROC disabled. "
                "Generate it via `EQ_generate_tracking_tasks out_dir=%s`.",
                labels_dir,
                self.tracking_dir,
            )
            return
        parquets = list(labels_dir.glob("*.parquet"))
        if not parquets:
            logger.info(
                "PerTaskAurocCallback: %s contains no parquet files; per-task AUROC disabled.",
                labels_dir,
            )
            return
        self._labels_dir = labels_dir
        logger.info(
            "PerTaskAurocCallback: enabled. labels_dir=%s every_n_steps=%d",
            labels_dir,
            self.every_n_steps,
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._labels_dir is None:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.every_n_steps != 0 or step == self._last_fired_step:
            return
        self._last_fired_step = step
        # Only rank 0 runs the pass; logging through ``self.log`` would otherwise
        # double-count under DDP.  ``trainer.is_global_zero`` is False on workers.
        if not trainer.is_global_zero:
            return
        self._run_per_task_pass(trainer, pl_module)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    @rank_zero_only
    def _run_per_task_pass(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        assert self._labels_dir is not None  # checked by caller

        dataloader = self._build_dataloader(trainer)
        if dataloader is None:
            return

        try:
            metrics_df = self._compute_per_task_metrics(pl_module, dataloader)
        except Exception:
            # Don't take down a long training run because a periodic auxiliary metric
            # blew up — log and move on.  The fast validation loop is unaffected.
            logger.exception("PerTaskAurocCallback: per-task pass failed at step %d", trainer.global_step)
            return

        self._log_metrics(pl_module, metrics_df)

    def _build_dataloader(self, trainer: L.Trainer):
        """Spin up a one-off val-shaped dataloader pointed at the labels dir.

        Reuses the live datamodule's batch size / worker count / pin-memory and
        the same dataset class (``EveryQueryPytorchDataset``); only swaps
        ``task_labels_dir``.  Returns ``None`` if the active datamodule isn't a
        compatible MTD ``Datamodule`` (tests, sanity checks).
        """
        live_dm = trainer.datamodule
        if live_dm is None or not hasattr(live_dm, "config") or not hasattr(live_dm, "data_class"):
            logger.info(
                "PerTaskAurocCallback: trainer.datamodule has no MTD config/data_class; "
                "skipping per-task pass."
            )
            return None

        try:
            new_config = dataclasses.replace(live_dm.config, task_labels_dir=str(self._labels_dir))
        except TypeError:
            logger.info(
                "PerTaskAurocCallback: live datamodule config isn't a MEDSTorchDataConfig dataclass; "
                "skipping per-task pass."
            )
            return None

        # Force single-process loading for the auxiliary pass — the live training
        # workers may already be saturated, and this hook is rare so worker
        # startup overhead dominates throughput at typical sizes.
        eval_dm = Datamodule(
            config=new_config,
            data_class=live_dm.data_class,
            batch_size=getattr(live_dm, "batch_size", 32),
            num_workers=0,
            pin_memory=False,
        )
        return eval_dm.val_dataloader()

    def _compute_per_task_metrics(self, pl_module: L.LightningModule, dataloader) -> pl.DataFrame:
        """Run inference on ``dataloader`` and return the per-task metrics frame."""
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        pred_batches: list[dict[str, torch.Tensor]] = []
        try:
            with torch.inference_mode():
                for batch in dataloader:
                    batch = pl_module.transfer_batch_to_device(batch, device, dataloader_idx=0)
                    out = pl_module.predict_step(batch)
                    pred_batches.append({k: v.detach().cpu() for k, v in out.items()})
        finally:
            if was_training:
                pl_module.train()

        probs = _gather_probabilities(pred_batches)
        identifiers = _identifiers_from_schema_df(dataloader.dataset.schema_df)

        if probs.height != identifiers.height:
            raise RuntimeError(
                f"Per-task pass row mismatch: {probs.height} predictions vs "
                f"{identifiers.height} identifiers — MTD invariant violation."
            )

        return compute_metrics(identifiers.hstack(probs))

    def _log_metrics(self, pl_module: L.LightningModule, metrics_df: pl.DataFrame) -> None:
        """Log per-task scalars + a few aggregates through Lightning's logger."""
        if metrics_df.is_empty():
            logger.info("PerTaskAurocCallback: no per-task metric rows to log.")
            return

        for row in metrics_df.iter_rows(named=True):
            slug = _slug_for_metric(row["query"], float(row["duration_days"]))
            for metric_name in ("occurs_auroc", "censor_auroc"):
                value = row.get(metric_name)
                if value is None:
                    continue
                pl_module.log(
                    f"{self.prefix}/{slug}/{metric_name}",
                    float(value),
                    on_step=True,
                    on_epoch=False,
                    rank_zero_only=True,
                    sync_dist=False,
                )

        occurs = metrics_df["occurs_auroc"].drop_nulls()
        if occurs.len() > 0:
            pl_module.log(
                f"{self.prefix}/occurs_auroc_mean",
                float(occurs.mean()),
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )
            pl_module.log(
                f"{self.prefix}/occurs_auroc_median",
                float(occurs.median()),
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )
            pl_module.log(
                f"{self.prefix}/n_groups_with_occurs_auroc",
                float(occurs.len()),
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )
