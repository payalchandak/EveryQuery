"""Cheap in-training per-task AUROC tracking, as a Lightning ``Callback``.

Scores a fixed, offline-sampled parquet of one positive + one negative row per
``(query, duration_days)`` task (see
``every_query.generate_tasks.sample_task_tracking_pairs``) every validation pass and logs
``tuning/occurs_auroc_macro_sampled``.  Because a task's AUROC equals
``P(score(pos) > score(neg))`` for a random pos/neg pair, the win/tie/loss indicator on that
one pair is an unbiased (high-variance) estimate; macro-averaging across tasks estimates
macro AUROC at ``O(n_tasks)`` forward examples instead of ``O(tuning-split size)``.

Lives as a ``Callback`` (not on the ``LightningModule``) so it wires in via
``trainer.callbacks`` like the other cross-cutting concerns, and so its typed ``__init__``
fails fast on a misconfigured hydra block — nothing here is checkpoint-relevant.
"""

import logging

import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities import move_data_to_device
from meds import tuning_split
from meds_torchdata import MEDSTorchDataConfig
from torch.utils.data import DataLoader

from every_query.data.dataset import EveryQueryBatch, EveryQueryPytorchDataset

logger = logging.getLogger(__name__)


class TaskAurocTrackingCallback(Callback):
    """Log ``tuning/occurs_auroc_macro_sampled`` from a fixed pos/neg pair set each val epoch.

    Always scores the ``tuning`` split — tracking mirrors ``tuning/loss`` checkpointing, so the
    split is intentionally fixed and not configurable (the tracking pairs must be sampled into
    ``{task_labels_dir}/tuning/`` via ``EQ_sample_task_tracking_pairs split=tuning``).

    Args:
        config: an instantiated ``MEDSTorchDataConfig`` whose ``task_labels_dir`` points at
            the output of ``EQ_sample_task_tracking_pairs``.
        batch_size: dataloader batch size for the (small) tracking set.
    """

    def __init__(self, config: MEDSTorchDataConfig, batch_size: int = 256):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self._loader: DataLoader | None = None

    def setup(self, trainer, pl_module, stage=None):
        # Built once — the tracking labels are a fixed, offline-generated file, so there's no
        # need to re-read/re-instantiate it every validation epoch.
        if stage not in ("fit", "validate") or self._loader is not None:
            return

        dataset = EveryQueryPytorchDataset(self.config, split=tuning_split)

        # Out-of-vocab query codes encode to PAD_INDEX (0); distinct OOV tasks would otherwise
        # collide there and corrupt the estimate, so they're skipped at scoring — warn here so
        # a silently-shrunk tracking set is visible.
        queries = dataset.query.to_list() if dataset.query is not None else []
        n_oov = sum(1 for q in queries if q not in dataset.code_to_index)
        if n_oov:
            logger.warning(
                "%d/%d task-tracking rows have out-of-vocab query codes; they are skipped "
                "(untrainable, and would otherwise collide at PAD_INDEX).",
                n_oov,
                len(queries),
            )

        if len(dataset) == 0:
            logger.warning(
                "Task-tracking dataset (tuning split) is empty; %r will not be logged.",
                "tuning/occurs_auroc_macro_sampled",
            )
            return

        self._loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            num_workers=0,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        # Rank 0 only: the tracking set is tiny and identical on every rank, so sharding buys
        # nothing; scoring it N times and all-reducing identical scalars is pure waste. Logging
        # with rank_zero_only=True / sync_dist=False keeps this off the collective path (a
        # naive is_global_zero gate + sync_dist=True would deadlock).
        if self._loader is None or not trainer.is_global_zero:
            return
        self._compute_and_log(pl_module.model, pl_module.device, pl_module.log)

    def _compute_and_log(self, model, device, log_fn):
        """Score the pair set and log the macro win/tie/loss AUROC estimate.

        Precondition: called inside Lightning's eval loop, so ``model`` is already in eval mode
        (``no_grad`` covers the numerics regardless).
        """
        probs_by_task: dict[tuple[int, float], dict[int, float]] = {}
        with torch.no_grad():
            for batch in self._loader:
                batch = move_data_to_device(batch, device)
                _, outputs = model(batch)
                # Squeeze only dim 1 so a trailing size-1 batch doesn't collapse to a 0-d scalar.
                probs = outputs.occurs_logits.detach().cpu().squeeze(1).sigmoid().float().tolist()
                queries = batch.query.detach().cpu().tolist()
                durations = batch.duration_days.detach().cpu().tolist()
                labels = batch.occurs.detach().cpu().tolist()
                for query, duration, label, prob in zip(queries, durations, labels, probs, strict=True):
                    if query == EveryQueryBatch.PAD_INDEX:
                        continue  # OOV/pad — see setup() warning
                    probs_by_task.setdefault((query, duration), {})[int(label)] = prob

        indicators = []
        for task_probs in probs_by_task.values():
            if 1 not in task_probs or 0 not in task_probs:
                continue
            pos_prob, neg_prob = task_probs[1], task_probs[0]
            indicators.append(1.0 if pos_prob > neg_prob else 0.0 if pos_prob < neg_prob else 0.5)

        if not indicators:
            return

        log_fn(
            "tuning/occurs_auroc_macro_sampled",
            sum(indicators) / len(indicators),
            rank_zero_only=True,
            sync_dist=False,
        )
        log_fn(
            "tuning/occurs_auroc_macro_sampled_n_tasks",
            float(len(indicators)),
            rank_zero_only=True,
            sync_dist=False,
        )
