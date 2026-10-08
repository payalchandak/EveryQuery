"""Lightning wrapper for :class:`ConditionalMultitaskARModel`."""

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Literal

import torch
from meds import held_out_split, train_split, tuning_split

from every_query.data.multitask_dataset import MultitaskBoundaryBatch
from every_query.data.multitask_eval_dataset import MultitaskEvalBatch
from every_query.model.conditional_multitask_ar_model import ConditionalMultitaskARModel
from every_query.model.lightning_module import EveryQueryLightningModule, _dict_to_factory


class ConditionalMultitaskLightningModule(EveryQueryLightningModule):
    """Train, evaluate and restore the all-vocabulary multitask architecture.

    Each Lightning loop is bound to one dataset, one batch type and one model path (issue #30).
    Nothing here branches on ``self.training``: validation, test and prediction are all eval-mode
    loops, but they carry different contracts, and the loop hook that Lightning calls is what
    selects the path::

        loop             dataset                          batch                   model path
        ---------------  -------------------------------  ----------------------  --------------------------
        training_step    MultitaskBoundaryPytorchDataset  MultitaskBoundaryBatch  forward, dense (B, K, V)
        validation_step  MultitaskBoundaryPytorchDataset  MultitaskBoundaryBatch  forward, dense (B, K, V)
        test_step        QuerySeqMultitaskEvalDataset     MultitaskEvalBatch      score_final_query, (B,)
        predict_step     QuerySeqMultitaskEvalDataset     MultitaskEvalBatch      score_final_query, (B,)

    Training and validation share the dense all-vocabulary objective (masked BCE over ``(B, K, V)``
    targets), so ``tuning/loss`` - what checkpointing and early stopping monitor - keeps its
    meaning.  Test and prediction score one ``QuerySeqSchema`` row each: the row's final query at
    its last real window, through :meth:`ConditionalMultitaskARModel.score_final_query`, which never
    reads ``targets`` (a :class:`MultitaskEvalBatch` has none) and never materializes a
    ``(B, K, V)`` tensor.  ``held_out/loss`` is the scalar BCE of those logits against the rows'
    final-query labels.

    Logged metrics: ``train/loss`` (per step and per epoch), ``tuning/loss`` and ``held_out/loss``
    (per epoch).  Full-vocabulary macro metrics are deliberately deferred, so the parent's epoch
    hooks run over empty metric dictionaries.
    """

    def __init__(
        self,
        model: ConditionalMultitaskARModel,
        optimizer: Callable[[Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer] | None = None,
        LR_scheduler: Callable[..., Any] | None = None,
        warmup_ratio: float = 0.0,
        grad_norm_log_every_n_steps: int = 1000,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
            warmup_ratio=warmup_ratio,
            grad_norm_log_every_n_steps=grad_norm_log_every_n_steps,
        )
        # Full-vocabulary macro metrics are deliberately deferred.  Keep the parent's epoch
        # hooks and optimizer plumbing, but give them no scalar-model metrics to update.
        self.metrics = {train_split: {}, tuning_split: {}, held_out_split: {}}

    def _log_metrics(
        self,
        loss: torch.Tensor,
        batch: MultitaskBoundaryBatch | MultitaskEvalBatch,
        split: Literal[train_split, tuning_split, held_out_split],
    ) -> None:
        """Log ``{split}/loss`` for one batch of either loop family.

        The only per-batch metric is the scalar loss, so unlike the scalar parent's version this
        takes no model ``outputs``: the training loops have a dense
        :class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskOutput` that
        nothing here reads, and the evaluation loops have no output object at all.  Only
        ``batch.batch_size`` is read from the batch.
        """
        is_train = split == train_split
        sync_dist = not is_train and torch.distributed.is_available() and torch.distributed.is_initialized()
        self.log(
            f"{split}/loss",
            loss.item(),
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
            sync_dist=sync_dist,
        )

    def training_step(self, batch: MultitaskBoundaryBatch) -> torch.Tensor:
        """Dense all-vocabulary objective: ``self.model(batch)`` against ``(B, K, V)`` targets.

        Skips the scalar parent's two-head gradient-norm logging, which has no meaning here.
        """
        loss, _ = self.model(batch)
        self._log_metrics(loss, batch, train_split)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: MultitaskBoundaryBatch) -> torch.Tensor:
        """Validation is deliberately training-style: the same dense forward and masked BCE.

        Checkpointing and early stopping monitor ``tuning/loss``, which is the dense multitask
        objective over :class:`MultitaskBoundaryBatch` targets; scoring QuerySeq rows here would
        silently change what that number means.  The QuerySeq path is :meth:`test_step` /
        :meth:`predict_step`.
        """
        loss, _ = self.model(batch)
        self._log_metrics(loss, batch, tuning_split)
        return loss

    def _score_final_query(self, batch: MultitaskEvalBatch) -> torch.Tensor:
        """Float32 logits ``(B,)`` of each row's scored code at that row's last real window.

        Shared by :meth:`test_step` and :meth:`predict_step`.  Goes through
        :meth:`ConditionalMultitaskARModel.score_final_query`, which never reads ``batch.targets``
        (a :class:`MultitaskEvalBatch` has none) and never builds ``(B, K, V)`` logits.
        """
        return self.model.score_final_query(batch, batch.scored_codes)

    @torch.no_grad()
    def test_step(self, batch: MultitaskEvalBatch) -> torch.Tensor:
        """Scalar held-out loss: BCE-with-logits of the final-query logits against ``batch.labels``.

        The mean over the batch's rows (a :class:`MultitaskEvalBatch` always has at least one) is
        logged as ``held_out/loss``.  The dense forward is never called.
        """
        logits = self._score_final_query(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels.float())
        self._log_metrics(loss, batch, held_out_split)
        return loss

    @torch.no_grad()
    def predict_step(self, batch: MultitaskEvalBatch) -> dict[str, torch.Tensor]:
        """One probability per QuerySeq row, with the row's label and scored code for alignment.

        Returns three CPU tensors of shape ``(B,)``, in the batch's row order:

        - ``probs``: float32 ``sigmoid(score_final_query)``;
        - ``labels``: bool ``answers[-1]`` of each row;
        - ``scored_codes``: int64 ``queries[-1]`` of each row.

        These keys are the contract ``EQ_predict_multitask`` consumes.  ``batch.targets`` is never
        touched and the dense forward is never called.
        """
        logits = self._score_final_query(batch)
        return {
            "probs": torch.sigmoid(logits).detach().cpu(),
            "labels": batch.labels.detach().cpu(),
            "scored_codes": batch.scored_codes.detach().cpu(),
        }

    ARCHITECTURES: ClassVar[dict[str, type]] = {
        "conditional_multitask_ar": ConditionalMultitaskARModel,
    }

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str | None = None) -> "ConditionalMultitaskLightningModule":
        """Restore a multitask checkpoint using its mandatory architecture discriminator."""
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        for key in ("model", "optimizer", "LR_scheduler"):
            if key not in hparams:
                raise KeyError(
                    f"Checkpoint does not contain {key} hyperparameters. Got {list(hparams.keys())}"
                )

        model_hparams = dict(hparams["model"]) if isinstance(hparams.get("model"), dict) else None
        if model_hparams is None:
            raise TypeError("Checkpoint model hyperparameters must be a dictionary")
        architecture = model_hparams.pop("architecture")
        if architecture not in cls.ARCHITECTURES:
            raise KeyError(
                f"Checkpoint declares unknown multitask architecture {architecture!r}; "
                f"expected one of {sorted(cls.ARCHITECTURES)}."
            )
        model = cls.ARCHITECTURES[architecture](**model_hparams)
        optimizer = _dict_to_factory(hparams["optimizer"])
        LR_scheduler = _dict_to_factory(hparams["LR_scheduler"])

        return super(EveryQueryLightningModule, cls).load_from_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
            warmup_ratio=hparams.get("warmup_ratio", 0.0),
        )
