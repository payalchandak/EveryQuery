import copy
import logging
import re
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, Literal

import hydra
import lightning as L
import torch
import torch.nn.parameter
from meds import held_out_split, train_split, tuning_split
from torchmetrics.classification import BinaryAUROC

from every_query.data.dataset import EveryQueryBatch

from .model import EveryQueryModel, EveryQueryOutput

logger = logging.getLogger(__name__)


def _factory_to_dict(factory: partial | None) -> dict[str, Any] | None:
    """Extracts a sufficient dictionary for reconstructing the optimizer or LR scheduler.

    Args:
        factory: A partial function that creates an optimizer or LR scheduler. This comes from the Hydra
            instantiation's "partial" functionality, which is used to partially initialize optimizers or LR
            schedulers without the need to pass in the parameters or optimizers, respectively.

    Returns:
        A dictionary suitable for storing, logging, and sufficient to reconstruct the given factory function.
        The dictionary will contain the special key "_target_" which contains the full path to the target
        function or module to be called. The rest of the dictionary will contain the keyword arguments passed
        in the partial. If the factory is None, returns None.

    Raises:
        TypeError: If the factory is not a partial function.
        ValueError: If the factory partial has any positional arguments or if it uses the reserved key
            "_target_" as a keyword argument.

    Examples:
        >>> print(_factory_to_dict(None))
        None
        >>> _factory_to_dict(partial(torch.optim.Adam, lr=0.001))
        {'_target_': 'torch.optim.adam.Adam', 'lr': 0.001}
        >>> from transformers import get_cosine_schedule_with_warmup
        >>> _factory_to_dict(partial(get_cosine_schedule_with_warmup, num_warmup_steps=10))
        {'_target_': 'transformers.optimization.get_cosine_schedule_with_warmup', 'num_warmup_steps': 10}

    Errors include type checking and some value checks:

        >>> _factory_to_dict(43)
        Traceback (most recent call last):
            ...
        TypeError: Expected a partial function, got <class 'int'>
        >>> _factory_to_dict(partial(torch.optim.Adam, 0.001))
        Traceback (most recent call last):
            ...
        ValueError: Expected a partial function with no positional arguments. Got (0.001,)
        >>> _factory_to_dict(partial(torch.optim.Adam, lr=0.001, _target_="foo"))
        Traceback (most recent call last):
            ...
        ValueError: Expected a partial function with no _target_ keyword argument. Got _target_=foo
    """
    if factory is None:
        return None

    if not isinstance(factory, partial):
        raise TypeError(f"Expected a partial function, got {type(factory)}")

    if factory.args:
        raise ValueError(f"Expected a partial function with no positional arguments. Got {factory.args}")

    kwargs = factory.keywords.copy()

    if "_target_" in kwargs:
        raise ValueError(
            "Expected a partial function with no _target_ keyword argument. "
            f"Got _target_={kwargs['_target_']}"
        )

    target = f"{factory.func.__module__}.{factory.func.__qualname__}"

    return {"_target_": target, **kwargs}


def _dict_to_factory(d: dict[str, Any] | None) -> partial | None:
    """Reconstructs a partial function from a dictionary.

    This is actually just a wrapper around `hydra.utils.instantiate` that sets the `_partial_` flag to True,
    so that it is clear we can use `_factory_to_dict` and `_dict_to_factory` to round-trip encode-decode the
    partial objects instantiated by `hydra.utils.instantiate`.

    Args:
        d: A dictionary containing the target function or module to be called under the key "_target_". The
            rest of the dictionary should contain the keyword arguments to be passed to the function.

    Returns:
        A partial function that creates an optimizer or LR scheduler.

    Examples:
        >>> d = {'_target_': 'torch.optim.adam.Adam', 'lr': 0.001}
        >>> factory = _dict_to_factory(d)
        >>> print(factory.func)
        <class 'torch.optim.adam.Adam'>
        >>> print(factory.keywords)
        {'lr': 0.001}
        >>> print(_factory_to_dict(factory))
        {'_target_': 'torch.optim.adam.Adam', 'lr': 0.001}
        >>> print(_dict_to_factory(None))
        None
    """

    return None if d is None else hydra.utils.instantiate(d, _partial_=True)


class EveryQueryLightningModule(L.LightningModule):
    """LightningModule for EveryQuery with factory-friendly optimizers and metrics."""

    def __init__(
        self,
        model: EveryQueryModel,
        optimizer: Callable[[Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer] | None = None,
        LR_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler] | None = None,
        grad_norm_log_every_n_steps: int = 1000,
    ):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer
        self.LR_scheduler_factory = LR_scheduler
        # Logging knob only — intentionally kept out of save_hyperparameters so old
        # checkpoints load unchanged and load_from_checkpoint needs no edits.
        self.grad_norm_log_every_n_steps = grad_norm_log_every_n_steps

        self.metrics = {
            train_split: {},
            tuning_split: {
                "censor_auc": BinaryAUROC().cpu(),
                "occurs_auc": BinaryAUROC().cpu(),
            },
            held_out_split: {
                "censor_auc": BinaryAUROC().cpu(),
                "occurs_auc": BinaryAUROC().cpu(),
            },
        }

        self.save_hyperparameters(
            {
                "model": getattr(model, "hparams", {}),
                "optimizer": _factory_to_dict(self.optimizer_factory),
                "LR_scheduler": _factory_to_dict(self.LR_scheduler_factory),
            }
        )

    def setup(self, stage=None):
        # keep metrics on CPU even if model moves to GPU
        for split in (tuning_split, held_out_split, "predict"):
            for m in self.metrics.get(split, {}).values():
                m.to("cpu")

    def _update_metric(self, name: str, split: str, **kwargs):
        metric = self.metrics.get(split, {}).get(name)
        if metric is None:
            return
        safe = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
                if k == "preds":
                    v = v.float()
                elif k == "target":
                    v = v.long()
            safe[k] = v
        metric.update(**safe)

    def _on_epoch_end(self, split: str):
        for metric_name, metric in self.metrics.get(split, {}).items():
            try:
                preds_state = getattr(metric, "preds", None)
                target_state = getattr(metric, "target", None)
                has_state = (
                    isinstance(preds_state, list)
                    and isinstance(target_state, list)
                    and len(preds_state) > 0
                    and len(target_state) > 0
                )

                all_targets = torch.cat([t.flatten() for t in target_state], dim=0)
                has_both_classes = all_targets.numel() >= 1 and all_targets.unique().numel() >= 2

                if has_state and has_both_classes:
                    self.log(f"{split}/{metric_name}", float(metric.compute()), sync_dist=True)

            except Exception:
                pass

            metric.reset()

    def on_validation_epoch_end(self):
        self._on_epoch_end(tuning_split)

    def on_test_epoch_end(self):
        self._on_epoch_end(held_out_split)

    def on_train_epoch_end(self):
        pass

    def _log_metrics(
        self,
        loss: torch.Tensor,
        outputs: EveryQueryOutput,
        batch: EveryQueryBatch,
        split: Literal[train_split, tuning_split, held_out_split],
    ):
        batch_size = batch.batch_size
        is_train = split == train_split
        sync_dist = not is_train and torch.distributed.is_available() and torch.distributed.is_initialized()

        self.log(
            f"{split}/loss",
            loss.item(),
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        if getattr(outputs, "censor_loss", None) is not None:
            self.log(
                f"{split}/censor_loss",
                float(outputs.censor_loss.detach().cpu()),
                on_step=is_train,
                on_epoch=not is_train,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
        if getattr(outputs, "occurs_loss", None) is not None:
            self.log(
                f"{split}/occurs_loss",
                float(outputs.occurs_loss.detach().cpu()),
                on_step=is_train,
                on_epoch=not is_train,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

        if not is_train:
            if (
                getattr(outputs, "censor_logits", None) is not None
                and getattr(batch, "censor", None) is not None
            ):
                self._update_metric(
                    name="censor_auc",
                    split=split,
                    preds=outputs.censor_logits.detach().cpu().squeeze(1).sigmoid().float(),
                    target=batch.censor.detach().cpu().long(),
                )
            if (
                getattr(outputs, "occurs_logits", None) is not None
                and getattr(batch, "occurs", None) is not None
            ):
                mask = (~batch.censor).detach().cpu().bool() if hasattr(batch, "censor") else None
                preds = outputs.occurs_logits.detach().cpu().squeeze(1).sigmoid().float()
                target = batch.occurs.detach().cpu().long()
                if mask is not None:
                    preds = preds[mask]
                    target = target[mask]
                if preds.numel() > 0:
                    self._update_metric(name="occurs_auc", split=split, preds=preds, target=target)

    def _encoder_grad_norm(self, task_loss: torch.Tensor) -> torch.Tensor:
        """Global L2 norm of d(task_loss)/d(encoder), over ``self.model.HF_model`` params only.

        Examples:
            >>> loss, outputs = demo_lightning_module.model(sample_batch)
            >>> n = demo_lightning_module._encoder_grad_norm(outputs.censor_loss)
            >>> n.ndim, bool(n.isfinite()), bool(n >= 0)
            (0, True, True)

            ``torch.autograd.grad`` never writes into ``.grad``, so optimization state
            is untouched:

            >>> all(p.grad is None for p in demo_lightning_module.model.HF_model.parameters())
            True

            The graph is retained, so the total loss can still be backpropagated
            afterwards (as Lightning does after training_step):

            >>> loss.backward()
        """
        # ponytail: reentrant grad-ckpt breaks autograd.grad; pass
        # gradient_checkpointing_kwargs={"use_reentrant": False} in model.py if
        # do_grad_ckpt runs ever need these norms.
        encoder_params = [p for p in self.model.HF_model.parameters() if p.requires_grad]
        # torch.autograd.grad (not .backward()) returns grads as a tuple and never
        # writes into p.grad, so this cannot pollute the optimizer step or trip
        # DDP's reducer (its allreduce hooks only fire on .grad accumulation during
        # .backward(), so these norms are per-rank/local — fine for logging).
        #
        # retain_graph=True: Lightning runs the real backward on the total loss
        # *after* training_step returns, and we call this twice (once per task
        # loss) on the same graph — the graph must survive all of that.
        #
        # allow_unused=True: on an all-censored batch, occurs_loss is the
        # differentiable zero `logits.sum() * 0.0` (see model._get_loss); its graph
        # may not reach every encoder parameter, and unused params come back as
        # None — we skip those (norm contribution 0).
        #
        # create_graph is left False (default): grads are not part of a new graph,
        # so there is no second-order autograd cost or memory retention.
        #
        # Under AMP this differentiates the *unscaled* loss (bypassing Lightning's
        # GradScaler), giving true norms; with "16-mixed" tiny intermediate fp16
        # grads can underflow so norms may slightly under-report, which doesn't
        # matter for comparing the two tasks' relative magnitude.
        grads = torch.autograd.grad(task_loss, encoder_params, retain_graph=True, allow_unused=True)
        # Accumulate the squared norm in float32 regardless of AMP dtype for a
        # numerically stable global norm: sqrt(sum_p ||g_p||^2).
        sq = torch.zeros((), device=task_loss.device)
        for g in grads:
            if g is not None:
                sq = sq + g.detach().float().pow(2).sum()
        return sq.sqrt()

    def training_step(self, batch: EveryQueryBatch) -> torch.Tensor:
        """Forward pass and metric logging for a single training batch."""
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, train_split)
        # self.global_step counts optimizer steps in Lightning, so this fires once
        # every N optimizer steps (including step 0, giving an early baseline).
        if torch.is_grad_enabled() and self.global_step % self.grad_norm_log_every_n_steps == 0:
            occurs_norm = self._encoder_grad_norm(outputs.occurs_loss)
            censor_norm = self._encoder_grad_norm(outputs.censor_loss)
            # The optimizer minimizes w * occurs_loss + (1 - w) * censor_loss, and
            # scalar weights factor out of the norm (||w ∇L|| = w ||∇L||), so the
            # weighted contributions reuse the norms above — no extra autograd.grad.
            w = self.model.occurs_loss_weight
            # >1 means the task pulls harder on the shared encoder than the other.
            scalars = {
                "occurs_encoder_grad_norm": occurs_norm,
                "censor_encoder_grad_norm": censor_norm,
                "encoder_grad_ratio": occurs_norm / (censor_norm + 1e-8),
                "weighted_occurs_encoder_grad_norm": w * occurs_norm,
                "weighted_censor_encoder_grad_norm": (1 - w) * censor_norm,
                "weighted_encoder_grad_ratio": (w * occurs_norm) / ((1 - w) * censor_norm + 1e-8),
            }
            for name, value in scalars.items():
                self.log(
                    f"train/{name}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch.batch_size,
                )
        return loss

    @torch.no_grad()
    def validation_step(self, batch: EveryQueryBatch) -> torch.Tensor:
        """Forward pass and metric logging for a single validation batch (no gradients)."""
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, tuning_split)
        return loss

    @torch.no_grad()
    def test_step(self, batch: EveryQueryBatch) -> torch.Tensor:
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, held_out_split)
        return loss

    @torch.no_grad()
    def predict_step(self, batch: EveryQueryBatch) -> dict[str, torch.Tensor]:
        """Produce prediction outputs (probabilities, embeddings, labels) for one batch.

        ``query`` and ``duration_days`` ride along from the batch so downstream
        consumers that inspect predictions at the batch level don't have to re-derive
        them.  Per-row identifiers (``subject_id`` / ``prediction_time``) are not
        needed here — ``EQ_predict`` labels output rows by indexing the dataset's
        ``schema_df`` in dataloader order, so there's no benefit to round-tripping
        them through a tensor.

        Examples:
            >>> result = demo_lightning_module.predict_step(sample_batch)
            >>> sorted(result.keys())
            ['censor', 'censor_probs', 'duration_days', 'occurs', 'occurs_probs', 'query', 'query_embed']

        ``query_embed`` is one ``hidden_size``-wide row per batch item:

            >>> demo_model_config.hidden_size
            64
            >>> result['query_embed'].shape
            torch.Size([2, 64])

        ``query`` / ``duration_days`` carry the per-row identifiers straight from the
        input batch (one scalar per batch item):

            >>> result['query'].shape, result['duration_days'].shape
            (torch.Size([2]), torch.Size([2]))

        Probabilities come from sigmoid, so they lie in [0, 1]:

            >>> 0.0 <= result['censor_probs'].min().item() <= result['censor_probs'].max().item() <= 1.0
            True
        """
        _, outputs = self.model(batch)

        empty_tensor = torch.tensor([])
        return {
            "occurs_probs": outputs.occurs_probs.detach().cpu(),
            "censor_probs": outputs.censor_probs.detach().cpu(),
            "occurs": batch.occurs.detach().cpu() if batch.occurs is not None else empty_tensor,
            "censor": batch.censor.detach().cpu() if batch.censor is not None else empty_tensor,
            "query": batch.query.detach().cpu() if batch.query is not None else empty_tensor,
            "duration_days": (
                batch.duration_days.detach().cpu() if batch.duration_days is not None else empty_tensor
            ),
            "query_embed": outputs.query_embed.detach().cpu(),
        }

    @staticmethod
    def _is_norm_bias_param(n: str) -> bool:
        """True when *n* is a bias or layer-norm weight (these get zero weight decay).

        Examples:
            >>> EveryQueryLightningModule._is_norm_bias_param("encoder.attention.self.query.bias")
            True
            >>> EveryQueryLightningModule._is_norm_bias_param("encoder.attention.self.query.weight")
            False
            >>> EveryQueryLightningModule._is_norm_bias_param("encoder.layer_norm.weight")
            True
            >>> EveryQueryLightningModule._is_norm_bias_param("encoder.layernorm2.weight")
            True

        CamelCase variant used by HuggingFace models:

            >>> EveryQueryLightningModule._is_norm_bias_param("bert.encoder.layer.0.output.LayerNorm.weight")
            True

        The regex requires a ``layer`` prefix, so ModernBERT-style norm names
        (``attn_norm``, ``mlp_norm``, ``final_norm``, bare ``norm``) are **not** matched:

            >>> EveryQueryLightningModule._is_norm_bias_param("layers.1.attn_norm.weight")
            False
            >>> EveryQueryLightningModule._is_norm_bias_param("final_norm.weight")
            False
        """
        return bool(re.search(r"(bias|layer(_?)norm(\d*)\.weight)", n, re.IGNORECASE))

    def _norm_bias_params(self) -> Iterator[torch.nn.parameter.Parameter]:
        for name, p in self.named_parameters():
            if self._is_norm_bias_param(name):
                yield p

    def _non_norm_bias_params(self) -> Iterator[torch.nn.parameter.Parameter]:
        for name, p in self.named_parameters():
            if not self._is_norm_bias_param(name):
                yield p

    @property
    def weight_decay(self) -> float | None:
        if self.optimizer_factory is None:
            return None
        return _factory_to_dict(self.optimizer_factory).get("weight_decay", None)

    @property
    def optimizer_no_decay_factory(
        self,
    ) -> Callable[[Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer]:
        new_factory = copy.deepcopy(self.optimizer_factory)
        if new_factory is None:
            raise ValueError("Optimizer factory is not set. Cannot configure optimizers.")
        if "weight_decay" in new_factory.keywords:
            new_factory.keywords["weight_decay"] = 0.0
        else:
            logger.warning("No weight decay parameter found in optimizer factory. No changes made.")
        return new_factory

    def configure_optimizers(self):
        """Builds optimizer (and optional LR scheduler) with norm/bias weight-decay separation.

        Returns an optimizer when no LR scheduler factory is set, or a dict with
        ``"optimizer"`` and ``"lr_scheduler"`` keys when one is provided.

        Raises:
            ValueError: If no optimizer factory was provided at init time.

        Examples:
            Without an LR scheduler the return value is just the optimizer:

            >>> result = demo_lightning_module.configure_optimizers()
            >>> isinstance(result, torch.optim.Optimizer)
            True
            >>> len(result.param_groups)
            2
            >>> result.param_groups[1]['weight_decay']
            0.0

            Raises ``ValueError`` when no optimizer factory is set:

            >>> module_no_opt = EveryQueryLightningModule(model=demo_model)
            >>> module_no_opt.configure_optimizers()
            Traceback (most recent call last):
                ...
            ValueError: Optimizer factory is not set. Cannot configure optimizers.

            With an LR scheduler, returns a dict containing optimizer and scheduler config:

            >>> module_with_sched = EveryQueryLightningModule(
            ...     model=demo_model,
            ...     optimizer=partial(torch.optim.AdamW, lr=1e-4),
            ...     LR_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1),
            ... )
            >>> result_sched = module_with_sched.configure_optimizers()
            >>> sorted(result_sched.keys())
            ['lr_scheduler', 'optimizer']
            >>> result_sched['lr_scheduler']['interval']
            'step'

            ``ReduceLROnPlateau`` gets epoch-level monitoring instead:

            >>> module_plateau = EveryQueryLightningModule(
            ...     model=demo_model,
            ...     optimizer=partial(torch.optim.AdamW, lr=1e-4),
            ...     LR_scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau),
            ... )
            >>> result_plateau = module_plateau.configure_optimizers()
            >>> result_plateau['lr_scheduler']['interval']
            'epoch'
            >>> result_plateau['lr_scheduler']['monitor']
            'tuning/loss'
        """
        if self.optimizer_factory is None:
            raise ValueError("Optimizer factory is not set. Cannot configure optimizers.")

        params = [
            {"params": self._non_norm_bias_params(), "weight_decay": self.weight_decay},
            {"params": self._norm_bias_params(), "weight_decay": 0.0},
        ]

        optimizer = self.optimizer_no_decay_factory(params)

        if self.LR_scheduler_factory is None:
            return optimizer

        scheduler = self.LR_scheduler_factory(optimizer)

        LR_config = {
            "scheduler": scheduler,
            "frequency": 1,
        }

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau requires observing stable trends to make a conclusion about LR decay, so an
            # epcoh level interval is more appropriate.

            LR_config["monitor"] = "tuning/loss"
            LR_config["strict"] = True
            LR_config["interval"] = "epoch"
        else:
            # All other schedulers operate at a step level as they do not monitor the loss to make a
            # conclusion about LR decay.

            LR_config["interval"] = "step"

        return {"optimizer": optimizer, "lr_scheduler": LR_config}

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str | None = None) -> "EveryQueryLightningModule":
        """Restore an ``EveryQueryLightningModule`` from a Lightning checkpoint.

        Extracts ``model``, ``optimizer``, and ``LR_scheduler`` hyper-parameters from the
        checkpoint, reconstructs the corresponding objects, then delegates to the base
        ``LightningModule.load_from_checkpoint`` for state-dict loading.

        Examples:
            Round-trip save / load preserves hyper-parameters:

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     ckpt_path = tmpdir + "/test.ckpt"
            ...     torch.save({
            ...         'state_dict': demo_lightning_module.state_dict(),
            ...         'hyper_parameters': dict(demo_lightning_module.hparams),
            ...         'pytorch-lightning_version': L.__version__,
            ...     }, ckpt_path)
            ...     loaded = EveryQueryLightningModule.load_from_checkpoint(ckpt_path)
            ...     loaded.hparams['optimizer'] == demo_lightning_module.hparams['optimizer']
            True
            >>> loaded.hparams['LR_scheduler'] is None
            True
            >>> loaded.hparams['model']['precision']
            '32-true'
            >>> loaded.hparams['model']['do_demo']
            True

        Weights survive the round-trip (state dict is loaded correctly):

            >>> torch.equal(next(loaded.parameters()), next(demo_lightning_module.parameters()))
            True

        A checkpoint missing a required hparam key raises ``KeyError``:

            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     bad_path = tmpdir + "/bad.ckpt"
            ...     torch.save({
            ...         'state_dict': {},
            ...         'hyper_parameters': {'model': {}},
            ...         'pytorch-lightning_version': L.__version__,
            ...     }, bad_path)
            ...     EveryQueryLightningModule.load_from_checkpoint(bad_path)
            Traceback (most recent call last):
                ...
            KeyError: "Checkpoint does not contain optimizer hyperparameters. Got ['model']"
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        for k in ["model", "optimizer", "LR_scheduler"]:
            if k not in hparams:
                raise KeyError(f"Checkpoint does not contain {k} hyperparameters. Got {list(hparams.keys())}")

        model = (
            EveryQueryModel(**hparams["model"])
            if isinstance(hparams.get("model"), dict)
            else EveryQueryModel()
        )
        optimizer = _dict_to_factory(hparams["optimizer"])  # type: ignore[arg-type]
        LR_scheduler = _dict_to_factory(hparams["LR_scheduler"])  # type: ignore[arg-type]

        return super().load_from_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
        )
