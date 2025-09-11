from torchmetrics.classification import BinaryAUROC
import hydra
import lightning as L
import copy
import logging
import re
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, Literal
from meds import held_out_split, train_split, tuning_split

import torch
import torch.nn.parameter
from model import EveryQueryModel

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


def _dict_to_factory(d: dict[str, Any] | None) -> partial:
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
    ):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer
        self.LR_scheduler_factory = LR_scheduler

        self.metrics = {
            train_split: {},
            tuning_split: {
                "censor_auc": BinaryAUROC(),
                "occurs_auc": BinaryAUROC(),
            },
            held_out_split: {
                "censor_auc": BinaryAUROC(),
                "occurs_auc": BinaryAUROC(),
            },
            "predict": {
                "censor_auc": BinaryAUROC(),
                "occurs_auc": BinaryAUROC(),
            },
        }

        self.save_hyperparameters(
            {
                "model": getattr(model, "hparams", {}),
                "optimizer": _factory_to_dict(self.optimizer_factory),
                "LR_scheduler": _factory_to_dict(self.LR_scheduler_factory),
            }
        )

    def _update_metric(self, name: str, split: str, **kwargs):
        metric = self.metrics.get(split, {}).get(name)
        if metric is None:
            return
        metric.update(**kwargs)

    def _on_epoch_end(self, split: str):
        for metric_name, metric in self.metrics.get(split, {}).items():
            self.log(f"{split}/{metric_name}", metric.compute(), sync_dist=True)
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

        self.log(f"{split}/loss", loss, on_step=is_train, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=sync_dist)
        if getattr(outputs, "censor_loss", None) is not None:
            self.log(f"{split}/censor_loss", outputs.censor_loss, on_step=is_train, on_epoch=not is_train, batch_size=batch_size, sync_dist=sync_dist)
        if getattr(outputs, "occurs_loss", None) is not None:
            self.log(f"{split}/occurs_loss", outputs.occurs_loss, on_step=is_train, on_epoch=not is_train, batch_size=batch_size, sync_dist=sync_dist)

        if not is_train:
            if getattr(outputs, "censor_logits", None) is not None and getattr(batch, "censor", None) is not None:
                self._update_metric(
                    name="censor_auc",
                    split=split,
                    preds=outputs.censor_logits.squeeze(1).sigmoid(),
                    target=batch.censor.long(),
                )
            if getattr(outputs, "occurs_logits", None) is not None and getattr(batch, "occurs", None) is not None:
                mask = ~batch.censor if hasattr(batch, "censor") else None
                preds = outputs.occurs_logits.squeeze(1).sigmoid()
                target = batch.occurs.long()
                if mask is not None:
                    preds = preds[mask]
                    target = target[mask]
                if preds.numel() > 0 and target.unique().numel() == 2:
                    self._update_metric(name="occurs_auc", split=split, preds=preds, target=target)

    def training_step(self, batch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, train_split)
        return loss

    def validation_step(self, batch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, tuning_split)
        return loss

    def test_step(self, batch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, held_out_split)
        return loss

    def predict_step(self, batch):
        loss, outputs = self.model(batch)
        return outputs

    @staticmethod
    def _is_norm_bias_param(n: str) -> bool:
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
    def optimizer_no_decay_factory(self) -> Callable[[Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer]:
        new_factory = copy.deepcopy(self.optimizer_factory)
        if new_factory is None:
            raise ValueError("Optimizer factory is not set. Cannot configure optimizers.")
        if "weight_decay" in new_factory.keywords:
            new_factory.keywords["weight_decay"] = 0.0
        else:
            logger.warning("No weight decay parameter found in optimizer factory. No changes made.")
        return new_factory

    def configure_optimizers(self):
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
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        for k in ["model", "optimizer", "LR_scheduler"]:
            if k not in hparams:
                raise KeyError(f"Checkpoint does not contain {k} hyperparameters. Got {list(hparams.keys())}")

        model = EveryQueryModel(**hparams["model"]) if isinstance(hparams.get("model"), dict) else EveryQueryModel()
        optimizer = _dict_to_factory(hparams["optimizer"])  # type: ignore[arg-type]
        LR_scheduler = _dict_to_factory(hparams["LR_scheduler"])  # type: ignore[arg-type]

        return super().load_from_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
        )

