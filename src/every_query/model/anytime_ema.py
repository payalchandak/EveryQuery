"""Anytime Pretraining weight averaging: a growing-timescale EMA of the trainable weights (issue #315).

Pair it with a horizon-free LR schedule (warmup then constant, or warmup then ``1/sqrt(t)``; see
``configs/lr_scheduler/``) and every validation point is a usable stopping point: no terminal
horizon is written into the schedule, and the averaged weights stand in for the cooldown a cosine
schedule would otherwise supply.

At optimizer step ``t`` the EMA decays with ``beta_t = 2 ** (-k / t)``, so its half-life is ``t / k``
steps and grows with the run.  ``k = 25`` ("EMA-25") is the published default.

Two weight sets coexist:

- the **raw** weights, which the optimizer updates.  They stay in the module's ``state_dict`` so a
  resumed run continues the exact optimization trajectory;
- the **EMA** weights, which validation and test score (they are swapped in for the loop and back
  out afterwards) and which :func:`apply_ema_weights` installs for prediction.  They are
  saved as this callback's checkpoint state.

Because validation scores the EMA weights, ``tuning/loss`` and everything that monitors it (best
checkpoint selection, early stopping) refer to the EMA model.  ``ModelCheckpoint`` always runs after
this callback (Lightning moves checkpoint callbacks to the end), so the checkpoint it writes from
``on_validation_end`` holds the restored raw weights plus the EMA as callback state.

Only strategies in which every rank holds a full parameter replica (single device, DDP) are
supported; sharded strategies (FSDP, DeepSpeed, ModelParallel) raise.

The EMA is always kept in fp32, whatever the parameters' dtype: under ``bf16-true`` the late-run
update weight ``~k / t`` is far below one bf16 rounding step, so a bf16 EMA would freeze.
Validation briefly holds one more copy of the trainable weights (the raw ones it swaps out).
Prediction does not swap; :func:`apply_ema_weights` installs the EMA for it instead.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import Callback
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy, ModelParallelStrategy

if TYPE_CHECKING:
    import lightning as L

logger = logging.getLogger(__name__)

_SHARDED_STRATEGIES = (FSDPStrategy, DeepSpeedStrategy, ModelParallelStrategy)


def ema_decay(step: int, k: float) -> float:
    """``1 - beta_t`` for ``beta_t = 2 ** (-k / step)``: the weight the newest raw weights receive.

    Computed via ``expm1`` so the late-run weights (``k / step`` near zero) keep full precision
    instead of rounding ``1 - beta`` to zero.

    Examples:
        At step 1 the EMA is (all but) a copy of the weights:

        >>> ema_decay(1, 25.0)
        0.99999997...

        After ``t`` steps the half-life is ``t / k``, so at step 25,000 with ``k=25`` the EMA
        halves every 1,000 steps and each update blends in ``1 - 2 ** (-1 / 1000)`` of the weights:

        >>> round(ema_decay(25_000, 25.0), 8)
        0.00069291
        >>> math.isclose(ema_decay(25_000, 25.0), 1 - 2 ** (-1 / 1000))
        True

        ``step`` must be a positive optimizer step:

        >>> ema_decay(0, 25.0)
        Traceback (most recent call last):
            ...
        ValueError: step must be positive, got 0
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return -math.expm1(-k * math.log(2.0) / step)


def _trainable_named_params(pl_module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {n: p for n, p in pl_module.named_parameters() if p.requires_grad}


class AnytimeEMA(Callback):
    """Maintain the Anytime Pretraining EMA-``k`` of a module's trainable parameters.

    Args:
        k: Growth rate of the averaging horizon: the half-life at optimizer step ``t`` is ``t / k``.
            Must be positive.  ``25`` reproduces the paper's EMA-25.

    The EMA updates once per optimizer step (keyed off ``trainer.global_step``), so gradient
    accumulation does not multiply updates.  It is initialized from the module's weights at the
    start of training, or restored from the checkpoint on resume.

    Examples:
        >>> AnytimeEMA(k=0)
        Traceback (most recent call last):
            ...
        ValueError: k must be positive, got 0

        A tiny model trained for three optimizer steps (two micro-batches per step): the EMA moves
        once per optimizer step, not once per micro-batch.

        >>> import lightning as L
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> class Tiny(L.LightningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.lin = torch.nn.Linear(2, 1)
        ...     def training_step(self, batch):
        ...         return self.lin(batch[0]).pow(2).mean()
        ...     def configure_optimizers(self):
        ...         return torch.optim.SGD(self.parameters(), lr=0.1)
        >>> _ = L.seed_everything(0, verbose=False)
        >>> ema = AnytimeEMA(k=1.0)
        >>> trainer = L.Trainer(
        ...     max_steps=3, accumulate_grad_batches=2, callbacks=[ema], logger=False,
        ...     enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        ... )
        >>> data = DataLoader(TensorDataset(torch.randn(16, 2)), batch_size=2)
        >>> trainer.fit(Tiny(), data)
        >>> ema.last_step
        3
        >>> sorted(ema.ema)
        ['lin.bias', 'lin.weight']
    """

    def __init__(self, k: float = 25.0):
        super().__init__()
        if not k > 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = float(k)
        self.ema: dict[str, torch.Tensor] = {}
        self.last_step = 0
        self._raw: dict[str, torch.Tensor] = {}
        # (EMA tensors, matching parameters) in one order, bound once per fit in on_train_start.
        self._ema_list: list[torch.Tensor] = []
        self._end_list: list[torch.Tensor] = []  # what the EMA lerps toward, all fp32
        # (fp32 buffer, low-precision parameter) pairs refreshed before each update.
        self._cast_bufs: list[torch.Tensor] = []
        self._cast_params: list[torch.Tensor] = []

    # ------------------------------------------------------------------ lifecycle

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if isinstance(trainer.strategy, _SHARDED_STRATEGIES):
            raise NotImplementedError(
                f"AnytimeEMA keeps a full parameter replica and does not support the sharded "
                f"strategy {type(trainer.strategy).__name__}."
            )

    @torch.no_grad()
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        params = _trainable_named_params(pl_module)
        if not self.ema:
            # Fresh run.  (A Lightning resume from a checkpoint without EMA state would also land
            # here and start a new averaging phase from the current weights; train.py's resume check
            # refuses that config change, so it only arises outside EQ_train.)
            self.ema = {n: p.detach().to(torch.float32, copy=True) for n, p in params.items()}
            self.last_step = trainer.global_step
            logger.info("AnytimeEMA: initialized EMA-%g from the weights at step %d.", self.k, self.last_step)
            self._bind(params)
            return

        self._check_matches(params)
        # Checkpoint state may have been restored onto another device.
        self.ema = {n: t.to(device=params[n].device, dtype=torch.float32) for n, t in self.ema.items()}
        self._bind(params)

    def _check_matches(self, params: dict[str, torch.nn.Parameter]) -> None:
        if set(self.ema) != set(params):
            missing = sorted(set(params) - set(self.ema))
            unexpected = sorted(set(self.ema) - set(params))
            raise RuntimeError(
                "AnytimeEMA state restored from the checkpoint does not match the module's trainable "
                f"parameters (missing={missing[:5]}, unexpected={unexpected[:5]})."
            )

    def _bind(self, params: dict[str, torch.nn.Parameter]) -> None:
        """Fix the per-step tensor lists once per fit; the parameter set cannot change mid-fit."""
        self._ema_list, self._end_list, self._cast_bufs, self._cast_params = [], [], [], []
        for n, ema in self.ema.items():
            p = params[n].detach()
            self._ema_list.append(ema)
            if p.dtype == torch.float32:
                self._end_list.append(p)
            else:  # bf16-true etc.: lerp toward a reused fp32 buffer, not a fresh copy per step
                buf = torch.empty_like(p, dtype=torch.float32)
                self._end_list.append(buf)
                self._cast_bufs.append(buf)
                self._cast_params.append(p)

    @torch.no_grad()
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args: Any) -> None:
        step = trainer.global_step
        if step <= self.last_step:
            return  # mid-accumulation micro-batch, or no optimizer step happened
        self.last_step = step

        if self._cast_bufs:
            torch._foreach_copy_(self._cast_bufs, self._cast_params)
        torch._foreach_lerp_(self._ema_list, self._end_list, ema_decay(step, self.k))

    # ------------------------------------------------------------- weight swapping

    @torch.no_grad()
    def _swap_in(self, pl_module: L.LightningModule) -> None:
        if not self.ema or self._raw:
            return  # nothing to average yet (e.g. pre-fit validation), or already swapped
        # On resume the restored EMA reaches here (sanity-check validation) before on_train_start
        # validates it, so check it here too rather than fail midway with a bare KeyError.
        self._check_matches(_trainable_named_params(pl_module))
        params = dict(pl_module.named_parameters())
        for n, ema in self.ema.items():
            p = params[n]
            self._raw[n] = p.detach().clone()
            p.copy_(ema)

    @torch.no_grad()
    def _swap_out(self, pl_module: L.LightningModule) -> None:
        if not self._raw:
            return
        params = dict(pl_module.named_parameters())
        for n, raw in self._raw.items():
            params[n].copy_(raw)
        self._raw = {}

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_in(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_out(pl_module)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_in(pl_module)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_out(pl_module)

    def on_exception(
        self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException
    ) -> None:
        # Never leave EMA weights where the optimizer (or an emergency checkpoint) expects raw ones.
        self._swap_out(pl_module)

    # ------------------------------------------------------------------ checkpoint

    def on_save_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        # ModelCheckpoint always runs after the swap-out, but any other save made mid-validation
        # (a hook's trainer.save_checkpoint, SLURM auto-requeue) would capture the EMA as the model
        # weights and lose the raw ones.  Put the raw weights back into what gets written.
        for n, raw in self._raw.items():
            checkpoint["state_dict"][n] = raw.detach().clone()

    def state_dict(self) -> dict[str, Any]:
        return {"k": self.k, "last_step": self.last_step, "ema": self.ema}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_k = state_dict.get("k", self.k)
        if saved_k != self.k:
            logger.warning(
                "AnytimeEMA: checkpoint was averaged with k=%g, continuing with k=%g.", saved_k, self.k
            )
        self.ema = dict(state_dict["ema"])
        self.last_step = int(state_dict["last_step"])


def ema_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    """The :class:`AnytimeEMA` weights stored in a loaded Lightning checkpoint, or ``None``.

    Examples:
        >>> ema_state_from_checkpoint({"callbacks": {"ModelCheckpoint{...}": {"best_model_path": ""}}})
        >>> w = torch.zeros(2)
        >>> state = {"k": 25.0, "last_step": 3, "ema": {"w": w}}
        >>> ema_state_from_checkpoint({"callbacks": {"AnytimeEMA": state}})
        {'w': tensor([0., 0.])}
        >>> ema_state_from_checkpoint({}) is None
        True
    """
    for key, state in (checkpoint.get("callbacks") or {}).items():
        if key.split("{", 1)[0] == AnytimeEMA.__qualname__ and isinstance(state, dict) and state.get("ema"):
            return state["ema"]
    return None


@torch.no_grad()
def apply_ema_weights(module: torch.nn.Module, ema: dict[str, torch.Tensor]) -> None:
    """Overwrite ``module``'s parameters with the EMA weights ``ema`` (keyed by parameter name).

    Every EMA entry must name a parameter of ``module``; parameters without an entry (frozen ones,
    which the EMA does not track) are left as they are.

    Examples:
        >>> lin = torch.nn.Linear(2, 1)
        >>> apply_ema_weights(lin, {"bias": torch.tensor([7.0])})
        >>> lin.bias
        Parameter containing:
        tensor([7.], requires_grad=True)
        >>> apply_ema_weights(lin, {"nope": torch.tensor([7.0])})
        Traceback (most recent call last):
            ...
        KeyError: "EMA weights name parameters the module does not have: ['nope']"
    """
    params = dict(module.named_parameters())
    unknown = sorted(set(ema) - set(params))
    if unknown:
        raise KeyError(f"EMA weights name parameters the module does not have: {unknown[:5]}")
    for n, t in ema.items():
        params[n].copy_(t)
