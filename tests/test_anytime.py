"""Anytime Pretraining (issue #315): the growing-timescale EMA callback and horizon-free LR schedules."""

import copy
import math
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)

from every_query.model.anytime_ema import (
    AnytimeEMA,
    apply_ema_weights,
    ema_decay,
    ema_state_from_checkpoint,
)
from every_query.model.lightning_module import EveryQueryLightningModule

# --------------------------------------------------------------------------- fixtures


class _Tiny(L.LightningModule):
    """Deterministic two-parameter regression model with a frozen parameter and validation loop."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        # (weights validation scored, the AnytimeEMA's weights at that moment), per validation batch.
        self.val_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def training_step(self, batch):
        x, y = batch
        return (self.lin(x) - y).pow(2).mean()

    def validation_step(self, batch):
        ema = next((cb for cb in self.trainer.callbacks if isinstance(cb, AnytimeEMA)), None)
        if ema is not None and ema.ema:
            self.val_pairs.append((self.lin.weight.detach().clone(), ema.ema["lin.weight"].clone()))
        x, y = batch
        loss = (self.lin(x) - y).pow(2).mean()
        self.log("tuning/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05)


def _data(n: int = 32, batch_size: int = 2) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, generator=g)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.1 * torch.randn(n, 1, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _trainer(tmp_path: Path, **kwargs) -> L.Trainer:
    defaults = {
        "default_root_dir": tmp_path,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "accelerator": "cpu",
        "deterministic": True,
    }
    defaults.update(kwargs)
    return L.Trainer(**defaults)


def _reference_ema(model: _Tiny, loader: DataLoader, steps: int, accumulate: int, k: float):
    """The issue's minimal non-Lightning recipe, with gradient accumulation, as an oracle."""
    model = copy.deepcopy(model)
    opt = model.configure_optimizers()
    ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    step = 0
    batches = iter(loader)
    while step < steps:
        for _ in range(accumulate):
            (model.training_step(next(batches)) / accumulate).backward()
        opt.step()
        opt.zero_grad()
        step += 1
        beta = 2 ** (-k / step)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    ema[n].lerp_(p, 1 - beta)
    return ema, {n: p.detach().clone() for n, p in model.named_parameters()}


# --------------------------------------------------------------------------- EMA math


@pytest.mark.parametrize("accumulate", [1, 3])
def test_ema_matches_reference_recipe(tmp_path, accumulate):
    """One EMA update per optimizer step, with the paper's beta_t = 2 ** (-k / t)."""
    L.seed_everything(0, verbose=False)
    model = _Tiny()
    ref_ema, ref_raw = _reference_ema(model, _data(), steps=5, accumulate=accumulate, k=2.0)

    cb = AnytimeEMA(k=2.0)
    _trainer(
        tmp_path, max_steps=5, accumulate_grad_batches=accumulate, callbacks=[cb], limit_val_batches=0
    ).fit(model, _data())

    assert cb.last_step == 5
    assert set(cb.ema) == {"lin.weight", "lin.bias"}, "frozen parameters must not be averaged"
    for n in ref_ema:
        torch.testing.assert_close(cb.ema[n], ref_ema[n])
    # The module itself is left on the raw optimizer trajectory.
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.detach(), ref_raw[n])
    # The EMA genuinely lags the raw weights (it is not a copy).
    assert not torch.allclose(cb.ema["lin.weight"], model.lin.weight)


def test_ema_decay_half_life():
    k, t = 25.0, 100_000
    w = ema_decay(t, k)
    half_life = math.log(0.5) / math.log(1 - w)
    assert half_life == pytest.approx(t / k)


# --------------------------------------------------------------------- weight swapping


def test_validation_scores_ema_weights_and_checkpoint_keeps_raw(tmp_path):
    L.seed_everything(0, verbose=False)
    model = _Tiny()
    cb = AnytimeEMA(k=2.0)
    ckpt = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=tmp_path / "ckpts", save_last=True, monitor="tuning/loss"
    )
    trainer = _trainer(
        tmp_path,
        max_steps=6,
        callbacks=[ckpt, cb],  # checkpoint listed first: Lightning must still run it after the swap-out
        enable_checkpointing=True,
        val_check_interval=3,
        limit_val_batches=1,
    )
    trainer.fit(model, _data(), val_dataloaders=_data())

    # Validation ran (at steps 3 and 6) and every time scored the EMA weights.
    assert len(model.val_pairs) >= 2
    for scored, ema in model.val_pairs:
        torch.testing.assert_close(scored, ema)
    # After validation the raw weights are back.
    assert not torch.allclose(model.lin.weight, cb.ema["lin.weight"])

    saved = torch.load(tmp_path / "ckpts" / "last.ckpt", map_location="cpu", weights_only=False)
    torch.testing.assert_close(saved["state_dict"]["lin.weight"], model.lin.weight.detach())
    torch.testing.assert_close(ema_state_from_checkpoint(saved)["lin.weight"], cb.ema["lin.weight"])
    # last.ckpt was written by the step-6 validation, after the swap back to raw weights.
    assert saved["callbacks"]["AnytimeEMA"]["last_step"] == 6


def test_validation_before_training_is_a_no_op(tmp_path):
    """train.py's baseline ``trainer.validate`` runs before any EMA exists."""
    model = _Tiny()
    before = model.lin.weight.detach().clone()
    _trainer(tmp_path, callbacks=[AnytimeEMA()], limit_val_batches=1).validate(model, _data())
    torch.testing.assert_close(model.lin.weight.detach(), before)


def test_exception_during_validation_restores_raw_weights():
    model = _Tiny()
    cb = AnytimeEMA()
    cb.ema = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    raw = model.lin.weight.detach().clone()

    cb.on_validation_start(Mock(), model)
    assert torch.equal(model.lin.weight, torch.zeros_like(raw))
    cb.on_exception(Mock(), model, RuntimeError("boom"))
    torch.testing.assert_close(model.lin.weight.detach(), raw)


def test_sharded_strategies_are_rejected():
    class FSDPStrategy:  # name-alike stand-in; constructing a real one needs a process group
        pass

    trainer = Mock(strategy=FSDPStrategy())
    with pytest.raises(NotImplementedError, match="FSDPStrategy"):
        AnytimeEMA().setup(trainer, Mock(), "fit")


# ------------------------------------------------------------------------------ resume


def test_resume_continues_the_same_ema(tmp_path):
    """One epoch + resume to two gives the same EMA as two uninterrupted epochs.

    Epoch boundaries keep the plain (non-resumable) dataloader on the same batches either way.
    """
    L.seed_everything(0, verbose=False)
    init = _Tiny()
    data = partial(_data, n=6)  # three optimizer steps per epoch

    straight = AnytimeEMA(k=2.0)
    _trainer(tmp_path / "a", max_epochs=2, callbacks=[straight], limit_val_batches=0).fit(
        copy.deepcopy(init), data()
    )

    first = AnytimeEMA(k=2.0)
    ckpt = L.pytorch.callbacks.ModelCheckpoint(dirpath=tmp_path / "b", save_last=True)
    _trainer(
        tmp_path / "b", max_epochs=1, callbacks=[first, ckpt], enable_checkpointing=True, limit_val_batches=0
    ).fit(copy.deepcopy(init), data())
    assert first.last_step == 3

    resumed = AnytimeEMA(k=2.0)
    ckpt2 = L.pytorch.callbacks.ModelCheckpoint(dirpath=tmp_path / "c", save_last=True)
    _trainer(
        tmp_path / "c",
        max_epochs=2,
        callbacks=[resumed, ckpt2],
        enable_checkpointing=True,
        limit_val_batches=0,
    ).fit(copy.deepcopy(init), data(), ckpt_path=tmp_path / "b" / "last.ckpt")

    assert resumed.last_step == 6
    for n in straight.ema:
        torch.testing.assert_close(resumed.ema[n], straight.ema[n])


def test_resume_rejects_mismatched_parameters():
    cb = AnytimeEMA()
    cb.load_state_dict({"k": 25.0, "last_step": 3, "ema": {"other.weight": torch.zeros(1)}})
    with pytest.raises(RuntimeError, match="does not match"):
        cb.on_train_start(Mock(global_step=3), _Tiny())


# ------------------------------------------------------------------ checkpoint helpers


def test_apply_ema_weights_from_checkpoint():
    model = _Tiny()
    ema = {"lin.weight": torch.full_like(model.lin.weight, 3.0), "lin.bias": torch.zeros(1)}
    apply_ema_weights(model, ema_state_from_checkpoint({"callbacks": {"AnytimeEMA": {"ema": ema}}}))
    assert torch.equal(model.lin.weight, torch.full_like(model.lin.weight, 3.0))
    assert torch.equal(model.frozen, torch.ones(2)), "untracked parameters are left alone"


# --------------------------------------------------------------------- LR schedules


def _module(demo_model, scheduler, **kwargs) -> EveryQueryLightningModule:
    return EveryQueryLightningModule(
        model=demo_model, optimizer=partial(torch.optim.AdamW, lr=1e-4), LR_scheduler=scheduler, **kwargs
    )


class _NoHorizon:
    """A trainer stand-in whose horizon must not be read."""

    logger = None

    @property
    def estimated_stepping_batches(self):
        raise AssertionError("a horizon-free schedule read the trainer's horizon")


class TestHorizonFreeSchedules:
    def test_constant_with_fixed_warmup_never_reads_horizon(self, demo_model):
        module = _module(demo_model, partial(get_constant_schedule_with_warmup), warmup_steps=10)
        module.trainer = _NoHorizon()
        lr = module.configure_optimizers()["lr_scheduler"]["scheduler"].lr_lambdas[0]
        assert [lr(s) for s in (0, 5, 10, 10**7)] == [0.0, 0.5, 1.0, 1.0]

    def test_inverse_sqrt_decays_without_a_horizon(self, demo_model):
        module = _module(demo_model, partial(get_inverse_sqrt_schedule, timescale=None), warmup_steps=10)
        module.trainer = _NoHorizon()
        lr = module.configure_optimizers()["lr_scheduler"]["scheduler"].lr_lambdas[0]
        assert lr(5) == pytest.approx(0.5)
        assert lr(10) == pytest.approx(1.0)
        assert lr(10 + 30) == pytest.approx(0.5)  # timescale = warmup = 10 -> 1/sqrt(40 / 10)

    def test_horizon_free_with_ratio_still_sizes_warmup_from_horizon(self, demo_model):
        module = _module(demo_model, partial(get_constant_schedule_with_warmup), warmup_ratio=0.1)
        module.trainer = Mock(estimated_stepping_batches=100)
        lr = module.configure_optimizers()["lr_scheduler"]["scheduler"].lr_lambdas[0]
        assert lr(5) == pytest.approx(0.5)

    def test_fixed_warmup_with_cosine_still_gets_the_horizon(self, demo_model):
        captured = {}

        def spy(optimizer, num_warmup_steps, num_training_steps):
            captured.update(num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        module = _module(demo_model, partial(spy), warmup_steps=7)
        module.trainer = Mock(estimated_stepping_batches=100)
        module.configure_optimizers()
        assert captured == {"num_warmup_steps": 7, "num_training_steps": 100}

    def test_warmup_steps_round_trips_through_checkpoint(self, demo_model, tmp_path):
        module = _module(demo_model, partial(get_constant_schedule_with_warmup), warmup_steps=123)
        path = tmp_path / "m.ckpt"
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": dict(module.hparams),
                "pytorch-lightning_version": L.__version__,
            },
            path,
        )
        loaded = EveryQueryLightningModule.load_from_checkpoint(str(path))
        assert loaded.warmup_steps == 123
        assert loaded.warmup_ratio == 0.0


# ------------------------------------------------------------------------- end to end


def test_cli_anytime_run_publishes_ema_weights(task_labels_dir, tensorized_cohort_dir, tmp_path):
    """``EQ_train +anytime=ema25`` trains end to end and EQ_predict's loader picks up the EMA weights."""
    from every_query.utils.model_loader import setup_model
    from tests.test_train_cli import _run_train_subprocess

    result = _run_train_subprocess(
        task_labels_dir,
        tensorized_cohort_dir,
        tmp_path,
        extra_overrides=["+anytime=ema25", "lightning_module.warmup_steps=1"],
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    ckpt = torch.load(tmp_path / "best_model.ckpt", map_location="cpu", weights_only=False)
    ema = ema_state_from_checkpoint(ckpt)
    assert ema is not None, f"no AnytimeEMA state in callbacks {list(ckpt.get('callbacks', {}))}"
    assert ckpt["hyper_parameters"]["warmup_steps"] == 1
    assert ckpt["hyper_parameters"]["LR_scheduler"]["_target_"].endswith("get_constant_schedule_with_warmup")

    _, ema_module, _ = setup_model(tmp_path)
    _, raw_module, _ = setup_model(tmp_path, use_ema_weights=False)
    ema_params = dict(ema_module.named_parameters())
    raw_params = dict(raw_module.named_parameters())
    for n, t in ema.items():
        torch.testing.assert_close(ema_params[n].detach(), t)
    torch.testing.assert_close(raw_params[next(iter(ema))].detach(), ckpt["state_dict"][next(iter(ema))])
    assert any(not torch.equal(ema_params[n], raw_params[n]) for n in ema), "EMA should differ from raw"
