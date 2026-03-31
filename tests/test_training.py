"""Lightning-level integration tests for the EveryQuery training stack.

Tests exercise the model, lightning module, and trainer using tiny
randomly-initialised models (no pretrained weights) and the session-scoped
``sample_batch`` / ``demo_dataset`` fixtures from ``conftest.py``.
"""

import logging
from functools import partial
from unittest.mock import patch

import lightning as L
import torch
from torch.utils.data import DataLoader

from every_query.lightning_module import EveryQueryLightningModule
from every_query.model import EveryQueryModel, EveryQueryOutput

# ── test_model_forward_shape ────────────────────────────────────────────


class TestModelForwardShape:
    """Forward pass produces correctly-shaped outputs and a finite scalar loss."""

    @torch.no_grad()
    def test_loss_is_finite_scalar(self, demo_model, sample_batch):
        loss, _ = demo_model(sample_batch)
        assert loss.shape == torch.Size([])
        assert loss.isfinite()

    @torch.no_grad()
    def test_query_embed_shape(self, demo_model, sample_batch, demo_model_config):
        _, outputs = demo_model(sample_batch)
        assert outputs.query_embed.shape == (sample_batch.batch_size, demo_model_config.hidden_size)

    @torch.no_grad()
    def test_censor_logits_shape(self, demo_model, sample_batch):
        _, outputs = demo_model(sample_batch)
        assert outputs.censor_logits.shape == (sample_batch.batch_size, 1)

    @torch.no_grad()
    def test_occurs_logits_shape(self, demo_model, sample_batch):
        _, outputs = demo_model(sample_batch)
        assert outputs.occurs_logits.shape == (sample_batch.batch_size, 1)

    @torch.no_grad()
    def test_output_type(self, demo_model, sample_batch):
        _, outputs = demo_model(sample_batch)
        assert isinstance(outputs, EveryQueryOutput)

    @torch.no_grad()
    def test_loss_decomposes(self, demo_model, sample_batch):
        loss, outputs = demo_model(sample_batch)
        assert outputs.censor_loss.isfinite() and outputs.occurs_loss.isfinite()
        assert torch.allclose(loss, outputs.censor_loss + outputs.occurs_loss)


# ── test_model_backward ─────────────────────────────────────────────────


class TestModelBackward:
    """Backward pass populates finite gradients on all trainable parameters."""

    def test_gradients_populated_and_finite(self, demo_model_config, sample_batch):
        model = EveryQueryModel(
            model_name_or_config=demo_model_config,
            do_demo=True,
            precision="32-true",
        )
        model.train()

        loss, _ = model(sample_batch)
        loss.backward()

        has_grad = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                assert p.grad.isfinite().all(), f"Non-finite gradient in {name}"

        assert has_grad, "No parameter received a gradient"


# ── test_lightning_training_step ─────────────────────────────────────────


class TestLightningTrainingStep:
    """``training_step`` returns a finite scalar loss with gradients attached."""

    def test_returns_finite_scalar_with_grad(self, demo_model_config, sample_batch):
        model = EveryQueryModel(
            model_name_or_config=demo_model_config,
            do_demo=True,
            precision="32-true",
        )
        module = EveryQueryLightningModule(
            model=model,
            optimizer=partial(torch.optim.AdamW, lr=1e-4),
        )

        with patch.object(module, "_log_metrics"):
            loss = module.training_step(sample_batch)

        assert loss.shape == torch.Size([])
        assert loss.isfinite()
        assert loss.requires_grad


# ── test_trainer_fit_two_steps ───────────────────────────────────────────


class TestTrainerFitTwoSteps:
    """``Trainer.fit`` completes two optimiser steps without error."""

    def test_fit_completes(self, demo_model_config, demo_dataset):
        model = EveryQueryModel(
            model_name_or_config=demo_model_config,
            do_demo=True,
            precision="32-true",
        )
        module = EveryQueryLightningModule(
            model=model,
            optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.01),
        )

        train_dl = DataLoader(
            demo_dataset,
            batch_size=2,
            collate_fn=demo_dataset.collate,
            shuffle=False,
        )
        val_dl = DataLoader(
            demo_dataset,
            batch_size=2,
            collate_fn=demo_dataset.collate,
            shuffle=False,
        )

        trainer = L.Trainer(
            max_steps=2,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            deterministic=True,
        )

        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
        assert trainer.global_step == 2


# ── test_checkpoint_roundtrip ────────────────────────────────────────────


class TestCheckpointRoundtrip:
    """Save → load preserves hyper-parameters and weights."""

    def test_hparams_and_weights_survive(self, demo_model_config, demo_dataset, tmp_path):
        model = EveryQueryModel(
            model_name_or_config=demo_model_config,
            do_demo=True,
            precision="32-true",
        )
        module = EveryQueryLightningModule(
            model=model,
            optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.01),
        )

        train_dl = DataLoader(
            demo_dataset,
            batch_size=2,
            collate_fn=demo_dataset.collate,
            shuffle=False,
        )
        val_dl = DataLoader(
            demo_dataset,
            batch_size=2,
            collate_fn=demo_dataset.collate,
            shuffle=False,
        )

        trainer = L.Trainer(
            max_steps=2,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            deterministic=True,
        )
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

        ckpt_path = tmp_path / "test.ckpt"
        trainer.save_checkpoint(ckpt_path)

        loaded = EveryQueryLightningModule.load_from_checkpoint(str(ckpt_path))

        assert loaded.hparams["model"]["precision"] == "32-true"
        assert loaded.hparams["model"]["do_demo"] is True
        assert loaded.hparams["optimizer"]["_target_"] == "torch.optim.adamw.AdamW"
        assert loaded.hparams["LR_scheduler"] is None

        batch = next(iter(train_dl))
        with torch.no_grad():
            _, orig_out = module.model(batch)
            _, load_out = loaded.model(batch)

        assert orig_out.query_embed.shape == load_out.query_embed.shape
        assert orig_out.censor_logits.shape == load_out.censor_logits.shape
        assert orig_out.occurs_logits.shape == load_out.occurs_logits.shape

        for p_orig, p_loaded in zip(module.parameters(), loaded.parameters(), strict=True):
            assert torch.equal(p_orig, p_loaded)


# ── test_demo_mode_checks ───────────────────────────────────────────────


class TestDemoModeChecks:
    """``do_demo=True`` exercises ``_check_inputs/_parameters/_outputs``."""

    @torch.no_grad()
    def test_clean_batch_passes_all_checks(self, demo_model, sample_batch):
        assert demo_model.do_demo
        loss, outputs = demo_model(sample_batch)
        assert loss.isfinite()
        assert isinstance(outputs, EveryQueryOutput)

    def test_check_inputs_called(self, demo_model, sample_batch):
        with patch.object(demo_model, "_check_inputs", wraps=demo_model._check_inputs) as mock:
            with torch.no_grad():
                demo_model(sample_batch)
            mock.assert_called_once()

    def test_check_parameters_called(self, demo_model, sample_batch):
        with patch.object(demo_model, "_check_parameters", wraps=demo_model._check_parameters) as mock:
            with torch.no_grad():
                demo_model(sample_batch)
            mock.assert_called_once()

    def test_check_outputs_called(self, demo_model, sample_batch):
        with patch.object(demo_model, "_check_outputs", wraps=demo_model._check_outputs) as mock:
            with torch.no_grad():
                demo_model(sample_batch)
            mock.assert_called_once()

    def test_nan_injection_logs_warning(self, demo_model_config, sample_batch, caplog):
        model = EveryQueryModel(
            model_name_or_config=demo_model_config,
            do_demo=True,
            precision="32-true",
        )
        with torch.no_grad():
            next(iter(model.censor_mlp.parameters())).fill_(float("nan"))

        with caplog.at_level(logging.WARNING, logger="every_query.model"), torch.no_grad():
            model(sample_batch)

        assert any("nan" in msg.lower() for msg in caplog.messages)
