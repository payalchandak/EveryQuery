"""Lightning module perturbation tests for EveryQuery.

Covers training orchestration logic inside ``EveryQueryLightningModule``,
specifically optimizer parameter-group construction via ``configure_optimizers``
and the relationship between raw logits and predicted probabilities.
"""

import copy
from functools import partial

import torch

from every_query.lightning_module import EveryQueryLightningModule

_CONFIGURED_WD = 0.01


class TestWeightDecayParamGroupSeparation:
    """``configure_optimizers`` must produce two param groups:

    * Group 0 — non-norm/bias parameters with ``weight_decay == configured_value``.
    * Group 1 — norm/bias parameters with ``weight_decay == 0.0``.

    Every parameter appears in exactly one group, and the group assignment
    must agree with ``_is_norm_bias_param``.
    """

    @staticmethod
    def _param_name_map(module):
        """Map parameter ``data_ptr`` to its name for reverse-lookup."""
        return {p.data_ptr(): name for name, p in module.named_parameters()}

    def _build_module_and_optimizer(self, demo_model):
        module = EveryQueryLightningModule(
            model=demo_model,
            optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=_CONFIGURED_WD),
        )
        optimizer = module.configure_optimizers()
        return module, optimizer

    def test_group0_has_configured_weight_decay(self, demo_model):
        _, optimizer = self._build_module_and_optimizer(demo_model)
        assert optimizer.param_groups[0]["weight_decay"] == _CONFIGURED_WD

    def test_group1_has_zero_weight_decay(self, demo_model):
        _, optimizer = self._build_module_and_optimizer(demo_model)
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_group1_names_all_pass_is_norm_bias(self, demo_model):
        module, optimizer = self._build_module_and_optimizer(demo_model)
        ptr_to_name = self._param_name_map(module)

        group1_params = optimizer.param_groups[1]["params"]
        assert len(group1_params) > 0, "norm/bias group is empty — test would be vacuously true"

        for p in group1_params:
            name = ptr_to_name[p.data_ptr()]
            assert EveryQueryLightningModule._is_norm_bias_param(name), (
                f"{name!r} is in the norm/bias group but _is_norm_bias_param returns False"
            )

    def test_group0_names_all_fail_is_norm_bias(self, demo_model):
        module, optimizer = self._build_module_and_optimizer(demo_model)
        ptr_to_name = self._param_name_map(module)

        group0_params = optimizer.param_groups[0]["params"]
        assert len(group0_params) > 0, "non-norm/bias group is empty — test would be vacuously true"

        for p in group0_params:
            name = ptr_to_name[p.data_ptr()]
            assert not EveryQueryLightningModule._is_norm_bias_param(name), (
                f"{name!r} is in the non-norm/bias group but _is_norm_bias_param returns True"
            )

    def test_all_params_accounted_for(self, demo_model):
        module, optimizer = self._build_module_and_optimizer(demo_model)

        expected_ptrs = {p.data_ptr() for p in module.parameters()}
        grouped_ptrs = set()
        total_grouped = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                grouped_ptrs.add(p.data_ptr())
                total_grouped += 1

        assert expected_ptrs == grouped_ptrs, "Optimizer param groups don't cover all module parameters"
        assert total_grouped == len(expected_ptrs), (
            f"Expected {len(expected_ptrs)} params but groups contain {total_grouped} "
            "(some parameters appear in multiple groups)"
        )


class TestPredictProbsEqualSigmoidOfLogits:
    """``predict_step`` probabilities must numerically equal ``sigmoid(logits).squeeze()``.

    The forward pass produces raw logits via the censor/occurs MLP heads.
    ``EveryQueryOutput.logits_to_probs`` converts them with ``sigmoid + squeeze``.
    ``predict_step`` exposes these as ``occurs_probs`` and ``censor_probs``.
    This test verifies the full chain is consistent.
    """

    @staticmethod
    def _predict_batch(sample_batch):
        """Clone ``sample_batch`` with the metadata fields ``predict_step`` requires."""
        batch = copy.copy(sample_batch)
        batch.subject_id = torch.arange(sample_batch.batch_size)
        batch.prediction_time = torch.arange(sample_batch.batch_size)
        return batch

    @torch.no_grad()
    def test_occurs_probs_match_sigmoid_of_logits(self, demo_model, demo_lightning_module, sample_batch):
        _, outputs = demo_model._forward(sample_batch)
        preds = demo_lightning_module.predict_step(self._predict_batch(sample_batch))

        expected = torch.sigmoid(outputs.occurs_logits).squeeze().cpu()
        assert torch.allclose(preds["occurs_probs"], expected), (
            f"occurs_probs mismatch:\n  predict_step: {preds['occurs_probs']}\n  sigmoid(logits): {expected}"
        )

    @torch.no_grad()
    def test_censor_probs_match_sigmoid_of_logits(self, demo_model, demo_lightning_module, sample_batch):
        _, outputs = demo_model._forward(sample_batch)
        preds = demo_lightning_module.predict_step(self._predict_batch(sample_batch))

        expected = torch.sigmoid(outputs.censor_logits).squeeze().cpu()
        assert torch.allclose(preds["censor_probs"], expected), (
            f"censor_probs mismatch:\n  predict_step: {preds['censor_probs']}\n  sigmoid(logits): {expected}"
        )
