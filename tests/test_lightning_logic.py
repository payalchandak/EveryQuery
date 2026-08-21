"""Lightning module perturbation tests for EveryQuery.

Covers training orchestration logic inside ``EveryQueryLightningModule``,
specifically optimizer parameter-group construction via ``configure_optimizers``
and the relationship between raw logits and predicted probabilities.
"""

import contextlib
import os
import socket
from functools import partial

import pytest
import torch
from meds import tuning_split
from meds_torchdata import MEDSTorchDataConfig

from every_query.data.dataset import EveryQueryBatch
from every_query.model import EveryQueryOutput
from every_query.model.lightning_module import EveryQueryLightningModule
from every_query.model.task_auroc_callback import TaskAurocTrackingCallback

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


class TestWarmupStepsFromRatio:
    """``configure_optimizers`` sizes warmup as ``warmup_ratio * estimated_stepping_batches``."""

    @pytest.mark.parametrize(
        ("total_steps", "ratio", "expected_warmup"),
        [(100, 0.10, 10), (1000, 0.0, 0), (7, 0.5, 3)],  # last case: int() truncates 3.5
    )
    def test_warmup_steps(self, demo_model, total_steps, ratio, expected_warmup):
        from unittest.mock import Mock

        captured = {}

        def spy_scheduler(optimizer, num_warmup_steps, num_training_steps):
            captured["num_warmup_steps"] = num_warmup_steps
            captured["num_training_steps"] = num_training_steps
            return Mock()

        module = EveryQueryLightningModule(
            model=demo_model,
            optimizer=partial(torch.optim.AdamW, lr=1e-4),
            LR_scheduler=partial(spy_scheduler),
            warmup_ratio=ratio,
        )
        module.trainer = Mock(estimated_stepping_batches=total_steps)

        module.configure_optimizers()

        assert captured == {"num_warmup_steps": expected_warmup, "num_training_steps": total_steps}

    def test_production_scheduler_partial_ramps_over_warmup(self, demo_model):
        """The exact config.yaml partial (cosine + num_cycles=0.5) accepts the derived step counts."""
        from unittest.mock import Mock

        from transformers import get_cosine_schedule_with_warmup

        module = EveryQueryLightningModule(
            model=demo_model,
            optimizer=partial(torch.optim.AdamW, lr=1e-4),
            LR_scheduler=partial(get_cosine_schedule_with_warmup, num_cycles=0.5),
            warmup_ratio=0.10,
        )
        module.trainer = Mock(estimated_stepping_batches=100)

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        lr_lambda = scheduler.lr_lambdas[0]
        assert lr_lambda(5) == pytest.approx(0.5)  # halfway through the 10-step warmup
        assert lr_lambda(10) == pytest.approx(1.0)  # warmup complete at step 10


class TestPredictProbsEqualSigmoidOfLogits:
    """``predict_step`` probabilities must numerically equal ``sigmoid(logits).squeeze()``.

    The forward pass produces raw logits via the censor/occurs MLP heads.
    ``EveryQueryOutput.logits_to_probs`` converts them with ``sigmoid + squeeze``.
    ``predict_step`` exposes these as ``occurs_probs`` and ``censor_probs``.
    This test verifies the full chain is consistent.
    """

    @torch.no_grad()
    def test_occurs_probs_match_sigmoid_of_logits(self, demo_model, demo_lightning_module, sample_batch):
        _, outputs = demo_model._forward(sample_batch)
        preds = demo_lightning_module.predict_step(sample_batch)

        expected = torch.sigmoid(outputs.occurs_logits).squeeze().cpu()
        assert torch.allclose(preds["occurs_probs"], expected), (
            f"occurs_probs mismatch:\n  predict_step: {preds['occurs_probs']}\n  sigmoid(logits): {expected}"
        )

    @torch.no_grad()
    def test_censor_probs_match_sigmoid_of_logits(self, demo_model, demo_lightning_module, sample_batch):
        _, outputs = demo_model._forward(sample_batch)
        preds = demo_lightning_module.predict_step(sample_batch)

        expected = torch.sigmoid(outputs.censor_logits).squeeze().cpu()
        assert torch.allclose(preds["censor_probs"], expected), (
            f"censor_probs mismatch:\n  predict_step: {preds['censor_probs']}\n  sigmoid(logits): {expected}"
        )


class _StubOccursModel(torch.nn.Module):
    """Fake model returning ``EveryQueryOutput`` built from a queue of caller-pinned occurs logits.

    Ignores the batch's actual sequence data entirely — each ``forward`` call consumes
    ``batch.query.shape[0]`` logits off the front of the queue (in the order the test
    enqueued them) so tests can pin exact per-row probabilities without a real transformer.
    """

    def __init__(self, logits: list[float]):
        super().__init__()
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._logits = list(logits)

    def forward(self, batch: EveryQueryBatch):
        n = batch.query.shape[0]
        row_logits = torch.tensor(self._logits[:n], dtype=torch.float32).unsqueeze(1)
        self._logits = self._logits[n:]
        outputs = EveryQueryOutput(last_hidden_state=None, occurs_logits=row_logits)
        return torch.tensor(0.0), outputs


def _make_tracking_batch(queries: list[int], durations: list[float], occurs: list[int]) -> EveryQueryBatch:
    n = len(queries)
    seq_len = 2
    return EveryQueryBatch(
        code=torch.zeros(n, seq_len, dtype=torch.long),
        numeric_value=torch.zeros(n, seq_len),
        numeric_value_mask=torch.zeros(n, seq_len, dtype=torch.bool),
        time_delta_days=torch.zeros(n, seq_len),
        occurs=torch.tensor(occurs, dtype=torch.long),
        query=torch.tensor(queries, dtype=torch.long),
        duration_days=torch.tensor(durations, dtype=torch.float32),
    )


class TestSampledTaskAurocTracking:
    """``TaskAurocTrackingCallback._compute_and_log`` — macro win/tie/loss AUROC over pairs.

    Each task in the tracking set contributes exactly one positive and one negative row; per-task AUROC
    collapses to an indicator on which of the two scored higher.
    """

    @staticmethod
    def _run(model, batches):
        """Score ``batches`` with ``model`` via the callback, returning the logged metrics dict."""
        cb = TaskAurocTrackingCallback(config=None)
        cb._loader = batches
        logged = {}
        cb._compute_and_log(
            model,
            torch.device("cpu"),
            lambda name, value, **kw: logged.__setitem__(name, value),
        )
        return logged

    def test_no_op_when_loader_empty(self):
        assert self._run(_StubOccursModel([]), []) == {}

    def test_macro_average_win_tie_loss(self):
        # Task 10: neg logit -10 (~0 prob), pos logit +10 (~1 prob) -> win  (indicator 1.0)
        # Task 20: neg logit +10 (~1 prob), pos logit -10 (~0 prob) -> loss (indicator 0.0)
        # Task 30: neg logit  0.0,          pos logit  0.0 (tie)    -> tie  (indicator 0.5)
        batch = _make_tracking_batch([10, 10, 20, 20, 30, 30], [7.0] * 6, [0, 1, 0, 1, 0, 1])
        logits = [-10.0, 10.0, 10.0, -10.0, 0.0, 0.0]

        logged = self._run(_StubOccursModel(logits), [batch])

        assert logged["tuning/occurs_auroc_macro_sampled_n_tasks"] == 3.0
        assert logged["tuning/occurs_auroc_macro_sampled"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)

    def test_tasks_missing_a_class_are_dropped(self):
        # Task 10 has both classes (win); task 20 only has a positive row and is dropped.
        batch = _make_tracking_batch([10, 10, 20], [7.0, 7.0, 7.0], [0, 1, 1])
        logits = [-10.0, 10.0, 0.0]

        logged = self._run(_StubOccursModel(logits), [batch])

        assert logged["tuning/occurs_auroc_macro_sampled_n_tasks"] == 1.0
        assert logged["tuning/occurs_auroc_macro_sampled"] == pytest.approx(1.0)

    def test_oov_query_rows_are_skipped(self):
        # Both tasks are out-of-vocab (query == PAD_INDEX 0); without the guard their pos/neg
        # probs collide into one bogus (0, 7.0) task. They must be dropped, leaving no metric.
        batch = _make_tracking_batch([0, 0, 0, 0], [7.0] * 4, [0, 1, 0, 1])
        logits = [-10.0, 10.0, 10.0, -10.0]

        assert self._run(_StubOccursModel(logits), [batch]) == {}


class TestCallbackHydraWiring:
    """The ``trainer.callbacks.task_auroc_tracking`` block ships commented out, so nothing else ever resolves
    its ``_target_`` paths or constructor args.

    This mirrors that block and instantiates it so a renamed arg, typo'd import path, or MEDSTorchDataConfig
    drift fails a test instead of silently passing CI.
    """

    def test_instantiates_callback_and_nested_config(self, tmp_path):
        import hydra

        # MEDSTorchDataConfig validates that both dirs exist.
        cohort_dir = tmp_path / "cohort"
        cohort_dir.mkdir()
        tracking_dir = tmp_path / "tracking"
        tracking_dir.mkdir()

        cfg = {
            "_target_": "every_query.model.task_auroc_callback.TaskAurocTrackingCallback",
            "batch_size": 256,
            "config": {
                "_target_": "meds_torchdata.MEDSTorchDataConfig",
                "tensorized_cohort_dir": str(cohort_dir),
                "task_labels_dir": str(tracking_dir),
                "static_inclusion_mode": "omit",
                "seq_sampling_strategy": "to_end",
                "max_seq_len": 256,
            },
        }

        callback = hydra.utils.instantiate(cfg)

        assert isinstance(callback, TaskAurocTrackingCallback)
        assert callback.batch_size == 256
        assert isinstance(callback.config, MEDSTorchDataConfig)


def _stub_module() -> EveryQueryLightningModule:
    """A lightning module with a throwaway model, for exercising the metric plumbing only."""
    return EveryQueryLightningModule(model=_StubOccursModel([]))


class TestEpochEndAUROCLogging:
    """``_on_epoch_end`` must log the epoch AUROCs, and fail loudly instead of dropping them.

    The metrics used to be plain CPU ``BinaryAUROC`` objects held in a plain dict, computed inside a bare
    ``except Exception: pass`` — so under DDP the states never followed the module onto the GPU and every
    failure of the distributed all-gather was swallowed silently.
    """

    @staticmethod
    def _log_calls(module) -> dict[str, float]:
        """Run ``_on_epoch_end`` for the tuning split, capturing what it logged."""
        logged = {}
        module.log = lambda name, value, **kw: logged.__setitem__(name, value)
        module._on_epoch_end(tuning_split)
        return logged

    def test_metrics_are_registered_submodules(self):
        """Registration is what makes torchmetrics' DDP sync work — a plain dict is invisible to ``.to()``."""
        module = _stub_module()
        named = dict(module.named_modules())

        assert f"metrics.{tuning_split}.censor_auc" in named
        assert f"metrics.{tuning_split}.occurs_auc" in named

    def test_metric_states_follow_the_module_device(self):
        module = _stub_module().to(torch.float64)
        metric = module.metrics[tuning_split]["censor_auc"]

        # CPU-only CI cannot check a real device move, so check the update path coerces onto the metric.
        module._update_metric("censor_auc", tuning_split, preds=torch.tensor([0.1]), target=torch.tensor([0]))
        assert metric.target[0].device == metric.device

    def test_no_new_persistent_checkpoint_keys(self):
        """Existing checkpoints must keep loading: metric states are non-persistent, so nothing is added."""
        assert [k for k in _stub_module().state_dict() if "metrics" in k] == []

    def test_logs_auroc_when_both_classes_present(self):
        module = _stub_module()
        module._update_metric(
            "censor_auc", tuning_split, preds=torch.tensor([0.1, 0.9]), target=torch.tensor([0, 1])
        )

        logged = self._log_calls(module)

        assert logged[f"{tuning_split}/censor_auc"] == pytest.approx(1.0)
        # occurs_auc never got data, so it is undefined and simply absent.
        assert f"{tuning_split}/occurs_auc" not in logged

    def test_skips_and_resets_when_only_one_class_present(self):
        module = _stub_module()
        module._update_metric(
            "censor_auc", tuning_split, preds=torch.tensor([0.1, 0.9]), target=torch.tensor([0, 0])
        )

        logged = self._log_calls(module)

        assert logged == {}
        assert module.metrics[tuning_split]["censor_auc"].target == []

    def test_state_is_reset_between_epochs(self):
        module = _stub_module()
        module._update_metric(
            "censor_auc", tuning_split, preds=torch.tensor([0.1, 0.9]), target=torch.tensor([0, 1])
        )
        self._log_calls(module)

        assert module.metrics[tuning_split]["censor_auc"].target == []

    def test_compute_failures_are_not_swallowed(self):
        """The old bare ``except Exception: pass`` is what made the DDP breakage invisible."""
        module = _stub_module()
        metric = module.metrics[tuning_split]["censor_auc"]
        module._update_metric(
            "censor_auc", tuning_split, preds=torch.tensor([0.1, 0.9]), target=torch.tensor([0, 1])
        )
        metric.compute = lambda: (_ for _ in ()).throw(RuntimeError("all-gather blew up"))

        with pytest.raises(RuntimeError, match="all-gather blew up"):
            self._log_calls(module)


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _ddp_epoch_end_worker(rank: int, world_size: int, port: int, queue) -> None:
    """One rank of the DDP epoch-end test: update rank-specific data, then log the epoch metrics."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        module = _stub_module()
        if rank == 0:
            # Both classes locally, plus the only data `occurs_auc` ever sees.
            preds, target = torch.tensor([0.1, 0.9]), torch.tensor([0, 1])
            module._update_metric(
                "occurs_auc", tuning_split, preds=torch.tensor([0.1, 0.9]), target=torch.tensor([0, 1])
            )
        else:
            # Negatives only: rank-dependent branching here is what used to hang the all-gather.
            preds, target = torch.tensor([0.2, 0.95]), torch.tensor([0, 0])
        module._update_metric("censor_auc", tuning_split, preds=preds, target=target)

        logged = {}
        module.log = lambda name, value, **kw: logged.__setitem__(name, value)
        module._on_epoch_end(tuning_split)
        queue.put((rank, logged))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
class TestEpochEndAUROCUnderDDP:
    """Two gloo ranks with rank-uneven, rank-degenerate data must both log the *global* AUROC.

    Regression test for the reported multi-GPU failure: the per-rank ``has_both_classes`` guard let one
    rank skip ``compute()`` while the other entered its blocking all-gather.
    """

    def test_both_ranks_log_the_global_auroc(self):
        ctx = torch.multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        port = _free_port()
        procs = [ctx.Process(target=_ddp_epoch_end_worker, args=(r, 2, port, queue)) for r in range(2)]
        for p in procs:
            p.start()

        results = {}
        try:
            for _ in procs:
                results.update(dict([queue.get(timeout=180)]))
        except Exception as e:  # pragma: no cover - only hit if the all-gather hangs
            raise AssertionError(f"DDP epoch-end did not complete on both ranks: {e}") from e
        finally:
            for p in procs:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()

        assert set(results) == {0, 1}
        for rank, logged in results.items():
            # Global pool: one positive (0.9) against negatives 0.1, 0.2 and 0.95 -> 2/3.
            # Rank 0 alone would score 1.0, so this also proves the states really were gathered.
            assert logged[f"{tuning_split}/censor_auc"] == pytest.approx(2 / 3), f"rank {rank}"
            # Only rank 0 updated `occurs_auc`; syncing a rank with no data at all is unsupported, so
            # every rank must skip it identically rather than one hanging or aborting the collective.
            assert f"{tuning_split}/occurs_auc" not in logged, f"rank {rank}"
