"""Loop contracts of ``ConditionalMultitaskLightningModule`` (issue #30, part 2).

Training / validation run the dense all-vocabulary ``forward`` over a ``MultitaskBoundaryBatch``;
test / prediction run ``score_final_query`` over a ``MultitaskEvalBatch``.  Everything is hand-built
on CPU from the model-test helpers (a tiny ``ConditionalMultitaskARModel``, no cohort):

1.  ``validation_step`` is the dense objective and calls ``forward`` (never ``score_final_query``);
2.  ``test_step`` is the BCE of the final-query logits against ``labels`` and logs ``held_out/loss``;
3.  ``predict_step`` returns row-aligned ``probs`` / ``labels`` / ``scored_codes`` without ``targets``;
4.  oracle: each prediction logit equals the dense forward's ``logits[b, last, scored_codes[b]]`` on an
    equivalent ``K = n_queries[b]`` training batch;
5.  a real ``Trainer`` runs validate / test / predict end to end with the expected metric names and
    exact row order;
6.  the evaluation loops never call the dense forward and never build a ``V``-wide projection.
"""

import math
from functools import partial

import lightning as L
import pytest
import torch
from meds import held_out_split, train_split, tuning_split
from torch.utils.data import DataLoader

from every_query.model.conditional_multitask_ar_model import ConditionalMultitaskARModel
from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
from tests.test_conditional_multitask_ar_model import VOCAB, make_batch, make_eval_batch, tiny_model


@pytest.fixture
def module() -> ConditionalMultitaskLightningModule:
    module = ConditionalMultitaskLightningModule(
        model=tiny_model(), optimizer=partial(torch.optim.AdamW, lr=1e-4)
    )
    module.eval()
    return module


@pytest.fixture
def calls(monkeypatch) -> dict[str, int]:
    """Count class-level calls to the two model paths without changing what they compute."""
    counts = {"forward": 0, "score_final_query": 0}
    for name in counts:
        real = getattr(ConditionalMultitaskARModel, name)

        def spy(self, *args, _real=real, _name=name, **kwargs):
            counts[_name] += 1
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(ConditionalMultitaskARModel, name, spy)
    return counts


def _capture_log(monkeypatch, module) -> list[tuple[str, float, dict]]:
    logged = []

    def log(name, value, **kwargs):
        logged.append((name, value, kwargs))

    monkeypatch.setattr(module, "log", log)
    return logged


# --- 1: validation is the dense training objective ---------------------------------------------


def test_validation_step_is_the_dense_training_objective(module, calls, monkeypatch):
    logged = _capture_log(monkeypatch, module)
    batch = make_batch()
    loss = module.validation_step(batch)
    assert calls == {"forward": 1, "score_final_query": 0}
    assert loss.ndim == 0 and loss.isfinite() and not loss.requires_grad

    expected, _ = module.model(batch)
    torch.testing.assert_close(loss, expected)
    assert [name for name, _, _ in logged] == [f"{tuning_split}/loss"]
    _, value, kwargs = logged[0]
    assert value == pytest.approx(expected.item())
    assert kwargs["batch_size"] == batch.batch_size and kwargs["on_epoch"] and not kwargs["on_step"]

    # The training step is the same objective on the same batch (only its logging differs).
    train_loss = module.training_step(batch)
    torch.testing.assert_close(train_loss.detach(), expected)
    assert logged[-1][0] == f"{train_split}/loss" and logged[-1][2]["on_step"]


# --- 2: test is the scalar final-query loss -------------------------------------------------------


def test_test_step_scores_the_final_query_and_logs_held_out_loss(module, calls, monkeypatch):
    logged = _capture_log(monkeypatch, module)
    batch = make_eval_batch(n_queries=[3, 1], labels=[True, False])
    loss = module.test_step(batch)
    assert calls == {"forward": 0, "score_final_query": 1}
    assert loss.ndim == 0 and loss.isfinite() and loss.dtype == torch.float32 and not loss.requires_grad

    logits = module.model.score_final_query(batch, batch.scored_codes)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels.float())
    torch.testing.assert_close(loss, expected)
    # Mean over the batch's rows, one scalar per row.
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.labels.float(), reduction="none"
    )
    torch.testing.assert_close(loss, per_row.mean())

    assert [name for name, _, _ in logged] == [f"{held_out_split}/loss"]
    _, value, kwargs = logged[0]
    assert value == pytest.approx(expected.item())
    assert kwargs["batch_size"] == 2 and kwargs["on_epoch"] and not kwargs["on_step"]
    assert kwargs["sync_dist"] is False  # not distributed here

    # The labels really participate: flipping them changes the loss.
    flipped = make_eval_batch(n_queries=[3, 1], labels=[False, True])
    assert not torch.isclose(module.test_step(flipped), loss)


# --- 3: prediction output contract ------------------------------------------------------------


def test_predict_step_returns_row_aligned_probabilities_without_targets(module, calls):
    batch = make_eval_batch(
        code=[[2, 3, 4, 5], [7, 8, 0, 0], [9, 10, 11, 0]],
        n_queries=[2, 3, 1],
        scored_codes=[4, 17, 30],
        labels=[True, False, True],
    )
    assert not hasattr(batch, "targets")

    out = module.predict_step(batch)
    assert calls == {"forward": 0, "score_final_query": 1}
    assert set(out) == {"probs", "labels", "scored_codes"}
    for tensor in out.values():
        assert tensor.shape == (3,) and tensor.device.type == "cpu" and not tensor.requires_grad
    assert out["probs"].dtype == torch.float32
    assert out["labels"].dtype == torch.bool
    assert out["scored_codes"].dtype == torch.int64
    assert ((out["probs"] >= 0) & (out["probs"] <= 1)).all()
    assert out["labels"].tolist() == [True, False, True]
    assert out["scored_codes"].tolist() == [4, 17, 30]
    torch.testing.assert_close(
        out["probs"], torch.sigmoid(module.model.score_final_query(batch, batch.scored_codes))
    )


# --- 4: oracle against the dense forward ------------------------------------------------------


def test_prediction_logits_match_the_dense_forward_per_row(module):
    """Each row of a padded eval batch scores exactly like a ``K = n_queries[b]`` training batch carrying the
    same windows and conditions, read at ``logits[0, last, scored_codes[b]]``."""
    code = [[2, 3, 4, 5], [7, 8, 0, 0], [9, 10, 11, 0], [12, 13, 14, 15]]
    n_queries = [3, 1, 2, 3]
    scored = [2, 17, 5, 30]
    batch = make_eval_batch(code=code, n_queries=n_queries, scored_codes=scored)
    assert batch.q_mask.tolist() == [[True] * 3, [True, False, False], [True, True, False], [True] * 3]

    logits = module._score_final_query(batch)
    probs = module.predict_step(batch)["probs"]
    assert logits.shape == probs.shape == (4,) and logits.dtype == torch.float32
    for b, (row, n, c) in enumerate(zip(code, n_queries, scored, strict=True)):
        dense = make_batch(n_windows=n, code=[row])  # same windows / conditions, K = n, dummy targets
        assert dense.targets.shape == (1, n, VOCAB)
        _, out = module.model(dense)
        expected = out.logits[0, n - 1, c]
        torch.testing.assert_close(logits[b], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(probs[b], torch.sigmoid(expected), atol=1e-5, rtol=1e-5)
    # Different rows really are being compared, not one repeated value.
    assert len({round(v, 4) for v in logits.tolist()}) > 1


# --- 5: real Trainer, all three evaluation loops ------------------------------------------------


def test_trainer_validate_test_and_predict_end_to_end(module):
    boundary_batches = [
        make_batch(),
        make_batch(code=[[9, 10, 11, 0], [12, 13, 14, 15]]),
        make_batch(n_windows=2, code=[[2, 3, 0, 0]]),
    ]
    eval_batches = [
        make_eval_batch(n_queries=[3, 1], scored_codes=[2, 3], labels=[True, False]),
        make_eval_batch(code=[[9, 10, 11, 0]], n_queries=[2], scored_codes=[4], labels=[True]),
        make_eval_batch(
            code=[[12, 13, 14, 15], [2, 3, 4, 5], [7, 8, 0, 0]],
            n_queries=[1, 3, 2],
            scored_codes=[5, 6, 7],
            labels=[False, True, False],
        ),
    ]
    # Direct per-batch values, taken before a Trainer is attached (``self.log`` is a no-op then).
    val_losses = [module.validation_step(b).item() for b in boundary_batches]
    test_losses = [module.test_step(b).item() for b in eval_batches]
    direct_probs = torch.cat([module.predict_step(b)["probs"] for b in eval_batches])

    def weighted_mean(losses, batches):
        sizes = [b.batch_size for b in batches]
        return sum(loss * n for loss, n in zip(losses, sizes, strict=True)) / sum(sizes)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=1,
        limit_val_batches=3,
        limit_test_batches=3,
        limit_predict_batches=3,
    )
    val_metrics = trainer.validate(module, dataloaders=DataLoader(boundary_batches, batch_size=None))
    assert len(val_metrics) == 1 and set(val_metrics[0]) == {f"{tuning_split}/loss"}
    assert math.isfinite(val_metrics[0][f"{tuning_split}/loss"])
    assert val_metrics[0][f"{tuning_split}/loss"] == pytest.approx(
        weighted_mean(val_losses, boundary_batches), rel=1e-5
    )

    test_metrics = trainer.test(module, dataloaders=DataLoader(eval_batches, batch_size=None))
    assert len(test_metrics) == 1 and set(test_metrics[0]) == {f"{held_out_split}/loss"}
    assert test_metrics[0][f"{held_out_split}/loss"] == pytest.approx(
        weighted_mean(test_losses, eval_batches), rel=1e-5
    )

    predictions = trainer.predict(module, dataloaders=DataLoader(eval_batches, batch_size=None))
    assert len(predictions) == 3 and all(set(p) == {"probs", "labels", "scored_codes"} for p in predictions)
    probs = torch.cat([p["probs"] for p in predictions])
    assert probs.shape == (6,) and probs.dtype == torch.float32 and probs.device.type == "cpu"
    # Row order is the input order: the scored codes / labels were chosen distinct per row.
    assert torch.cat([p["scored_codes"] for p in predictions]).tolist() == [2, 3, 4, 5, 6, 7]
    assert torch.cat([p["labels"] for p in predictions]).tolist() == [True, False, True, False, True, False]
    torch.testing.assert_close(probs, direct_probs)

    # The parent's epoch hooks ran over the deliberately empty metric dictionaries.
    assert module.metrics == {train_split: {}, tuning_split: {}, held_out_split: {}}


# --- 6: no dense path during evaluation ----------------------------------------------------------


def test_evaluation_loops_never_call_the_dense_forward(module, monkeypatch):
    def boom(self, batch):
        raise AssertionError("dense forward called during a QuerySeq evaluation loop")

    monkeypatch.setattr(ConditionalMultitaskARModel, "forward", boom)
    batch = make_eval_batch(n_queries=[3, 1])
    assert module.predict_step(batch)["probs"].shape == (2,)
    assert module.test_step(batch).isfinite()
    with pytest.raises(AssertionError, match="dense forward"):
        module.validation_step(make_batch())
    with pytest.raises(AssertionError, match="dense forward"):
        module.training_step(make_batch())


def test_evaluation_loops_never_build_a_vocabulary_wide_projection(module, monkeypatch):
    """Mirror of the model-level guard: no ``(.., V)`` matmul happens under test / predict."""
    calls = []
    real_matmul = torch.Tensor.__matmul__

    def spy(a, b):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return real_matmul(a, b)

    monkeypatch.setattr(torch.Tensor, "__matmul__", spy)
    batch = make_eval_batch(n_queries=[3, 1])
    module.predict_step(batch)
    module.test_step(batch)
    assert not any(shape[-1] == VOCAB for _, shape in calls), "no (.., V) projection may be built"

    module.validation_step(make_batch())
    assert any(shape[-1] == VOCAB for _, shape in calls), "validation must project onto the whole vocabulary"
