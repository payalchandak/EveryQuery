"""Contract tests for the decoder-only all-vocabulary multitask model."""

from functools import partial
from pathlib import Path

import polars as pl
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from every_query.data.multitask_dataset import MultitaskBoundaryBatch
from every_query.data.multitask_eval_dataset import MultitaskEvalBatch
from every_query.data.ontology import (
    EMBEDDING_MIX_FILE,
    EVENT_TO_QUERY_NODES_FILE,
    ONTOLOGY_VOCAB_FILE,
    build_event_to_query_nodes,
    build_ontology,
    derive_ancestor_targets,
    extended_vocab_size,
    load_closure_index,
    load_nodes,
)
from every_query.model.conditional_multitask_ar_model import (
    TYPE_CONDITION_ANSWER,
    TYPE_CONDITION_CODE,
    TYPE_WINDOW,
    ConditionalMultitaskARModel,
)
from every_query.model.ontology_embedding import OntologyEmbedding


@pytest.fixture(autouse=True)
def _setup_doctest_namespace():
    yield


VOCAB = 31
MODEL_CFG = {
    "hidden_size": 24,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 48,
    "max_position_embeddings": 64,
    "vocab_size": VOCAB,
    "pad_token_id": 0,
    "attention_dropout": 0.0,
}


def tiny_model(**kwargs) -> ConditionalMultitaskARModel:
    torch.manual_seed(7)
    overrides = dict(MODEL_CFG, **kwargs.pop("config_overrides", {}))
    model = ConditionalMultitaskARModel(config_overrides=overrides, max_windows=5, **kwargs)
    model.eval()
    return model


def make_batch(
    *,
    n_windows: int = 3,
    code: list[list[int]] | None = None,
    q_mask: list[list[bool]] | None = None,
    starts: bool = True,
    time_pos_ids: torch.Tensor | None = None,
) -> MultitaskBoundaryBatch:
    code = code or [[2, 3, 4, 5], [7, 8, 0, 0]]
    B, S = len(code), len(code[0])
    start_durations = torch.tensor([[0.0, 2.0, -1.0, 4.0, 0.0][:n_windows]] * B)
    start_codes = torch.tensor([[0, 0, 9, 0, 0][:n_windows]] * B)
    durations = torch.tensor([[7.0, -1.0, 30.0, 4.0, 2.0][:n_windows]] * B)
    bounds = torch.tensor([[0, 10, 0, 0, 0][:n_windows]] * B)
    mask = torch.tensor(q_mask or [[True] * n_windows] * B)
    targets = torch.zeros(B, n_windows, VOCAB, dtype=torch.bool)
    if n_windows:
        targets[:, :, 2] = True
        targets[:, 0, 0] = True  # PAD is intentionally true to prove loss masking.
    conditions = torch.tensor([[11, 12, 13, 14][: n_windows - 1]] * B, dtype=torch.long)
    answers = torch.tensor([[True, False, True, False][: n_windows - 1]] * B, dtype=torch.bool)
    kwargs = {}
    if starts:
        kwargs.update(q_start_durations=start_durations, q_start_codes=start_codes)
    return MultitaskBoundaryBatch(
        code=torch.tensor(code),
        numeric_value=torch.zeros(B, S),
        numeric_value_mask=torch.zeros(B, S, dtype=torch.bool),
        time_delta_days=torch.zeros(B, S),
        q_durations=durations,
        q_bound_codes=bounds,
        q_mask=mask,
        targets=targets,
        condition_codes=conditions,
        condition_answers=answers,
        time_pos_ids=time_pos_ids,
        **kwargs,
    )


def make_eval_batch(
    *,
    code: list[list[int]] | None = None,
    n_queries: list[int] | None = None,
    scored_codes: list[int] | None = None,
    labels: list[bool] | None = None,
) -> MultitaskEvalBatch:
    """A right-padded ``MultitaskEvalBatch`` whose real windows / conditions are ``make_batch``'s.

    Row ``i`` has ``n_queries[i]`` real windows (default: every row has three), so the equivalent
    training batch for row ``i`` is ``make_batch(n_windows=n_queries[i], code=[code[i]])``.
    """
    code = code or [[2, 3, 4, 5], [7, 8, 0, 0]]
    B, S = len(code), len(code[0])
    n_queries = n_queries or [3] * B
    k = max(n_queries)
    scored_codes = scored_codes or [2 + i for i in range(B)]
    labels = labels or [i % 2 == 0 for i in range(B)]

    start_durations = torch.zeros(B, k)
    start_codes = torch.zeros(B, k, dtype=torch.long)
    durations = torch.zeros(B, k)
    bounds = torch.zeros(B, k, dtype=torch.long)
    mask = torch.zeros(B, k, dtype=torch.bool)
    conditions = torch.zeros(B, k - 1, dtype=torch.long)  # PAD beyond the real conditions
    answers = torch.zeros(B, k - 1, dtype=torch.bool)
    for i, n in enumerate(n_queries):
        start_durations[i, :n] = torch.tensor([0.0, 2.0, -1.0, 4.0, 0.0][:n])
        start_codes[i, :n] = torch.tensor([0, 0, 9, 0, 0][:n])
        durations[i, :n] = torch.tensor([7.0, -1.0, 30.0, 4.0, 2.0][:n])
        bounds[i, :n] = torch.tensor([0, 10, 0, 0, 0][:n])
        mask[i, :n] = True
        conditions[i, : n - 1] = torch.tensor([11, 12, 13, 14][: n - 1], dtype=torch.long)
        answers[i, : n - 1] = torch.tensor([True, False, True, False][: n - 1])
    return MultitaskEvalBatch(
        code=torch.tensor(code),
        numeric_value=torch.zeros(B, S),
        numeric_value_mask=torch.zeros(B, S, dtype=torch.bool),
        time_delta_days=torch.zeros(B, S),
        q_start_durations=start_durations,
        q_start_codes=start_codes,
        q_durations=durations,
        q_bound_codes=bounds,
        q_mask=mask,
        condition_codes=conditions,
        condition_answers=answers,
        scored_codes=torch.tensor(scored_codes, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.bool),
        n_queries=torch.tensor(n_queries, dtype=torch.long),
    )


def test_forward_shape_dtype_mask_and_bias():
    model = tiny_model()
    batch = make_batch(q_mask=[[True, False, True]] * 2)
    # The tied projection must explicitly escape autocast; `.float()` operands alone are
    # downcast again by PyTorch's matmul autocast policy.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, out = model(batch)
    assert loss.isfinite()
    assert out.logits.shape == out.valid_mask.shape == (2, 3, VOCAB)
    assert out.logits.dtype == torch.float32 and out.probs.dtype == torch.float32
    assert torch.equal(model.code_bias, torch.full((VOCAB,), -3.0))
    assert not out.valid_mask[:, :, 0].any()
    assert not out.valid_mask[:, 1].any()
    assert out.valid_mask[:, (0, 2), 1:].all()


def test_exact_stream_order_and_token_count():
    model = tiny_model()
    batch = make_batch()
    tokens = model._query_tokens(batch)
    windows = model._window_embeds(batch)
    assert tokens.shape == (2, 3 * 3 - 2, MODEL_CFG["hidden_size"])
    torch.testing.assert_close(tokens[:, 0::3], windows)

    embedding = model.HF_model.get_input_embeddings()
    bp = model.block_pos_embed(torch.arange(2)).unsqueeze(0)
    tt = model.token_type_embed.weight
    expected_c = embedding(batch.condition_codes) + tt[TYPE_CONDITION_CODE] + bp
    expected_a = model.answer_embed(batch.condition_answers.long()) + tt[TYPE_CONDITION_ANSWER] + bp
    torch.testing.assert_close(tokens[:, 1::3], expected_c)
    torch.testing.assert_close(tokens[:, 2::3], expected_a)


def test_window_specs_use_matching_rows_and_distinct_roles():
    model = tiny_model()
    batch = make_batch()
    got = model._window_embeds(batch)
    starts, start_codes = model._start_fields(batch)
    embedding = model.HF_model.get_input_embeddings()
    start_d = model.start_duration_embed((starts / 365).unsqueeze(-1))
    start_e = embedding(start_codes) + model.start_marker
    end_d = model.end_duration_embed((batch.q_durations / 365).unsqueeze(-1))
    end_e = embedding(batch.q_bound_codes) + model.bound_marker
    expected = (
        torch.where((start_codes > 0).unsqueeze(-1), start_e, start_d)
        + torch.where((batch.q_bound_codes > 0).unsqueeze(-1), end_e, end_d)
        + model.token_type_embed.weight[TYPE_WINDOW]
        + model.block_pos_embed(torch.arange(3)).unsqueeze(0)
    )
    torch.testing.assert_close(got, expected)
    assert model.start_duration_embed is not model.end_duration_embed
    assert model.start_marker is not model.bound_marker


@pytest.mark.parametrize(
    ("start_duration", "start_code", "duration", "bound_code"),
    [
        (0.0, 0, 7.0, 0),
        (0.0, 0, -1.0, 10),
        (2.0, 0, 7.0, 0),
        (2.0, 0, -1.0, 10),
        (-1.0, 9, 7.0, 0),
        (-1.0, 9, -1.0, 10),
    ],
)
def test_all_six_start_end_combinations_are_finite(start_duration, start_code, duration, bound_code):
    model = tiny_model()
    batch = make_batch(n_windows=1)
    batch.q_start_durations.fill_(start_duration)
    batch.q_start_codes.fill_(start_code)
    batch.q_durations.fill_(duration)
    batch.q_bound_codes.fill_(bound_code)
    loss, out = model(batch)
    assert loss.isfinite() and out.logits.isfinite().all()


def test_current_answer_and_future_conditions_cannot_change_earlier_windows():
    model = tiny_model()
    base = make_batch()
    changed_answer = make_batch()
    changed_answer.condition_answers[:, 1] = ~changed_answer.condition_answers[:, 1]
    changed_future = make_batch()
    changed_future.condition_codes[:, 1] = 20
    _, a = model(base)
    _, b = model(changed_answer)
    _, c = model(changed_future)
    # A1 and C1 physically follow W1, so neither can change W0 or W1.
    torch.testing.assert_close(a.logits[:, :2], b.logits[:, :2])
    torch.testing.assert_close(a.logits[:, :2], c.logits[:, :2])
    assert not torch.equal(a.logits[:, 2], b.logits[:, 2])
    assert not torch.equal(a.logits[:, 2], c.logits[:, 2])


def test_earlier_answer_can_influence_later_window():
    model = tiny_model()
    a = make_batch()
    b = make_batch()
    b.condition_answers[:, 0] = ~b.condition_answers[:, 0]
    _, out_a = model(a)
    _, out_b = model(b)
    torch.testing.assert_close(out_a.logits[:, 0], out_b.logits[:, 0])
    assert not torch.equal(out_a.logits[:, 1:], out_b.logits[:, 1:])


def test_patient_padding_is_invisible_and_query_mask_covers_whole_block():
    model = tiny_model()
    narrow = make_batch(code=[[2, 3, 4, 5], [7, 8, 0, 0]], q_mask=[[True, False, True]] * 2)
    wide = make_batch(code=[[2, 3, 4, 5, 0, 0], [7, 8, 0, 0, 0, 0]], q_mask=[[True, False, True]] * 2)
    captured = {}

    def hook(_module, _args, kwargs):
        captured["mask"] = kwargs["attention_mask"].detach().clone()

    handle = model.HF_model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        _, out_wide = model(wide)
    finally:
        handle.remove()
    _, out_narrow = model(narrow)
    torch.testing.assert_close(out_narrow.logits[:, (0, 2)], out_wide.logits[:, (0, 2)])
    n_patient = (wide.code != 0).sum(1)
    for row, n in enumerate(n_patient.tolist()):
        assert captured["mask"][row, n : n + 7].tolist() == [1, 1, 1, 0, 0, 0, 1]


def test_left_or_interior_padding_is_rejected():
    """The stream layout assumes right padding; anything else must raise, not corrupt silently."""
    model = tiny_model()
    right = make_batch(code=[[2, 3, 4, 5], [7, 8, 0, 0]])
    loss, _ = model(right)
    assert loss.isfinite()
    left = make_batch(code=[[2, 3, 4, 5], [0, 0, 7, 8]])
    with pytest.raises(ValueError, match="prefix mask"):
        model(left)
    interior = make_batch(code=[[2, 3, 4, 5], [7, 0, 8, 0]])
    with pytest.raises(ValueError, match="prefix mask"):
        model(interior)


def test_duration_embedding_is_on_the_code_embedding_scale():
    """A duration-bounded window token must not be dominated by the duration MLP.

    Only the final Linear of each MLP is rescaled (weight std = initializer_range, bias 0); the hidden ReLU
    layer keeps its default init, so the output norm measures ~3-5x an embedding row (7d / 365d, over seeds)
    instead of the ~12-19x of the default final-layer init.
    """
    torch.manual_seed(0)
    H = 384
    model = ConditionalMultitaskARModel(
        config_overrides=dict(MODEL_CFG, hidden_size=H, intermediate_size=4 * H), max_windows=5
    )
    code_row_norm = model.HF_model.get_input_embeddings().weight.norm(dim=1).mean().item()
    for mlp in (model.start_duration_embed, model.end_duration_embed):
        final = mlp.model[-1]
        assert isinstance(final, torch.nn.Linear)
        assert torch.equal(final.bias, torch.zeros(H))
        for days in (7.0, 365.0):
            emb = mlp(torch.tensor([[days / 365.0]]))
            ratio = emb.norm().item() / code_row_norm
            assert ratio < 6.0, f"duration embedding for {days}d is {ratio:.1f}x a code row"


def test_tied_readout_identity_and_gradient():
    model = tiny_model()
    assert "lm_head" not in dict(model.named_modules())
    weight = model.HF_model.get_input_embeddings().weight
    loss, _ = model(make_batch())
    loss.backward()
    assert weight is model.HF_model.get_input_embeddings().weight
    assert weight.grad is not None and weight.grad.isfinite().all() and weight.grad.abs().sum() > 0


def test_pad_target_is_excluded_from_loss():
    model = tiny_model()
    a = make_batch()
    b = make_batch()
    b.targets[:, :, 0] = ~b.targets[:, :, 0]
    loss_a, _ = model(a)
    loss_b, _ = model(b)
    torch.testing.assert_close(loss_a, loss_b)


def test_no_valid_elements_has_differentiable_zero_loss():
    model = tiny_model()
    batch = make_batch(q_mask=[[False] * 3] * 2)
    loss, out = model(batch)
    assert loss.item() == 0 and not out.valid_mask.any()
    loss.backward()


def test_k_one_and_legacy_starts():
    model = tiny_model()
    explicit = make_batch(n_windows=1)
    legacy = make_batch(n_windows=1, starts=False)
    assert explicit.condition_codes.shape == explicit.condition_answers.shape == (2, 0)
    assert model._query_tokens(explicit).shape[1] == 1
    _, a = model(explicit)
    _, b = model(legacy)
    torch.testing.assert_close(a.logits, b.logits)


def test_exactly_one_start_field_absent_raises():
    model = tiny_model()
    batch = make_batch()
    batch.q_start_codes = None
    with pytest.raises(ValueError, match="given together"):
        model(batch)


def test_validation_errors_without_dataclass_reconstruction():
    model = tiny_model()
    too_many = make_batch(n_windows=5)
    # Extend every K-shaped tensor consistently after construction.
    too_many.q_durations = torch.zeros(2, 6)
    too_many.q_bound_codes = torch.zeros(2, 6, dtype=torch.long)
    too_many.q_start_durations = torch.zeros(2, 6)
    too_many.q_start_codes = torch.zeros(2, 6, dtype=torch.long)
    too_many.q_mask = torch.ones(2, 6, dtype=torch.bool)
    too_many.targets = torch.zeros(2, 6, VOCAB, dtype=torch.bool)
    too_many.condition_codes = torch.ones(2, 5, dtype=torch.long)
    too_many.condition_answers = torch.zeros(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="max_windows"):
        model(too_many)

    wrong_vocab = make_batch()
    wrong_vocab.targets = torch.zeros(2, 3, VOCAB - 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="vocabulary width"):
        model(wrong_vocab)

    short_positions = tiny_model(config_overrides={"max_position_embeddings": 8})
    with pytest.raises(ValueError, match="max_position_embeddings"):
        short_positions(make_batch())


def test_rope_time_pair_and_future_starts_do_not_advance_time():
    model = tiny_model(use_rope_time=True)
    with pytest.raises(ValueError, match="time_pos_ids"):
        model(make_batch())
    batch = make_batch(time_pos_ids=torch.tensor([[0, 12, 24, 36], [0, 5, 0, 0]]))
    n_patient = (batch.code != 0).sum(1)
    query_positions = n_patient[:, None] + torch.arange(7)[None, :]
    pos = model._position_ids(batch, n_patient, query_positions, 11)
    assert pos[0, 4:11].tolist() == [36] * 7
    assert pos[1, 2:9].tolist() == [5] * 7

    off = tiny_model(use_rope_time=False)
    with pytest.raises(ValueError, match="strip_delta_tokens"):
        off(batch)


def _save_checkpoint(module, path):
    import lightning as L

    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )


def test_lightning_predict_and_checkpoint_round_trip(tmp_path):
    from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule

    model = tiny_model()
    module = ConditionalMultitaskLightningModule(model=model, optimizer=partial(torch.optim.AdamW, lr=1e-4))
    assert all(not metrics for metrics in module.metrics.values())
    # Prediction scores QuerySeq rows (``MultitaskEvalBatch``); the loop contracts themselves are
    # covered in ``tests/multitask/test_conditional_multitask_lightning.py``.
    prediction = module.predict_step(make_eval_batch())
    assert set(prediction) == {"probs", "labels", "scored_codes"}
    assert all(t.shape == (2,) for t in prediction.values())
    ckpt = tmp_path / "model.ckpt"
    _save_checkpoint(module, ckpt)
    loaded = ConditionalMultitaskLightningModule.load_from_checkpoint(str(ckpt))
    assert isinstance(loaded.model, ConditionalMultitaskARModel)
    torch.testing.assert_close(loaded.model.code_bias, model.code_bias)

    bad = torch.load(ckpt, weights_only=False)
    bad["hyper_parameters"]["model"] = dict(bad["hyper_parameters"]["model"])
    bad["hyper_parameters"]["model"].pop("architecture")
    bad_ckpt = tmp_path / "missing-architecture.ckpt"
    torch.save(bad, bad_ckpt)
    with pytest.raises(KeyError, match="architecture"):
        ConditionalMultitaskLightningModule.load_from_checkpoint(str(bad_ckpt))


def test_configs_and_position_budget():
    from every_query.train.train import CONFIGS, required_position_embeddings

    for name in (
        "conditional_multitask_ar_config.yaml",
        "_demo_train_conditional_multitask_ar.yaml",
    ):
        cfg = yaml.safe_load((Path(CONFIGS) / name).read_text())
        dm_cfg = cfg["datamodule"]
        # Issue #30: the split datamodule pins the training dataset class itself, keeps the training
        # labels under config.task_labels_dir, and takes the (optional) QuerySeq grid root separately.
        assert dm_cfg["_target_"].endswith("ConditionalMultitaskDataModule")
        assert "data_class" not in dm_cfg
        assert dm_cfg["config"]["task_labels_dir"] == "???"
        assert dm_cfg["eval_tasks_dir"] is None
        assert dm_cfg["max_windows"] == "${lightning_module.model.max_windows}"
        # The ontology is set once, on the model; the datamodule's copy (for the evaluation adapter)
        # interpolates from it, as the scalar configs' does.
        assert dm_cfg["dataset_kwargs"]["ontology_dir"] == "${lightning_module.model.ontology_dir}"
        assert cfg["lightning_module"]["model"]["ontology_dir"] is None
        # train.py fills this in from the cohort's codes.parquet; shipping it as an explicit null is
        # what makes the key present for that assignment and inert for a run without an ontology.
        assert cfg["lightning_module"]["model"]["cohort_vocab_fingerprint"] is None
        assert dm_cfg["dataset_kwargs"]["expected_vocab_size"].endswith("config_overrides.vocab_size}")
        assert cfg["lightning_module"]["model"]["max_windows"] == 5
    model_cfg = OmegaConf.create(
        {
            "_target_": "every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel",
            "max_windows": 5,
        }
    )
    assert required_position_embeddings(model_cfg, 256) == 271


def test_lazy_package_exports_do_not_cycle():
    from every_query.model import ConditionalMultitaskARModel as Exported

    assert Exported is ConditionalMultitaskARModel


# ---------------------------------------------------------------------------
# Issue #28: target-only scoring must equal the gathered full-vocabulary logits
# ---------------------------------------------------------------------------


def _gather_final(logits: torch.Tensor, q_mask: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    last = q_mask.sum(dim=1) - 1
    return logits[torch.arange(logits.shape[0]), last, codes]


@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
def test_score_final_query_matches_gathered_full_vocab_logits(autocast):
    """``score_final_query`` == ``forward``'s logits at ``[b, last_b, code_b]`` for several scored codes and
    several real lengths in one padded batch (the numerical regression of #28)."""
    model = tiny_model()
    q_mask = [[True, True, True], [True, False, False], [True, True, False], [True, True, True]]
    code = [[2, 3, 4, 5], [7, 8, 0, 0], [9, 10, 11, 0], [12, 13, 14, 15]]
    batch = make_batch(code=code, q_mask=q_mask)
    scored = torch.tensor([2, 17, 5, 30], dtype=torch.long)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
        _, out = model(batch)
        target_only = model.score_final_query(batch, scored)
    expected = _gather_final(out.logits, batch.q_mask, scored)
    assert target_only.shape == (4,) and target_only.dtype == torch.float32
    torch.testing.assert_close(target_only, expected, atol=1e-5, rtol=1e-5)
    # Different codes and different last windows really are being compared, not one repeated value.
    assert len({round(v, 4) for v in target_only.tolist()}) > 1


def test_score_final_query_reads_the_last_real_window_not_the_padded_width():
    """Padding a 1-window row out to K=3 must not move which window is scored."""
    model = tiny_model()
    padded = make_batch(n_windows=3, q_mask=[[True, False, False]] * 2)
    alone = make_batch(n_windows=1)
    scored = torch.tensor([4, 6], dtype=torch.long)
    torch.testing.assert_close(
        model.score_final_query(padded, scored), model.score_final_query(alone, scored), atol=1e-5, rtol=1e-5
    )


def test_score_final_query_never_reads_targets_or_builds_dense_logits(monkeypatch):
    model = tiny_model()
    batch = make_batch()
    batch.targets = None  # an evaluation batch has no dense targets
    calls = []
    real_matmul = torch.Tensor.__matmul__

    def spy(a, b):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return real_matmul(a, b)

    monkeypatch.setattr(torch.Tensor, "__matmul__", spy)
    out = model.score_final_query(batch, torch.tensor([2, 3]))
    assert out.shape == (2,)
    assert not any(shape[-1] == VOCAB for _, shape in calls), "no (.., V) projection may be built"


def test_score_final_query_validates_codes_and_mask():
    model = tiny_model()
    batch = make_batch()
    with pytest.raises(ValueError, match="never be PAD"):
        model.score_final_query(batch, torch.tensor([0, 2]))
    with pytest.raises(ValueError, match=r"lie in \[0, 31\)"):
        model.score_final_query(batch, torch.tensor([2, VOCAB]))
    with pytest.raises(ValueError, match="shape"):
        model.score_final_query(batch, torch.tensor([[2, 3]]))
    with pytest.raises(ValueError, match="int64"):
        model.score_final_query(batch, torch.tensor([2, 3], dtype=torch.int32))
    with pytest.raises(ValueError, match="prefix mask"):
        model.score_final_query(make_batch(q_mask=[[True, False, True]] * 2), torch.tensor([2, 3]))
    with pytest.raises(ValueError, match="at least one real window"):
        model.score_final_query(make_batch(q_mask=[[False, False, False]] * 2), torch.tensor([2, 3]))


def test_training_forward_is_the_window_hidden_states_projection():
    """The all-vocabulary training forward is unchanged by the refactor: it is exactly
    ``window_hidden_states`` followed by the tied projection, bias, PAD/q_mask masking and BCE."""
    model = tiny_model()
    batch = make_batch(q_mask=[[True, True, False]] * 2)
    calls = []
    real = model.window_hidden_states

    def spy(b):
        calls.append(b)
        return real(b)

    model.window_hidden_states = spy
    loss, out = model(batch)
    assert calls == [batch], "forward must read its hidden states through window_hidden_states"

    hidden = real(batch)
    emb = model.HF_model.get_input_embeddings().weight
    logits = hidden.float() @ emb.float().T + model.code_bias.float()
    valid = batch.q_mask.unsqueeze(-1) & (torch.arange(VOCAB) != 0).view(1, 1, -1)
    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.targets.float(), reduction="none"
    )
    expected_loss = (per_element * valid).sum() / valid.sum().clamp_min(1)
    torch.testing.assert_close(out.logits, logits)
    assert torch.equal(out.valid_mask, valid.expand(2, 3, -1))
    torch.testing.assert_close(loss, expected_loss)


# ---------------------------------------------------------------------------
# Ontology: derived-at-load ancestor targets, the mixed readout, both widths
# ---------------------------------------------------------------------------

# Leaf names for indices 1..VOCAB-1: a dual-role name (``A//B`` is a leaf and the prefix of two
# leaves, so it gains ``A//B//ANY``), a leaf with no ancestor at all (``E``), a multi-parent leaf
# (``F//G`` under ``F`` by prefix and under ``P//Q`` by ``parent_codes``), and a big ``Z`` family.
_LEAF_NAMES = ["A//B//C", "A//B//D", "A//B", "E", "F//G"] + [f"Z//{i}" for i in range(6, VOCAB)]
_DECLARED_PARENTS = {"F//G": ["P//Q"]}


def write_tiny_ontology(root: Path, names: list[str] = _LEAF_NAMES) -> tuple[Path, int]:
    """Write the three ``EQ_build_ontology`` artifacts for ``names`` (leaf ids 1..VOCAB-1).

    ``names`` defaults to ``_LEAF_NAMES``; passing it with two entries swapped writes a *same-width*
    ontology of the same codes at a permuted numbering.  Overwrites in place, as re-running
    ``EQ_build_ontology`` into the same directory does.
    """
    frame = pl.DataFrame(
        {
            "code": names,
            "code/vocab_index": list(range(1, VOCAB)),
            "parent_codes": [_DECLARED_PARENTS.get(c) for c in names],
        }
    )
    nodes, mix = build_ontology(frame)
    root.mkdir(parents=True, exist_ok=True)
    nodes.write_parquet(root / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(root / EMBEDDING_MIX_FILE)
    build_event_to_query_nodes(nodes, mix).write_parquet(root / EVENT_TO_QUERY_NODES_FILE)
    return root, extended_vocab_size(root)


def ontology_model(tmp_path: Path, **kwargs) -> tuple[ConditionalMultitaskARModel, Path, int]:
    onto, v_ext = write_tiny_ontology(tmp_path / "ontology")
    overrides = dict(kwargs.pop("config_overrides", {}), vocab_size=v_ext)
    return tiny_model(ontology_dir=str(onto), config_overrides=overrides, **kwargs), onto, v_ext


def _node_ids(onto: Path) -> dict[str, int]:
    nodes = load_nodes(onto)
    return dict(zip(nodes["node_name"].to_list(), nodes["token_id"].to_list(), strict=True))


def test_ontology_dir_installs_the_mixed_table_and_exposes_both_widths(tmp_path):
    """``ontology_dir`` is a supported option: the input table is the ancestor-mixed one, the model reports
    the cohort width beside the table width, and the closure rides in non-persistent buffers."""
    model, onto, v_ext = ontology_model(tmp_path)
    assert v_ext > VOCAB, "the fixture ontology must mint ancestor nodes"
    assert isinstance(model.HF_model.get_input_embeddings(), OntologyEmbedding)
    assert model.vocab_size == v_ext and model.base_vocab_size == VOCAB and model.has_ontology
    assert model.code_bias.shape == (v_ext,)
    assert model.hparams["ontology_dir"] == str(onto)
    assert not any(k.startswith("closure_") for k in model.state_dict())
    ids = _node_ids(onto)
    assert {"A", "A//B//ANY", "F", "P", "P//Q", "Z"} <= {n for n, i in ids.items() if i >= VOCAB}
    closure = model._closure()
    pairs = set(zip(closure.leaf_ids.tolist(), closure.ancestor_ids.tolist(), strict=True))
    assert (ids["A//B//C"], ids["A"]) in pairs and (ids["A//B"], ids["A//B//ANY"]) in pairs
    assert (ids["F//G"], ids["P//Q"]) in pairs and (ids["F//G"], ids["F"]) in pairs
    assert not any(leaf == ids["E"] for leaf, _ in pairs)
    assert all(leaf < VOCAB <= ancestor for leaf, ancestor in pairs)

    # No ontology: one width, no wrapper, no buffers.
    plain = tiny_model()
    assert plain.base_vocab_size == plain.vocab_size == VOCAB and not plain.has_ontology
    assert not isinstance(plain.HF_model.get_input_embeddings(), OntologyEmbedding)

    # The table must be sized to V_ext (train.py's job); the cohort width is refused loudly.
    with pytest.raises(ValueError, match="V_ext"):
        tiny_model(ontology_dir=str(onto))


def test_forward_with_leaf_targets_equals_forward_with_extended_targets(tmp_path):
    """Leaf-only ``(B, K, V)`` targets are widened in ``forward`` to exactly what a pre-derived ``(B, K,
    V_ext)`` batch gives: same loss, same logits, same mask."""
    model, onto, v_ext = ontology_model(tmp_path)
    ids = _node_ids(onto)
    leaf = make_batch()
    assert leaf.targets.shape[-1] == VOCAB
    wide_targets = derive_ancestor_targets(leaf.targets, load_closure_index(onto, VOCAB))
    assert wide_targets.shape == (2, 3, v_ext)
    # ``make_batch`` sets leaf 2 (``A//B//D``) true everywhere, so both of its ancestors derive true.
    assert wide_targets[..., ids["A"]].all() and wide_targets[..., ids["A//B//ANY"]].all()
    assert not wide_targets[..., ids["Z"]].any() and not wide_targets[..., ids["P//Q"]].any()

    wide = make_batch()
    wide.targets = wide_targets
    loss_leaf, out_leaf = model(leaf)
    loss_wide, out_wide = model(wide)
    torch.testing.assert_close(loss_leaf, loss_wide)
    torch.testing.assert_close(out_leaf.logits, out_wide.logits)
    assert torch.equal(out_leaf.valid_mask, out_wide.valid_mask)
    assert out_leaf.logits.shape == out_leaf.valid_mask.shape == (2, 3, v_ext)
    assert not out_leaf.valid_mask[:, :, 0].any()
    # The derivation changes the loss: an ancestor column that is true costs more than one left false.
    unrelated = make_batch()
    unrelated.targets = torch.cat([leaf.targets, torch.zeros(2, 3, v_ext - VOCAB, dtype=torch.bool)], -1)
    assert not torch.allclose(model(unrelated)[0], loss_leaf)

    # Any other width is neither a leaf batch nor a full batch.
    for width in (VOCAB - 1, VOCAB + 1, v_ext + 1):
        bad = make_batch()
        bad.targets = torch.zeros(2, 3, width, dtype=torch.bool)
        with pytest.raises(ValueError, match="vocabulary width"):
            model(bad)


def test_tied_readout_uses_the_mixed_table_under_an_ontology(tmp_path):
    """Both projections read the **effective** table ``A @ W`` - the rows the input side sees - not the
    raw parameter, and the gradient reaches raw ancestor rows through the mix."""
    model, _, v_ext = ontology_model(tmp_path)
    embedding = model.HF_model.get_input_embeddings()
    raw = embedding.weight
    assert raw.shape[0] == v_ext
    mixed = torch.sparse.mm(embedding.mix, raw.detach())
    assert not torch.allclose(mixed, raw), "the mix must actually move the ancestor rows"
    batch = make_batch()
    hidden = model.window_hidden_states(batch)
    loss, out = model(batch)
    expected = hidden.float() @ mixed.float().T + model.code_bias.float()
    torch.testing.assert_close(out.logits, expected)
    assert not torch.allclose(out.logits, hidden.float() @ raw.detach().float().T + model.code_bias.float())

    loss.backward()
    assert raw.grad is not None and raw.grad.isfinite().all()
    assert raw.grad[VOCAB:].abs().sum() > 0, "ancestor rows must receive gradient through the mix"
    assert raw.grad[1:VOCAB].abs().sum() > 0


@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
def test_score_final_query_on_an_ancestor_code_matches_the_dense_forward(tmp_path, autocast):
    model, onto, v_ext = ontology_model(tmp_path)
    ids = _node_ids(onto)
    q_mask = [[True, True, True], [True, False, False], [True, True, False], [True, True, True]]
    code = [[2, 3, 4, 5], [7, 8, 0, 0], [9, 10, 11, 0], [12, 13, 14, 15]]
    batch = make_batch(code=code, q_mask=q_mask)
    scored = torch.tensor([ids["A"], ids["Z"], 5, v_ext - 1], dtype=torch.long)
    assert (scored[[0, 1, 3]] >= VOCAB).all()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
        _, out = model(batch)
        target_only = model.score_final_query(batch, scored)
    torch.testing.assert_close(
        target_only, _gather_final(out.logits, batch.q_mask, scored), atol=1e-5, rtol=1e-5
    )
    assert len({round(v, 4) for v in target_only.tolist()}) > 1
    with pytest.raises(ValueError, match=rf"lie in \[0, {v_ext}\)"):
        model.score_final_query(batch, torch.tensor([2, v_ext, 3, 4]))


def test_score_final_query_shares_one_mixed_table_per_pass(tmp_path):
    """``score_final_query`` never goes through ``forward``, so the per-forward cache the wrapper's pre-hook
    clears would otherwise go stale: the mixed table must be recomputed for every pass and read the
    current weights."""
    model, onto, _ = ontology_model(tmp_path)
    ids = _node_ids(onto)
    batch = make_batch()
    scored = torch.tensor([ids["A"], 2], dtype=torch.long)
    before = model.score_final_query(batch, scored)
    # ``A``'s raw row is a component of its own mixed row and of every ``A//...`` leaf's.
    with torch.no_grad():
        model.HF_model.get_input_embeddings().weight[ids["A"]] += 1.0
    after = model.score_final_query(batch, scored)
    assert not torch.allclose(before, after), "a stale cached table would not see the weight change"
    # And the dense forward, which goes through the pre-hook, agrees with the direct path afterwards.
    _, out = model(batch)
    torch.testing.assert_close(after, _gather_final(out.logits, batch.q_mask, scored), atol=1e-5, rtol=1e-5)


def test_checkpoint_round_trip_with_an_ontology(tmp_path):
    from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule

    model, onto, v_ext = ontology_model(tmp_path)
    module = ConditionalMultitaskLightningModule(model=model, optimizer=partial(torch.optim.AdamW, lr=1e-4))
    ckpt = tmp_path / "onto.ckpt"
    _save_checkpoint(module, ckpt)
    loaded = ConditionalMultitaskLightningModule.load_from_checkpoint(str(ckpt))
    assert loaded.model.ontology_dir == str(onto)
    assert loaded.model.vocab_size == v_ext and loaded.model.base_vocab_size == VOCAB
    assert isinstance(loaded.model.HF_model.get_input_embeddings(), OntologyEmbedding)
    loaded.model.eval()
    batch = make_batch()
    loss, out = model(batch)
    loss_loaded, out_loaded = loaded.model(batch)
    torch.testing.assert_close(loss, loss_loaded)
    torch.testing.assert_close(out.logits, out_loaded.logits)
    ids = _node_ids(onto)
    scored = torch.tensor([ids["A"], ids["Z"]], dtype=torch.long)
    torch.testing.assert_close(
        model.score_final_query(batch, scored), loaded.model.score_final_query(batch, scored)
    )


def test_cohort_vocab_fingerprint_guards_the_ontology_and_round_trips(tmp_path):
    """``cohort_vocab_fingerprint`` (``train.py`` fills it from the cohort's ``codes.parquet``) turns the
    model's own closure load into an identity check: the tiny ontology passes against its cohort's fingerprint
    and is refused against a same-width cohort with two codes swapped, which every width check accepts.

    It is a hyperparameter, so a reloaded checkpoint re-runs the check against whatever now sits
    at ``ontology_dir``.
    """
    from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
    from every_query.utils.digest import vocab_fingerprint

    cohort = dict(zip(_LEAF_NAMES, range(1, VOCAB), strict=True))
    fingerprint = vocab_fingerprint(cohort)
    model, onto, v_ext = ontology_model(tmp_path, cohort_vocab_fingerprint=fingerprint)
    assert model.cohort_vocab_fingerprint == model.hparams["cohort_vocab_fingerprint"] == fingerprint
    assert model.vocab_size == v_ext and model.base_vocab_size == VOCAB
    plain = tiny_model()
    assert plain.cohort_vocab_fingerprint is None and plain.hparams["cohort_vocab_fingerprint"] is None
    # Without an ontology the fingerprint is only recorded (EQ_predict_multitask checks the cohort against
    # it).
    recorded = tiny_model(cohort_vocab_fingerprint="not-checked-here")
    assert (
        recorded.cohort_vocab_fingerprint
        == recorded.hparams["cohort_vocab_fingerprint"]
        == "not-checked-here"
    )

    # The cohort with two codes swapped has the same V; only the fingerprint tells the model it is not
    # this one.
    swapped_names = list(_LEAF_NAMES)
    a, b = swapped_names.index("A//B//C"), swapped_names.index("E")
    swapped_names[a], swapped_names[b] = swapped_names[b], swapped_names[a]
    swapped = dict(zip(swapped_names, range(1, VOCAB), strict=True))
    assert vocab_fingerprint(swapped) != fingerprint
    with pytest.raises(ValueError, match=r"different codes\.parquet than this cohort.*observed nodes digest"):
        tiny_model(
            ontology_dir=str(onto),
            config_overrides={"vocab_size": v_ext},
            cohort_vocab_fingerprint=vocab_fingerprint(swapped),
        )

    module = ConditionalMultitaskLightningModule(model=model, optimizer=partial(torch.optim.AdamW, lr=1e-4))
    ckpt = tmp_path / "fingerprinted.ckpt"
    _save_checkpoint(module, ckpt)
    loaded = ConditionalMultitaskLightningModule.load_from_checkpoint(str(ckpt))
    assert loaded.model.cohort_vocab_fingerprint == fingerprint
    batch = make_batch()
    torch.testing.assert_close(model(batch)[1].logits, loaded.model.eval()(batch)[1].logits)

    # Rebuild the ontology in place from the swapped numbering (same V_ext, so the table still fits): the
    # checkpoint now refuses to load, where a width-only check would have paired every leaf with the wrong
    # rows.
    _, v_ext_swapped = write_tiny_ontology(onto, swapped_names)
    assert v_ext_swapped == v_ext
    with pytest.raises(ValueError, match=r"different codes\.parquet than this cohort"):
        ConditionalMultitaskLightningModule.load_from_checkpoint(str(ckpt))
