"""Model perturbation tests for EveryQuery.

Verifies internal operations of ``EveryQueryModel._forward`` and ``_hf_inputs``
by constructing a baseline from ``sample_batch``, perturbing exactly one input
dimension, and asserting the expected output change (or invariance).
"""

import copy

import torch


class TestQueryTokenPositionAffectsQueryEmbed:
    """Changing the query token at position 0 must change ``query_embed``.

    The model extracts ``query_embed = embeddings[:, 0, :]`` from the
    backbone output.  The tests verify three properties:

    1. ``query_embed`` is literally ``last_hidden_state[:, 0, :]`` (extraction
       site is position 0).
    2. Replacing one sample's query token at ``code[row, 0]`` changes that
       sample's ``query_embed``.
    3. The other sample's ``query_embed`` is unaffected (batch independence).
    """

    @torch.no_grad()
    def test_query_embed_is_hidden_state_at_position_zero(self, demo_model, sample_batch):
        """``query_embed`` must equal ``last_hidden_state[:, 0, :]``."""
        _, out = demo_model._forward(sample_batch)
        assert out.last_hidden_state is not None, "do_demo=True should retain last_hidden_state"
        assert torch.equal(out.query_embed, out.last_hidden_state[:, 0, :])

    @torch.no_grad()
    def test_different_query_token_changes_query_embed(self, demo_model, sample_batch):
        _, baseline_out = demo_model._forward(sample_batch)

        perturbed = copy.deepcopy(sample_batch)
        target_row = 0
        original_token = perturbed.code[target_row, 0].item()

        assert original_token != perturbed.PAD_INDEX, (
            "precondition: query token must not be PAD_INDEX, otherwise swapping "
            "it also flips the attention mask (two-variable perturbation)"
        )

        candidate = 1
        while candidate == original_token or candidate == perturbed.PAD_INDEX:
            candidate += 1
        perturbed.code[target_row, 0] = candidate

        assert perturbed.code[target_row, 0].item() != original_token, "precondition: token was replaced"

        _, perturbed_out = demo_model._forward(perturbed)

        assert not torch.equal(
            baseline_out.query_embed[target_row], perturbed_out.query_embed[target_row]
        ), "query_embed for the perturbed sample should change"

        other_rows = [i for i in range(sample_batch.batch_size) if i != target_row]
        for r in other_rows:
            assert torch.equal(
                baseline_out.query_embed[r], perturbed_out.query_embed[r]
            ), f"query_embed for unperturbed sample {r} should be unchanged"


class TestUncensoredSamplesContributeToOccursLoss:
    """Flipping an uncensored sample's ``occurs`` label must change ``occurs_loss``.

    ``_forward`` computes ``occurs_loss`` with ``mask=~batch.censor``, so
    uncensored samples (``censor=False``) are included in the BCE
    computation.  Changing their target must therefore change the loss.
    """

    @torch.no_grad()
    def test_flipping_uncensored_occurs_changes_loss(self, demo_model, sample_batch):
        uncensored_mask = ~sample_batch.censor
        assert uncensored_mask.any(), "sample_batch must contain at least one uncensored sample"

        _, baseline_out = demo_model._forward(sample_batch)
        assert baseline_out.occurs_loss.isfinite(), (
            "Precondition: baseline occurs_loss must be finite "
            "(NaN would make the inequality assertion vacuously true)"
        )

        perturbed = copy.deepcopy(sample_batch)
        idx = uncensored_mask.nonzero(as_tuple=True)[0][0]
        perturbed.occurs[idx] = 1 - perturbed.occurs[idx]

        _, perturbed_out = demo_model._forward(perturbed)

        assert not torch.equal(baseline_out.occurs_loss, perturbed_out.occurs_loss), (
            "occurs_loss should change when an uncensored sample's occurs label is flipped"
        )


class TestCensoredSamplesExcludedFromOccursLoss:
    """Flipping a censored sample's ``occurs`` label must not change ``occurs_loss``.

    ``_forward`` computes ``occurs_loss`` with ``mask=~batch.censor``, so any
    sample where ``censor`` is ``True`` is excluded from the BCE computation.
    """

    @torch.no_grad()
    def test_flipping_censored_occurs_label_preserves_occurs_loss(self, demo_model, sample_batch):
        censored_mask = sample_batch.censor
        assert censored_mask.any(), (
            "Precondition: sample_batch must contain at least one censored sample"
        )
        assert (~censored_mask).any(), (
            "Precondition: sample_batch must contain at least one uncensored sample "
            "(otherwise occurs_loss is NaN from empty-tensor BCE and torch.equal(nan, nan) is False)"
        )

        _, baseline_out = demo_model._forward(sample_batch)
        assert baseline_out.occurs_loss.isfinite(), (
            "Precondition: baseline occurs_loss must be finite for equality check to be meaningful"
        )

        perturbed = copy.deepcopy(sample_batch)
        perturbed.occurs[censored_mask] = 1 - perturbed.occurs[censored_mask]

        _, perturbed_out = demo_model._forward(perturbed)

        assert torch.equal(baseline_out.occurs_loss, perturbed_out.occurs_loss), (
            "occurs_loss must not change when only censored samples' occurs labels are flipped"
        )


class TestDurationPathSwitching:
    """Setting ``duration_days=None`` must switch ``_hf_inputs`` from the
    ``inputs_embeds`` path to the ``input_ids`` path.

    When ``duration_days`` is present the method embeds a duration token and
    returns ``inputs_embeds`` with the sequence expanded by one position (the
    injected duration embedding); when ``None`` it falls back to raw token IDs
    via ``input_ids`` at the original sequence length.
    """

    @torch.no_grad()
    def test_with_duration_returns_inputs_embeds(self, demo_model, sample_batch):
        assert sample_batch.duration_days is not None, (
            "Precondition: sample_batch must have duration_days for this test"
        )
        seq_len = sample_batch.code.shape[1]
        hf_out = demo_model._hf_inputs(sample_batch)

        assert "inputs_embeds" in hf_out, (
            "_hf_inputs should return 'inputs_embeds' when duration_days is present"
        )
        assert "input_ids" not in hf_out, (
            "_hf_inputs should NOT return 'input_ids' when duration_days is present"
        )
        assert hf_out["inputs_embeds"].shape[1] == seq_len + 1, (
            "inputs_embeds seq dim should be seq_len + 1 (duration token inserted)"
        )
        assert hf_out["attention_mask"].shape[1] == seq_len + 1, (
            "attention_mask seq dim should match inputs_embeds"
        )

    @torch.no_grad()
    def test_without_duration_returns_input_ids(self, demo_model, sample_batch):
        perturbed = copy.deepcopy(sample_batch)
        perturbed.duration_days = None
        seq_len = perturbed.code.shape[1]

        hf_out = demo_model._hf_inputs(perturbed)

        assert "input_ids" in hf_out, (
            "_hf_inputs should return 'input_ids' when duration_days is None"
        )
        assert "inputs_embeds" not in hf_out, (
            "_hf_inputs should NOT return 'inputs_embeds' when duration_days is None"
        )
        assert hf_out["input_ids"].shape[1] == seq_len, (
            "input_ids seq dim should equal the original seq_len"
        )
        assert hf_out["attention_mask"].shape[1] == seq_len, (
            "attention_mask seq dim should match input_ids"
        )

    @torch.no_grad()
    def test_duration_path_inserts_attended_position(self, demo_model, sample_batch):
        """The duration path must insert an always-attended position at index 1.

        Compares the attention masks from both paths to prove that
        position 1 was genuinely injected rather than inherited from the
        original codes (which may already be non-padding at that index).
        """
        assert sample_batch.duration_days is not None, (
            "Precondition: sample_batch must have duration_days for this test"
        )
        dur_out = demo_model._hf_inputs(sample_batch)

        no_dur_batch = copy.deepcopy(sample_batch)
        no_dur_batch.duration_days = None
        no_dur_out = demo_model._hf_inputs(no_dur_batch)

        dur_mask = dur_out["attention_mask"]
        no_dur_mask = no_dur_out["attention_mask"]

        assert torch.equal(dur_mask[:, 0], no_dur_mask[:, 0]), (
            "Position 0 (query token) should be identical in both paths"
        )
        assert torch.equal(dur_mask[:, 2:], no_dur_mask[:, 1:]), (
            "Positions 2+ in the duration mask should equal positions 1+ in "
            "the no-duration mask (proving the rest was shifted by one)"
        )
        assert dur_mask[:, 1].all(), (
            "The injected duration position at index 1 must be attended to for all samples"
        )


class TestPaddingTokenInvariance:
    """Padding positions must not affect ``query_embed`` or logits.

    ``_hf_inputs`` builds the attention mask as ``batch.code != batch.PAD_INDEX``
    so positions with ``PAD_INDEX`` are masked to zero.  Extending the batch with
    additional trailing padding must leave the model output unchanged.
    """

    @torch.no_grad()
    def test_extra_trailing_padding_does_not_change_output(self, demo_model, sample_batch):
        _, baseline_out = demo_model._forward(sample_batch)

        perturbed = copy.deepcopy(sample_batch)
        B, orig_seq_len = perturbed.code.shape
        extra = 4

        perturbed.code = torch.cat(
            [perturbed.code, torch.full((B, extra), perturbed.PAD_INDEX, dtype=perturbed.code.dtype)],
            dim=1,
        )

        assert perturbed.code.shape[1] == orig_seq_len + extra
        assert (perturbed.code[:, orig_seq_len:] == perturbed.PAD_INDEX).all(), (
            "Sanity: trailing positions must be PAD_INDEX so _hf_inputs masks them out"
        )

        _, perturbed_out = demo_model._forward(perturbed)

        assert torch.allclose(baseline_out.query_embed, perturbed_out.query_embed, atol=1e-5), (
            "query_embed should not change when extra trailing padding is added"
        )
        assert torch.allclose(baseline_out.censor_logits, perturbed_out.censor_logits, atol=1e-5), (
            "censor_logits should not change when extra trailing padding is added"
        )
        assert torch.allclose(baseline_out.occurs_logits, perturbed_out.occurs_logits, atol=1e-5), (
            "occurs_logits should not change when extra trailing padding is added"
        )

    @torch.no_grad()
    def test_extra_trailing_non_pad_tokens_do_change_output(self, demo_model, sample_batch):
        """Negative control: attended (non-PAD) trailing tokens must change output.

        Proves the invariance in the companion test is specifically due to the
        attention mask zeroing out PAD_INDEX positions, not an artefact of the
        token embedding values or model insensitivity to extra positions.
        """
        _, baseline_out = demo_model._forward(sample_batch)

        control = copy.deepcopy(sample_batch)
        B = control.code.shape[0]
        extra = 4
        non_pad_token = 5
        assert non_pad_token != control.PAD_INDEX

        control.code = torch.cat(
            [control.code, torch.full((B, extra), non_pad_token, dtype=control.code.dtype)],
            dim=1,
        )

        _, control_out = demo_model._forward(control)

        assert not torch.allclose(baseline_out.query_embed, control_out.query_embed, atol=1e-5), (
            "Negative control: non-PAD trailing tokens should change query_embed, "
            "proving the companion test's invariance is due to attention masking"
        )


class TestDurationDaysValueAffectsOutput:
    """Scaling ``duration_days`` must change loss, query_embed, and logits.

    ``_hf_inputs`` normalises duration via ``duration_days / 365`` and feeds it
    through ``self.duration_embed``, so a multiplicative change propagates
    through the backbone into all downstream quantities.
    """

    @torch.no_grad()
    def test_scaling_duration_days_changes_output(self, demo_model, sample_batch):
        assert sample_batch.duration_days is not None, "fixture must provide duration_days"
        assert (sample_batch.duration_days != 0).all(), (
            "fixture duration_days must be non-zero so that *10 is a meaningful perturbation"
        )

        baseline_loss, baseline_out = demo_model._forward(sample_batch)

        perturbed = copy.deepcopy(sample_batch)
        perturbed.duration_days = perturbed.duration_days * 10

        perturbed_loss, perturbed_out = demo_model._forward(perturbed)

        assert not torch.equal(baseline_out.query_embed, perturbed_out.query_embed), (
            "query_embed should change when duration_days is scaled"
        )
        assert not torch.equal(baseline_out.censor_logits, perturbed_out.censor_logits), (
            "censor_logits should change when duration_days is scaled"
        )
        assert not torch.equal(baseline_out.occurs_logits, perturbed_out.occurs_logits), (
            "occurs_logits should change when duration_days is scaled"
        )
        assert not torch.equal(baseline_loss, perturbed_loss), (
            "loss should change when duration_days is scaled"
        )


class TestLogitsIndependentOfTargets:
    """Flipping ``censor`` and ``occurs`` targets must not change logits.

    Logits are produced by MLP heads on ``query_embed``, which depends only on
    the input codes and embeddings fed through the backbone.  The target tensors
    ``batch.censor`` and ``batch.occurs`` participate only in the loss
    computation, so perturbing them should leave logits identical while changing
    the loss.
    """

    @torch.no_grad()
    def test_flipped_targets_preserve_logits(self, demo_model, sample_batch):
        assert sample_batch.batch_size > 0, "Precondition: batch must be non-empty"

        baseline_loss, baseline_out = demo_model._forward(sample_batch)

        perturbed = copy.deepcopy(sample_batch)
        perturbed.censor = ~perturbed.censor
        perturbed.occurs = 1 - perturbed.occurs

        assert not torch.equal(sample_batch.censor, perturbed.censor), (
            "Precondition: censor targets must actually differ after flip"
        )
        assert not torch.equal(sample_batch.occurs, perturbed.occurs), (
            "Precondition: occurs targets must actually differ after flip"
        )

        perturbed_loss, perturbed_out = demo_model._forward(perturbed)

        assert torch.equal(baseline_out.query_embed, perturbed_out.query_embed), (
            "query_embed should be unchanged when only targets are flipped"
        )
        assert torch.equal(baseline_out.censor_logits, perturbed_out.censor_logits), (
            "censor_logits should be unchanged when only targets are flipped"
        )
        assert torch.equal(baseline_out.occurs_logits, perturbed_out.occurs_logits), (
            "occurs_logits should be unchanged when only targets are flipped"
        )

    @torch.no_grad()
    def test_flipped_targets_change_loss(self, demo_model, sample_batch):
        baseline_loss, baseline_out = demo_model._forward(sample_batch)

        perturbed = copy.deepcopy(sample_batch)
        perturbed.censor = ~perturbed.censor
        perturbed.occurs = 1 - perturbed.occurs

        perturbed_loss, perturbed_out = demo_model._forward(perturbed)

        assert not torch.equal(baseline_out.censor_loss, perturbed_out.censor_loss), (
            "censor_loss should change when censor targets are flipped"
        )
        assert not torch.equal(baseline_loss, perturbed_loss), (
            "total loss should change when censor/occurs targets are flipped"
        )


class TestCrossLossTargetIndependence:
    """``censor_loss`` must be independent of ``batch.occurs``.

    ``censor_loss`` is computed from ``censor_logits`` and ``batch.censor`` only
    (``censor_loss = self._get_loss(censor_logits, batch.censor, mask=None)``).
    ``batch.occurs`` participates only in the ``occurs_loss`` path.  If a bug
    accidentally threads ``batch.occurs`` into the ``censor_loss`` computation,
    this test class catches it.

    The existing ``TestLogitsIndependentOfTargets`` flips *both* targets
    simultaneously and therefore cannot isolate this single-target property.
    """

    @torch.no_grad()
    def test_flipping_occurs_preserves_censor_loss(self, demo_model, sample_batch):
        """Perturbing only ``occurs`` must leave ``censor_logits`` and ``censor_loss`` bit-identical."""
        _, baseline_out = demo_model._forward(sample_batch)
        assert baseline_out.censor_loss.isfinite(), (
            "Precondition: baseline censor_loss must be finite for equality check"
        )

        perturbed = copy.deepcopy(sample_batch)
        perturbed.occurs = 1 - perturbed.occurs

        assert not torch.equal(sample_batch.occurs, perturbed.occurs), (
            "Precondition: occurs targets must actually differ after flip"
        )
        assert torch.equal(sample_batch.censor, perturbed.censor), (
            "Precondition: censor targets must be untouched"
        )

        _, perturbed_out = demo_model._forward(perturbed)

        assert torch.equal(baseline_out.censor_logits, perturbed_out.censor_logits), (
            "censor_logits must not change when only occurs targets are flipped "
            "(logits depend on input codes/embeddings, not target tensors)"
        )
        assert torch.equal(baseline_out.censor_loss, perturbed_out.censor_loss), (
            "censor_loss must not change when only occurs targets are flipped"
        )

    @torch.no_grad()
    def test_flipping_occurs_changes_occurs_loss(self, demo_model, sample_batch):
        """Negative control: flipping ``occurs`` must change ``occurs_loss``.

        Proves the occurs-flip perturbation is non-trivial — the model does use
        ``batch.occurs`` in at least the ``occurs_loss`` path.
        """
        assert (~sample_batch.censor).any(), (
            "Precondition: at least one uncensored sample is required "
            "for occurs_loss to be sensitive to occurs targets"
        )

        _, baseline_out = demo_model._forward(sample_batch)
        assert baseline_out.occurs_loss.isfinite(), (
            "Precondition: baseline occurs_loss must be finite"
        )

        perturbed = copy.deepcopy(sample_batch)
        perturbed.occurs = 1 - perturbed.occurs

        _, perturbed_out = demo_model._forward(perturbed)

        assert not torch.equal(baseline_out.occurs_loss, perturbed_out.occurs_loss), (
            "occurs_loss should change when occurs targets are flipped "
            "(negative control for censor_loss invariance)"
        )

    @torch.no_grad()
    def test_flipping_censor_changes_censor_loss(self, demo_model, sample_batch):
        """Negative control: flipping ``censor`` must change ``censor_loss``.

        Proves ``censor_loss`` is not trivially constant — it genuinely depends
        on ``batch.censor``, and the invariance to ``batch.occurs`` is a real
        isolation property, not an artefact of ``censor_loss`` being insensitive
        to all targets.

        Edge-case guard: with a two-sample batch and ``mean`` reduction,
        ``~censor`` permutes the targets.  The mean BCE is permutation-invariant
        when all logits are identical, so we assert logit diversity first.
        """
        _, baseline_out = demo_model._forward(sample_batch)
        assert baseline_out.censor_loss.isfinite(), (
            "Precondition: baseline censor_loss must be finite"
        )
        assert not torch.all(baseline_out.censor_logits == baseline_out.censor_logits[0]), (
            "Precondition: censor_logits must vary across samples; uniform logits "
            "make mean-reduced BCE invariant to target permutation in this 2-sample batch"
        )

        perturbed = copy.deepcopy(sample_batch)
        perturbed.censor = ~perturbed.censor

        assert not torch.equal(sample_batch.censor, perturbed.censor), (
            "Precondition: censor targets must actually differ after flip"
        )

        _, perturbed_out = demo_model._forward(perturbed)

        assert not torch.equal(baseline_out.censor_loss, perturbed_out.censor_loss), (
            "censor_loss must change when censor targets are flipped "
            "(proves censor_loss is not trivially constant)"
        )
