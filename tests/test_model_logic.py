"""Model perturbation tests for EveryQuery.

Verifies internal operations of ``EveryQueryModel._forward`` and ``_hf_inputs``
by constructing a baseline from ``sample_batch``, perturbing exactly one input
dimension, and asserting the expected output change (or invariance).
"""

import copy

import torch

from every_query.model import EveryQueryOutput


class TestQueryTokenPositionAffectsQueryEmbed:
    """Changing the query token at position 0 must change ``query_embed``.

    The model extracts ``query_embed = embeddings[:, 0, :]`` from the
    backbone output, so any change to the token at position 0 of
    ``batch.code`` should propagate into a different ``query_embed``.
    """

    @torch.no_grad()
    def test_different_query_token_changes_query_embed(self, demo_model, sample_batch):
        _, baseline_out = demo_model._forward(sample_batch)

        perturbed = copy.deepcopy(sample_batch)
        original_tokens = perturbed.code[:, 0].clone()

        # Pick a replacement token that differs from every original query token
        # and is not the PAD_INDEX (0). vocab_size is at least 100.
        candidate = 1
        while (original_tokens == candidate).any() or candidate == perturbed.PAD_INDEX:
            candidate += 1
        perturbed.code[:, 0] = candidate

        _, perturbed_out = demo_model._forward(perturbed)

        assert not torch.equal(baseline_out.query_embed, perturbed_out.query_embed), (
            "query_embed should change when the query token at position 0 is replaced"
        )
