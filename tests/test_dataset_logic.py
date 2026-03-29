"""Dataset perturbation tests for EveryQuery.

Covers query prepending and collation logic inside ``EveryQueryPytorchDataset``,
specifically that ``_seeded_getitem`` prepends the encoded query token and that
the ``collate`` method correctly maps label fields to batch annotations.
"""

import copy

import torch

from every_query.dataset import EveryQueryBatch, EveryQueryPytorchDataset


class TestQueryTokenPrependedAtPosition0:
    """``_seeded_getitem`` must prepend the encoded query token at position 0.

    The method builds a ``QueryData`` from the encoded query string for the
    requested index, converts it to a JNRT, and concatenates it *before* the
    patient's dynamic data.  The first code in the resulting sequence should
    therefore equal ``encode_query(dataset.query[idx])``.
    """

    def test_first_code_is_encoded_query(self, demo_dataset):
        idx = 0
        item = demo_dataset._seeded_getitem(idx)
        dense = item["dynamic"].to_dense()
        first_code = int(dense["code"][0])

        expected = demo_dataset.encode_query(demo_dataset.query[idx])
        assert expected != EveryQueryBatch.PAD_INDEX, (
            "Precondition: query must encode to a real vocab entry, not PAD_INDEX — "
            "otherwise the equality check below is trivially satisfiable"
        )
        assert first_code == expected, (
            f"Position-0 code ({first_code}) should equal the encoded query "
            f"token ({expected}) for index {idx}"
        )

    def test_prepend_adds_one_code_and_preserves_original(self, demo_dataset):
        """The original patient codes must follow the query token unchanged."""
        import numpy as np
        from meds_torchdata import MEDSPytorchDataset

        idx = 0
        seed = 42

        base_item = MEDSPytorchDataset._seeded_getitem(demo_dataset, idx, seed)
        base_codes = base_item["dynamic"].to_dense()["code"]

        prepended_item = demo_dataset._seeded_getitem(idx, seed)
        prepended_codes = prepended_item["dynamic"].to_dense()["code"]

        assert len(prepended_codes) == len(base_codes) + 1, (
            f"Prepended sequence should be exactly 1 longer than base "
            f"({len(prepended_codes)} vs {len(base_codes)})"
        )
        np.testing.assert_array_equal(
            prepended_codes[1:],
            base_codes,
            err_msg="Original patient codes must be preserved after the query token",
        )

    def test_each_item_gets_its_own_query_token(self, demo_dataset):
        """Items with different query strings must get different position-0 codes."""
        idx_a, idx_b = 0, 1
        query_a = demo_dataset.query[idx_a]
        query_b = demo_dataset.query[idx_b]

        assert query_a != query_b, (
            "Precondition: indices 0 and 1 should have different query strings"
        )

        item_a = demo_dataset._seeded_getitem(idx_a)
        item_b = demo_dataset._seeded_getitem(idx_b)

        code_a = int(item_a["dynamic"].to_dense()["code"][0])
        code_b = int(item_b["dynamic"].to_dense()["code"][0])

        assert code_a == demo_dataset.encode_query(query_a)
        assert code_b == demo_dataset.encode_query(query_b)
        assert code_a != code_b, (
            f"Items with different query strings ({query_a!r}, {query_b!r}) "
            f"should get different position-0 codes ({code_a}, {code_b})"
        )


class TestCollateMapsBooleanValueToCensor:
    """``collate`` must map per-item ``boolean_value`` labels to ``batch.censor``.

    The child collate (line 410) sets ``out["censor"] = out[self.LABEL_COL]``
    where ``LABEL_COL`` is ``"boolean_value"``.  We verify the final
    ``batch.censor`` values against the per-item labels extracted *before*
    collation, not against ``batch.boolean_value`` (which is the same tensor
    object and would make the check tautological).
    """

    def test_censor_matches_per_item_boolean_labels(self, demo_dataset):
        items = [demo_dataset[0], demo_dataset[1]]

        expected = torch.tensor([bool(item[demo_dataset.LABEL_COL]) for item in items])

        batch = demo_dataset.collate(items)

        assert batch.censor is not None, "batch.censor should be populated when task labels exist"
        assert torch.equal(batch.censor, expected), (
            f"batch.censor ({batch.censor.tolist()}) should match the per-item "
            f"boolean_value labels ({expected.tolist()})"
        )

    def test_censor_reflects_mutated_item_labels(self, demo_dataset):
        """Flipping an item's boolean_value before collation must flip the
        corresponding censor entry, proving collate reads from the items."""
        items = [copy.deepcopy(demo_dataset[0]), copy.deepcopy(demo_dataset[1])]

        original_0 = bool(items[0][demo_dataset.LABEL_COL])
        items[0][demo_dataset.LABEL_COL] = not original_0

        expected = torch.tensor([not original_0, bool(items[1][demo_dataset.LABEL_COL])])

        batch = demo_dataset.collate(items)

        assert torch.equal(batch.censor, expected), (
            f"batch.censor ({batch.censor.tolist()}) should reflect the mutated "
            f"item labels ({expected.tolist()})"
        )

    def test_censor_is_bool_tensor(self, demo_dataset):
        items = [demo_dataset[0], demo_dataset[1]]
        batch = demo_dataset.collate(items)

        assert batch.censor.dtype == torch.bool, (
            f"Expected bool dtype for censor, got {batch.censor.dtype}"
        )


class TestDurationDaysPassthrough:
    """Collated batches must carry ``duration_days`` from the underlying label data.

    ``_seeded_getitem`` propagates ``duration_days`` when present, and ``collate``
    gathers the per-item values into a float tensor on the resulting
    ``EveryQueryBatch``.  The fixture ``demo_dataset`` is built with
    ``duration_days=30.0`` for every row.
    """

    def test_duration_days_present(self, demo_dataset: EveryQueryPytorchDataset):
        items = [demo_dataset[0], demo_dataset[1]]
        batch = demo_dataset.collate(items)

        assert batch.duration_days is not None, "duration_days should be present in the collated batch"

    def test_duration_days_matches_label_values(self, demo_dataset: EveryQueryPytorchDataset):
        indices = [0, 1]
        items = [demo_dataset[i] for i in indices]
        batch = demo_dataset.collate(items)

        expected = torch.tensor(
            [demo_dataset.duration_days[i] for i in indices], dtype=torch.float
        )
        torch.testing.assert_close(batch.duration_days, expected)

    def test_collate_reads_per_item_duration(self, demo_dataset: EveryQueryPytorchDataset):
        """Perturb item-level duration_days to distinct values so the test
        cannot pass if collate ignores the per-item values."""
        items = [copy.deepcopy(demo_dataset[0]), copy.deepcopy(demo_dataset[1])]
        items[0]["duration_days"] = 7.0
        items[1]["duration_days"] = 365.0

        batch = demo_dataset.collate(items)

        expected = torch.tensor([7.0, 365.0], dtype=torch.float)
        torch.testing.assert_close(batch.duration_days, expected)

    def test_duration_days_is_float_tensor(self, demo_dataset: EveryQueryPytorchDataset):
        items = [demo_dataset[0], demo_dataset[1]]
        batch = demo_dataset.collate(items)

        assert batch.duration_days.dtype == torch.float, (
            f"Expected float dtype, got {batch.duration_days.dtype}"
        )

    def test_duration_days_shape_matches_batch_size(self, demo_dataset: EveryQueryPytorchDataset):
        items = [demo_dataset[i] for i in range(min(3, len(demo_dataset)))]
        batch = demo_dataset.collate(items)

        assert batch.duration_days.shape == (len(items),), (
            f"Expected shape ({len(items)},), got {batch.duration_days.shape}"
        )


class TestDifferentQueryStringProducesDifferentIndex:
    """Items with distinct ``query`` strings must produce different encoded
    indices at position 0 of collated ``batch.code``.

    ``_seeded_getitem`` encodes the item's query string via ``encode_query``
    and prepends it as position 0 of the dynamic sequence.  Two items whose
    raw ``query`` values differ must therefore produce different token IDs at
    that position after collation.
    """

    def test_different_query_strings_yield_different_position_0(self, demo_dataset):
        idx_a = 0
        query_a = demo_dataset.query[idx_a]
        idx_b = None
        for i in range(1, len(demo_dataset)):
            if demo_dataset.query[i] != query_a:
                idx_b = i
                break

        assert idx_b is not None, (
            "Precondition: demo_dataset must contain at least two items with different query strings"
        )
        query_b = demo_dataset.query[idx_b]

        enc_a = demo_dataset.encode_query(query_a)
        enc_b = demo_dataset.encode_query(query_b)

        assert enc_a != EveryQueryBatch.PAD_INDEX, (
            f"Precondition: {query_a!r} must encode to a real vocab index, not PAD_INDEX"
        )
        assert enc_b != EveryQueryBatch.PAD_INDEX, (
            f"Precondition: {query_b!r} must encode to a real vocab index, not PAD_INDEX"
        )
        assert enc_a != enc_b, (
            f"encode_query should return different indices for different query strings "
            f"({query_a!r} -> {enc_a}, {query_b!r} -> {enc_b})"
        )

        items = [demo_dataset[idx_a], demo_dataset[idx_b]]
        batch = demo_dataset.collate(items)

        assert batch.code[0, 0].item() == enc_a, (
            f"Position 0 for sample 0 should carry encoded {query_a!r} ({enc_a}), "
            f"got {batch.code[0, 0].item()}"
        )
        assert batch.code[1, 0].item() == enc_b, (
            f"Position 0 for sample 1 should carry encoded {query_b!r} ({enc_b}), "
            f"got {batch.code[1, 0].item()}"
        )
