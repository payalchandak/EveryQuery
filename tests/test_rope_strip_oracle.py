"""Differential test: `strip_delta_tokens` against a naive per-row implementation.

`strip_delta_tokens` compacts four parallel tensors at once, re-bases each row's clock to its
first surviving token, and *recomputes* rather than compacts ``time_delta_days``.  A
misalignment between any two of those outputs would not raise -- it would silently hand the
encoder a stream whose values belong to different tokens than its codes, corrupting every
training sequence while every shape assertion still passed.

The oracle below walks one row at a time in plain Python, straight from the docstring's stated
contract.  It shares no code path with the vectorised implementation.
"""

import random

import pytest
import torch

from every_query.data.rope_time import HOURS_PER_DAY, strip_delta_tokens

PAD = 0
DELTA_IDS = torch.tensor([90, 91, 92])


def _oracle_row(code, numeric_value, numeric_value_mask, time_delta_days, delta_ids, protect_first_n):
    """One row, by hand.

    Returns the five compacted lists (unpadded).
    """
    n = len(code)
    # Cumulative elapsed time is summed BEFORE the strip, or the deltas' own time is lost.
    cum, running = [], 0.0
    for j in range(n):
        running += time_delta_days[j]
        cum.append(running)

    kept = [j for j in range(n) if (j < protect_first_n or code[j] not in delta_ids) and code[j] != PAD]
    if not kept:
        return [], [], [], [], []

    base = cum[kept[0]]
    out_code = [code[j] for j in kept]
    out_val = [numeric_value[j] for j in kept]
    out_mask = [numeric_value_mask[j] for j in kept]
    out_pos = [round((cum[j] - base) * HOURS_PER_DAY) for j in kept]
    # "days since the previous SURVIVING token", not the original per-token delta.
    out_tdd = [0.0] + [cum[kept[k]] - cum[kept[k - 1]] for k in range(1, len(kept))]
    return out_code, out_val, out_mask, out_tdd, out_pos


def _random_batch(seed, b=4, n=12):
    rng = random.Random(seed)
    code, val, mask, tdd = [], [], [], []
    for _ in range(b):
        row_len = rng.randrange(1, n + 1)
        c, v, m, t = [], [], [], []
        for j in range(n):
            if j >= row_len:
                c.append(PAD)
                v.append(0.0)
                m.append(False)
                t.append(0.0)
                continue
            # A third delta tokens, a third valued events, a third plain events.
            roll = rng.random()
            if roll < 0.34:
                c.append(rng.choice([90, 91, 92]))
                v.append(0.0)
                m.append(False)
                t.append(round(rng.choice([0.25, 0.5, 1.0, 3.0, 30.0]), 4))
            else:
                c.append(rng.randrange(1, 20))
                valued = roll > 0.67
                v.append(round(rng.uniform(-3, 3), 4) if valued else 0.0)
                m.append(valued)
                t.append(0.0)
        code.append(c)
        val.append(v)
        mask.append(m)
        tdd.append(t)
    return (
        torch.tensor(code),
        torch.tensor(val, dtype=torch.float64),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(tdd, dtype=torch.float64),
    )


@pytest.mark.parametrize("protect_first_n", [0, 1])
@pytest.mark.parametrize("seed", range(8))
def test_strip_matches_naive_per_row_oracle(seed, protect_first_n):
    code, val, mask, tdd = _random_batch(seed)
    delta_set = set(DELTA_IDS.tolist())

    out_code, out_val, out_mask, out_tdd, out_pos = strip_delta_tokens(
        code, val, mask, tdd, DELTA_IDS, pad_index=PAD, protect_first_n=protect_first_n
    )

    for i in range(code.shape[0]):
        e_code, e_val, e_mask, e_tdd, e_pos = _oracle_row(
            code[i].tolist(),
            val[i].tolist(),
            mask[i].tolist(),
            tdd[i].tolist(),
            delta_set,
            protect_first_n,
        )
        k = len(e_code)
        assert out_code[i, :k].tolist() == e_code, f"row {i}: codes"
        assert out_val[i, :k].tolist() == pytest.approx(e_val, abs=1e-6), f"row {i}: values"
        assert out_mask[i, :k].tolist() == e_mask, f"row {i}: value mask"
        assert out_tdd[i, :k].tolist() == pytest.approx(e_tdd, abs=1e-6), f"row {i}: time deltas"
        assert out_pos[i, :k].tolist() == e_pos, f"row {i}: time positions"
        # Everything past the kept prefix must be padding, not stale data.
        assert (out_code[i, k:] == PAD).all(), f"row {i}: tail not padded"


@pytest.mark.parametrize("seed", range(8))
def test_strip_preserves_total_elapsed_time(seed):
    """The whole point: dropping the delta tokens must not drop the time they encoded."""
    code, val, mask, tdd = _random_batch(seed)
    _, _, _, _, pos = strip_delta_tokens(code, val, mask, tdd, DELTA_IDS, pad_index=PAD)
    keep = (~torch.isin(code, DELTA_IDS)) & (code != PAD)

    for i in range(code.shape[0]):
        idx = keep[i].nonzero().flatten()
        if len(idx) < 2:
            continue
        cum = tdd[i].cumsum(0)
        expected_span = (cum[idx[-1]] - cum[idx[0]]).item() * HOURS_PER_DAY
        actual_span = float(pos[i, len(idx) - 1].item())
        assert actual_span == pytest.approx(expected_span, abs=1.0), (
            f"row {i}: elapsed span {actual_span}h != {expected_span}h across the kept tokens"
        )


def test_positions_are_monotonic_and_rebased():
    """Rotary positions must never go backwards, and each row starts at its own zero."""
    for seed in range(8):
        code, val, mask, tdd = _random_batch(seed)
        _, _, _, _, pos = strip_delta_tokens(code, val, mask, tdd, DELTA_IDS, pad_index=PAD)
        keep = (~torch.isin(code, DELTA_IDS)) & (code != PAD)
        for i in range(code.shape[0]):
            k = int(keep[i].sum().item())
            if k == 0:
                continue
            row = pos[i, :k]
            assert row[0].item() == 0, f"seed {seed} row {i} not re-based to zero"
            assert (row[1:] >= row[:-1]).all(), f"seed {seed} row {i} positions go backwards"


def test_row_of_only_delta_tokens_survives_without_corrupting_neighbours():
    """A degenerate row must not shift another row's data into it."""
    code = torch.tensor([[90, 91, 92, 90], [5, 90, 6, 7]])
    val = torch.zeros(2, 4, dtype=torch.float64)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    tdd = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 0.0, 0.0]], dtype=torch.float64)

    out_code, _, _, _, pos = strip_delta_tokens(code, val, mask, tdd, DELTA_IDS, pad_index=PAD)

    assert (out_code[0] == PAD).all(), "an all-delta row must compact to pure padding"
    assert out_code[1, :3].tolist() == [5, 6, 7]
    assert pos[1, :3].tolist() == [0, 48, 48]
