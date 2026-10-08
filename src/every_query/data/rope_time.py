"""Continuous-time position ids for rotary attention, and removal of the delta-token stream.

The MEICAR-style token stream interleaves quantized ``TIMELINE//DELTA//*`` tokens with real
clinical events, so elapsed time is recoverable only by summing those categorical tokens.  This
module supports the alternative time representation: drop the delta tokens from the encoder input
entirely and expose each remaining token's **elapsed time** (hours since the row's first kept
token) as ``time_pos_ids``, which
:class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel` feeds to its
backbone's rotary position machinery.  Attention then sees *continuous relative time* rather than
token distance plus quantized delta tokens.  The two halves are kept in step by
:func:`~every_query.model.answers.validate_rope_time_pair`.

Mechanics:

- ``time_delta_days`` is non-zero exactly at the delta-token positions (each delta token carries
  its own elapsed-time value), so cumulative time per token is ``cumsum(time_delta_days)``
  computed **before** the delta tokens are dropped, then compacted alongside the kept tokens.
  Dropping them first would silently discard the elapsed time they encode.
- Removing delta tokens shortens sequences materially (~13% on MIMIC-IV in the source
  experiment), so the freed positions admit more real clinical events per fixed ``max_seq_len``
  window.  That is part of the change under test, not an incidental effect.
- ``time_pos_ids`` are relative-offset invariant under RoPE, so the zero point is arbitrary; each
  row is re-based to start at 0.  Hour resolution keeps ICU-scale (sub-day) dynamics while a
  ten-year record still fits in < 90k position units.

Unlike the single-query :class:`~every_query.data.dataset.EveryQueryPytorchDataset`, the
conditional model prepends **no** query token to the patient stream — queries live exclusively in
the decoder — so nothing in the encoder input needs protecting from the strip.  ``protect_first_n``
exists for callers that do prepend.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Vocabulary prefix of the quantized time-delta tokens emitted by the MEDS tokenizer.
DELTA_TOKEN_PREFIX = "TIMELINE//DELTA"
HOURS_PER_DAY = 24.0


def delta_vocab_ids(code_to_index: dict[str, int]) -> torch.Tensor:
    """Vocab indices of every ``TIMELINE//DELTA*`` code, ascending.

    Args:
        code_to_index: Cohort vocabulary mapping (code string -> vocab index).

    Returns:
        A 1-D ``long`` tensor of delta-token vocab indices; empty when the cohort has none.

    Examples:
        >>> delta_vocab_ids({"A": 3, "TIMELINE//DELTA//1d": 9, "TIMELINE//DELTA//1h": 7}).tolist()
        [7, 9]
        >>> delta_vocab_ids({"A": 3}).tolist()
        []
    """
    ids = sorted(idx for code, idx in code_to_index.items() if code.startswith(DELTA_TOKEN_PREFIX))
    return torch.as_tensor(ids, dtype=torch.long)


def build_keep_mask(
    code: torch.Tensor,
    delta_ids: torch.Tensor,
    pad_index: int = 0,
    protect_first_n: int = 0,
) -> torch.Tensor:
    """``(B, N)`` bool mask of tokens surviving the strip (non-delta, non-padding).

    Exposed separately from :func:`strip_delta_tokens` so a caller can compact *other*
    per-token batch fields (e.g. ``static_mask``) with exactly the same selection.

    Examples:
        >>> build_keep_mask(torch.tensor([[5, 90, 6, 0]]), torch.tensor([90])).tolist()
        [[True, False, True, False]]

        ``protect_first_n`` pins leading positions in place even if they look like deltas:

        >>> build_keep_mask(
        ...     torch.tensor([[90, 90, 6, 0]]), torch.tensor([90]), protect_first_n=1
        ... ).tolist()
        [[True, False, True, False]]
    """
    is_delta = torch.isin(code, delta_ids.to(code.device))
    if protect_first_n:
        is_delta[:, :protect_first_n] = False
    return (~is_delta) & (code != pad_index)


def compact_by_keep(
    tensor: torch.Tensor,
    keep: torch.Tensor,
    new_n: int,
    fill: int | float | bool = 0,
) -> torch.Tensor:
    """Left-pack ``tensor``'s kept entries per row, right-padding to ``new_n`` with ``fill``.

    Args:
        tensor: ``(B, N)`` tensor to compact.
        keep: ``(B, N)`` bool mask from :func:`build_keep_mask`.
        new_n: Output width.
        fill: Padding value for the vacated tail.

    Examples:
        >>> keep = torch.tensor([[True, False, True]])
        >>> compact_by_keep(torch.tensor([[5, 90, 6]]), keep, 2).tolist()
        [[5, 6]]
    """
    B = tensor.shape[0]
    out = torch.full((B, new_n), fill, dtype=tensor.dtype)
    for i in range(B):
        sel = keep[i]
        n = int(sel.sum().item())
        if n:
            out[i, :n] = tensor[i, sel]
    return out


def strip_delta_tokens(
    code: torch.Tensor,
    numeric_value: torch.Tensor,
    numeric_value_mask: torch.Tensor,
    time_delta_days: torch.Tensor,
    delta_ids: torch.Tensor,
    pad_index: int = 0,
    protect_first_n: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop delta tokens from a right-padded batch, preserving elapsed time exactly.

    Args:
        code: ``(B, N)`` long token ids.
        numeric_value: ``(B, N)`` float values aligned to ``code``.
        numeric_value_mask: ``(B, N)`` bool — True where ``numeric_value`` is meaningful.
        time_delta_days: ``(B, N)`` float — per-token elapsed time increment in days.
        delta_ids: 1-D long tensor of delta-token vocab indices (see :func:`delta_vocab_ids`).
        pad_index: Padding token id, used for both detection and refill.
        protect_first_n: Leading positions never dropped (for callers that prepend tokens).

    Returns:
        ``(code, numeric_value, numeric_value_mask, time_delta_days, time_pos_ids)``, each
        compacted with delta tokens removed and right-padded to the new max length.
        ``time_pos_ids[i, j]`` is the elapsed time at kept token ``j`` in integer hours,
        measured from row ``i``'s first kept token.

    Examples:
        A row with two delta tokens (id 90) carrying 1 day and 0.5 day.  The delta tokens go
        away, but the time they encoded survives in ``time_pos_ids``:

        >>> code = torch.tensor([[5, 90, 6, 90, 7, 0]])
        >>> nv = torch.tensor([[0.0, 0.0, 1.5, 0.0, 0.0, 0.0]])
        >>> nvm = torch.tensor([[False, False, True, False, False, False]])
        >>> tdd = torch.tensor([[0.0, 1.0, 0.0, 0.5, 0.0, 0.0]])
        >>> c, v, m, t, pos = strip_delta_tokens(code, nv, nvm, tdd, torch.tensor([90]))
        >>> c.tolist()
        [[5, 6, 7]]
        >>> pos.tolist()  # elapsed hours at kept tokens: 0, 24 (1d), 36 (+0.5d)
        [[0, 24, 36]]

        Values ride along with their own token:

        >>> v[0].tolist()
        [0.0, 1.5, 0.0]
        >>> m[0].tolist()
        [False, True, False]

        ``time_delta_days`` is recomputed from the surviving positions, so it still means
        "days since the previous token" rather than collapsing to zeros:

        >>> t[0].tolist()
        [0.0, 1.0, 0.5]

        Each row is re-based to start at zero, so two rows whose events sit at the same
        *relative* offsets get identical positions regardless of absolute start:

        >>> code = torch.tensor([[5, 90, 6, 0], [5, 90, 6, 0]])
        >>> nv = torch.zeros(2, 4)
        >>> nvm = torch.zeros(2, 4, dtype=torch.bool)
        >>> tdd = torch.tensor([[0.0, 1.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0]])
        >>> _, _, _, _, pos = strip_delta_tokens(code, nv, nvm, tdd, torch.tensor([90]))
        >>> pos.tolist()
        [[0, 24], [0, 24]]

        With no delta codes in the vocabulary the only change is that padding is compacted:

        >>> code = torch.tensor([[5, 6, 0, 0]])
        >>> _, _, _, _, pos = strip_delta_tokens(
        ...     code, torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool),
        ...     torch.tensor([[0.0, 2.0, 0.0, 0.0]]), torch.tensor([], dtype=torch.long),
        ... )
        >>> pos.tolist()
        [[0, 48]]
    """
    B, _ = code.shape
    keep = build_keep_mask(code, delta_ids, pad_index, protect_first_n)

    # Cumulative time must be taken BEFORE the strip: the delta tokens are what carry it.
    cum_days = torch.nan_to_num(time_delta_days, nan=0.0).cumsum(dim=1)
    time_pos = (cum_days * HOURS_PER_DAY).round().long()

    kept_counts = keep.sum(dim=1)
    new_n = max(int(kept_counts.max().item()), 1 + protect_first_n)

    out_code = compact_by_keep(code, keep, new_n, pad_index)
    out_nv = compact_by_keep(numeric_value, keep, new_n, 0)
    out_nvm = compact_by_keep(numeric_value_mask, keep, new_n, False)

    # Re-base each row to its own first kept token: RoPE only sees relative offsets, so the
    # absolute zero point is arbitrary, and re-basing keeps the magnitudes small.
    out_pos = compact_by_keep(time_pos, keep, new_n, 0)
    n_kept = keep.sum(dim=1)
    for i in range(B):
        if n_kept[i]:
            out_pos[i, : n_kept[i]] -= out_pos[i, 0].clone()

    # `time_delta_days` must be RECOMPUTED, not merely compacted.  The gaps lived on the delta
    # tokens we just dropped, so compacting the kept tokens' own values would leave a field
    # named "time delta" holding all zeros, one attribute away from `time_pos_ids` holding the
    # real elapsed time.  Instead each kept token gets its true gap from the previous kept
    # token, so the field keeps its documented meaning after the strip.
    out_tdd = torch.zeros(B, new_n, dtype=time_delta_days.dtype)
    if new_n > 1:
        out_tdd[:, 1:] = (out_pos[:, 1:] - out_pos[:, :-1]).to(time_delta_days.dtype) / HOURS_PER_DAY
    live = torch.arange(new_n).unsqueeze(0) < n_kept.unsqueeze(1)
    out_tdd = torch.where(live, out_tdd, torch.zeros_like(out_tdd))

    return out_code, out_nv, out_nvm, out_tdd, out_pos
