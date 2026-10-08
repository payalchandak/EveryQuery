"""Answer vocabulary and model-construction helpers shared by every query-answering model.

These pieces are architecture-independent: the binary answer encoding is a property of the
*question* every query asks, and the two helpers below are model-construction plumbing that any
wrapper around a HuggingFace backbone needs.  They live here rather than beside any one model so
that a model module can be retired without taking them with it.

:func:`validate_rope_time_pair` is the guard that keeps ``use_rope_time`` (a model setting) and
``time_pos_ids`` (a batch field) from drifting apart; see its docstring for the post-mortem it
encodes.
"""

import torch

# Answer-token vocabulary for teacher forcing.  Answers are binary: every query asks
# "is this code observed in (t, t+d)?" and the answer is YES/NO.  There is no separate
# "censored" answer class — censoring is expressed *as a query*, by asking about the
# end-of-timeline code (TIMELINE//END): "[TIMELINE//END, d]" answered YES means the record
# ends within d (the window is not fully observed), NO means data continue past t+d.  A
# downstream query conditions on that answer, which is strictly more expressive than the
# original EveryQuery's implicit "P(occurs | data exist after d)".
ANSWER_NO = 0
ANSWER_YES = 1
N_ANSWER_CLASSES = 2


def validate_rope_time_pair(use_rope_time: bool, time_pos: torch.Tensor | None) -> torch.Tensor | None:
    """Enforce that ``use_rope_time`` (model) and ``time_pos_ids`` (batch) arrive as a pair.

    Returns the validated ``time_pos`` tensor (``None`` when rope-time is off).  Both
    half-configurations are hard errors rather than silent fallbacks, because either direction
    yields a model that trains, validates and checkpoints with entirely normal-looking numbers
    while its backbone is missing the elapsed-time signal:

    * ``use_rope_time=True`` with no ``time_pos_ids`` — the model would quietly train or score on
      token-index positions, indistinguishable from success until the numbers are already
      published.  The upstream experiment hit exactly this and had to issue an erratum after a
      full eval grid had been computed against a model that never received its time positions.
      The misconfiguration is a real one: pairing a RoPE checkpoint with a dataset built without
      ``strip_delta_tokens=True``.
    * ``use_rope_time=False`` with ``time_pos_ids`` present — worse, and the reason this direction
      is checked at all.  ``time_pos_ids`` is emitted only by a dataset built with
      ``strip_delta_tokens=True``, so its presence on the batch means the strip was **asked for**,
      and normally that the ``TIMELINE//DELTA*`` tokens have already been deleted from
      ``batch.code``.  Ignoring the positions then leaves the backbone with **no elapsed-time
      information at all**: the delta tokens are gone from the stream and the elapsed hours that
      replaced them are dropped on the floor.  Nothing downstream recovers them —
      ``time_delta_days`` survives on the batch but never reaches the backbone, which reads only
      ``code``, the attention mask and these positions.

      One case emits the positions without deleting anything: a cohort whose vocabulary has no
      ``TIMELINE//DELTA*`` codes at all, where the dataset warns and carries on.  Refusing is
      still correct there — the strip was requested deliberately, nothing is consuming the
      positions, and the run is misconfigured in a way the user needs told about rather than
      smoothed over.

    Shared by every model that consumes a strippable batch, so no architecture can silently drift
    from the contract.
    """
    if not use_rope_time:
        if time_pos is not None:
            raise ValueError(
                "use_rope_time=False but the batch carries time_pos_ids, which only a dataset "
                "built with strip_delta_tokens=True emits (MultitaskBoundaryPytorchDataset, "
                "QuerySeqMultitaskEvalDataset) — so the TIMELINE//DELTA* tokens have already "
                "been stripped from the model input, and token-index positions would leave the "
                "model with no elapsed-time information at all.  Either half of the pair fixes "
                "it: set `lightning_module.model.use_rope_time=true` to consume the positions "
                "(most likely what was meant, since the strip was switched on deliberately), or "
                "`datamodule.dataset_kwargs.strip_delta_tokens=false` to keep the delta tokens "
                "in the stream.  Refusing to silently discard elapsed time."
            )
        return None
    if time_pos is None:
        raise ValueError(
            "use_rope_time=True but the batch carries no time_pos_ids.  Build the dataset with "
            "strip_delta_tokens=True — via `datamodule.dataset_kwargs.strip_delta_tokens=true` — "
            "so elapsed-hour positions are emitted, or set "
            "`lightning_module.model.use_rope_time=false` to run on the unstripped stream.  "
            "Refusing to fall back to token-index positions silently."
        )
    return time_pos


def _init_aux_embeddings(std: float, *embeddings: torch.nn.Embedding) -> None:
    """Re-init embedding tables built outside the HF backbone to the backbone's scale.

    HF models initialize their own submodules in ``post_init()`` with
    ``config.initializer_range`` (0.02).  Tables constructed afterwards on the wrapper keep
    ``nn.Embedding``'s default ``N(0, 1)``, ~50x wider, so shared type/position vectors
    dominate the summed input at init.  Call this instead of ``self.apply(...)``, which would
    also reinitialize the already-initialized backbone.

        >>> emb = torch.nn.Embedding(1000, 64)
        >>> _init_aux_embeddings(0.02, emb)
        >>> bool(0.015 < emb.weight.std().item() < 0.025)
        True
    """
    for embedding in embeddings:
        torch.nn.init.normal_(embedding.weight, mean=0.0, std=std)
