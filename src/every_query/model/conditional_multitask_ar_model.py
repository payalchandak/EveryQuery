"""Decoder-only all-vocabulary model for ordered multitask windows.

The combined causal stream is::

    [patient events, W0, C0, A0, ..., W(K-2), C(K-2), A(K-2), W(K-1)]

``W_i`` describes the start and end of window ``i``.  ``C_i`` names a code from
that window and ``A_i`` supplies its teacher-forced answer, so the answer can
condition later windows without leaking into the prediction for its own window.
Each ``W_i`` hidden state is projected onto the backbone's input-embedding table,
giving one logit for every vocabulary code without a separate output matrix.

:meth:`ConditionalMultitaskARModel.forward` is the all-vocabulary training pass (dense targets,
masked BCE).  :meth:`ConditionalMultitaskARModel.score_final_query` is the evaluation pass over a
``QuerySeqSchema`` grid: the same hidden states, but only the last real window of each row is
projected, onto only that row's scored code, so no ``(B, K, V)`` tensor is ever built.

Ontology
--------
With ``ontology_dir`` set the model is sized to the ontology's extended vocabulary ``V_ext``
(``train.py`` does this): the tied table gains one row per ancestor node, the input embedding is
the ancestor-mixed :class:`~every_query.model.ontology_embedding.OntologyEmbedding`, and the
readout projects onto the same **mixed** table.  The training labels stay leaf-only: a ``(B, K, V)``
batch (``V`` = :attr:`base_vocab_size`, the cohort's width) is widened inside :meth:`forward` to
``(B, K, V_ext)`` by :func:`~every_query.data.ontology.derive_ancestor_targets`, since under the
window rule an ancestor's bit is exactly the OR of its descendant leaves' bits.  Nothing wider than
``(B, K, V)`` crosses host to device, and nothing about the sampler or its sidecars changes.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import LlamaConfig, LlamaModel
from transformers.modeling_outputs import BaseModelOutput

from every_query.model.answers import (
    N_ANSWER_CLASSES,
    _init_aux_embeddings,
    validate_rope_time_pair,
)
from every_query.model.model import MLP
from every_query.model.ontology_embedding import OntologyEmbedding

TOKENS_PER_WINDOW = 3

TYPE_PATIENT = 0
TYPE_WINDOW = 1
TYPE_CONDITION_CODE = 2
TYPE_CONDITION_ANSWER = 3
N_TOKEN_TYPES = 4


@dataclass
class ConditionalMultitaskOutput(BaseModelOutput):
    """All-vocabulary predictions and the elements that participate in loss.

    Attributes:
        logits: ``(B, K, V)`` float32 logits.
        valid_mask: ``(B, K, V)`` boolean mask.  It combines ``q_mask`` with
            exclusion of the PAD vocabulary row.
    """

    logits: torch.FloatTensor | None = None
    valid_mask: torch.BoolTensor | None = None

    @property
    def probs(self) -> torch.Tensor | None:
        """Float32 sigmoid probabilities with the same shape as ``logits``."""
        if self.logits is None:
            return None
        return torch.sigmoid(self.logits.float())


class ConditionalMultitaskARModel(torch.nn.Module):
    """One Llama backbone over a patient prefix and ``3K-2`` query tokens.

    Args:
        precision: Lightning precision string used to choose the initial backbone dtype.
        config_overrides: Keyword overrides for a fresh :class:`LlamaConfig`.
        max_windows: Maximum supported ``K``; sizes the learned block positions.
        use_rope_time: If true, consume ``batch.time_pos_ids`` as elapsed-hour RoPE positions.
        ontology_dir: Directory of ``EQ_build_ontology`` artifacts.  When set,
            ``config_overrides.vocab_size`` must be the ontology's ``V_ext``; the input embedding
            becomes the ancestor-mixed table, the readout projects onto it, and leaf-only
            ``(B, K, V)`` targets are widened to ``V_ext`` in :meth:`forward` (see the module
            docstring).
        cohort_vocab_fingerprint: :func:`~every_query.utils.digest.vocab_fingerprint` of the cohort's
            ``codes.parquet`` - the identity of the vocabulary the leaf rows ``[0, V)`` mean.
            ``train.py`` fills it in from the training cohort; it is persisted in the hyperparameters,
            so with an ``ontology_dir`` every construction (training and every checkpoint load)
            re-verifies that the ontology at that path was built from *this* cohort, not merely one
            of the same width, and ``EQ_predict_multitask`` checks the inference cohort against it.
            ``None`` (pre-existing checkpoints) keeps the width-only checks.
    """

    PRECISION_TO_MODEL_WEIGHTS_DTYPE: ClassVar[dict[str, torch.dtype]] = {
        "32-true": torch.float32,
        "16-true": torch.float16,
        "16-mixed": torch.float32,
        "bf16-true": torch.bfloat16,
        "bf16-mixed": torch.float32,
        "transformer-engine": torch.bfloat16,
    }

    def __init__(
        self,
        precision: str = "32-true",
        config_overrides: dict[str, Any] | None = None,
        max_windows: int = 5,
        use_rope_time: bool = False,
        ontology_dir: str | None = None,
        cohort_vocab_fingerprint: str | None = None,
    ):
        super().__init__()
        if max_windows < 1:
            raise ValueError(f"max_windows must be at least 1, got {max_windows}")

        self.HF_model_config = LlamaConfig(**(config_overrides or {}))
        self.HF_model_config.use_cache = False
        self.HF_model_config.output_hidden_states = False
        self.HF_model_config.output_attentions = False
        extra_kwargs = {"torch_dtype": self.PRECISION_TO_MODEL_WEIGHTS_DTYPE.get(precision)}
        self.HF_model = LlamaModel._from_config(self.HF_model_config, **extra_kwargs)

        H = self.HF_model_config.hidden_size
        self.start_duration_embed = MLP(layers=[1, 64, H], dropout_prob=0)
        self.end_duration_embed = MLP(layers=[1, 64, H], dropout_prob=0)
        # Put a duration-bounded window token on the same scale as the code / type / marker / block
        # embeddings: at the default Linear init the MLP output norm is ~10x an embedding row, so a
        # window token was dominated by its duration.  Only the output layer is rescaled; the
        # hidden ReLU layer keeps its default init.
        for mlp in (self.start_duration_embed, self.end_duration_embed):
            final = mlp.model[-1]
            torch.nn.init.normal_(final.weight, mean=0.0, std=self.HF_model_config.initializer_range)
            torch.nn.init.zeros_(final.bias)
        self.start_marker = torch.nn.Parameter(torch.randn(H) * self.HF_model_config.initializer_range)
        self.bound_marker = torch.nn.Parameter(torch.randn(H) * self.HF_model_config.initializer_range)
        self.answer_embed = torch.nn.Embedding(N_ANSWER_CLASSES, H)
        self.token_type_embed = torch.nn.Embedding(N_TOKEN_TYPES, H)
        self.block_pos_embed = torch.nn.Embedding(max_windows, H)
        # The optimizer's existing ``bias`` rule puts this parameter in the no-decay group.
        self.code_bias = torch.nn.Parameter(torch.full((self.HF_model_config.vocab_size,), -3.0))

        _init_aux_embeddings(
            self.HF_model_config.initializer_range,
            self.answer_embed,
            self.token_type_embed,
            self.block_pos_embed,
        )

        self.max_windows = max_windows
        self.use_rope_time = use_rope_time
        self.ontology_dir = ontology_dir
        self.cohort_vocab_fingerprint = cohort_vocab_fingerprint
        # ``V``: the cohort's own width.  Equal to ``vocab_size`` (the table width) without an
        # ontology; with one, the leaf block of the ``V_ext``-wide table.
        self._base_vocab_size = self.HF_model_config.vocab_size
        if ontology_dir is not None:
            # Lazy, as in ``ConditionalARModel``: ``every_query.data`` reaches back into this
            # package through the sequence dataset, so a module-level import would cycle.
            from every_query.data.ontology import load_closure_index, load_mix_matrix
            from every_query.model.ontology_embedding import wrap_tok_embeddings

            # Substituting the embedding module (not the call sites) is what lets patient, start,
            # bound and condition codes all inherit the ontology mix; ``wrap_tok_embeddings`` also
            # checks the table is exactly ``V_ext`` rows.
            wrap_tok_embeddings(self, load_mix_matrix(ontology_dir))
            # With the cohort's fingerprint the loader also checks the ontology's observed nodes
            # *are* that cohort's ``codes.parquet`` rows; the width checks alone would accept any
            # same-width ontology and silently pair the leaf columns with the wrong closure rows.
            closure = load_closure_index(ontology_dir, vocab_fingerprint=cohort_vocab_fingerprint)
            if closure.v_ext != self.vocab_size:
                raise ValueError(
                    f"The ontology at {ontology_dir} extends the vocabulary to V_ext={closure.v_ext} but "
                    f"config_overrides.vocab_size={self.vocab_size}; size the model from the ontology "
                    "(train.py does this automatically when lightning_module.model.ontology_dir is set)."
                )
            self._base_vocab_size = closure.base_vocab_size
            # Non-persistent: the closure is re-read from ``ontology_dir`` on load, exactly like the
            # mix matrix, so a checkpoint never carries a copy that could drift from the artifacts.
            self.register_buffer("closure_leaf_ids", closure.leaf_ids, persistent=False)
            self.register_buffer("closure_ancestor_ids", closure.ancestor_ids, persistent=False)
        self.hparams = {
            "architecture": "conditional_multitask_ar",
            "precision": precision,
            "config_overrides": dict(config_overrides) if config_overrides else None,
            "max_windows": max_windows,
            "use_rope_time": use_rope_time,
            "ontology_dir": ontology_dir,
            "cohort_vocab_fingerprint": cohort_vocab_fingerprint,
        }

    @property
    def max_seq_len(self) -> int:
        """Maximum length of the combined patient and query stream."""
        return self.HF_model_config.max_position_embeddings

    @property
    def vocab_size(self) -> int:
        """``V_ext``: the tied table's width, i.e. every code the model can score."""
        return self.HF_model_config.vocab_size

    @property
    def base_vocab_size(self) -> int:
        """``V``: the cohort's own width - the width of a leaf-only training batch.

        Equals :attr:`vocab_size` without an ontology.  With one, ``[base_vocab_size, vocab_size)``
        are the ancestor rows the model derives targets for and can score, but that never occur in
        a patient stream.
        """
        return self._base_vocab_size

    @property
    def has_ontology(self) -> bool:
        return self.ontology_dir is not None

    def _closure(self):
        """The registered closure buffers as a :class:`~every_query.data.ontology.ClosureIndex`."""
        from every_query.data.ontology import ClosureIndex

        return ClosureIndex(
            self.closure_leaf_ids, self.closure_ancestor_ids, self.base_vocab_size, self.vocab_size
        )

    def _readout_weight(self) -> torch.Tensor:
        """The table the window hidden states are projected onto: the **effective** input table.

        ``OntologyEmbedding.weight`` deliberately returns the raw learned table, but every input
        lookup (patient, start, bound and condition codes) reads a row of the mixed table
        ``A @ W``.  Tying the readout to the raw rows would score ancestors through rows the input
        side never sees, so with an ontology this is ``mixed_weight()``; without one the two tables
        are the same object.  The mixed table is computed at most once per forward and shared with
        the input lookups (``window_hidden_states`` clears the cache first).
        """
        embedding = self.HF_model.get_input_embeddings()
        if isinstance(embedding, OntologyEmbedding):
            return embedding.mixed_weight()
        return embedding.weight

    def _widen_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Leaf-only ``(B, K, V)`` targets -> ``(B, K, V_ext)``; a ``V_ext``-wide batch passes through."""
        if targets.shape[-1] == self.base_vocab_size < self.vocab_size:
            from every_query.data.ontology import derive_ancestor_targets

            return derive_ancestor_targets(targets, self._closure())
        return targets

    def _start_fields(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return explicit start tensors, filling the paired legacy absence with zeros."""
        durations = getattr(batch, "q_start_durations", None)
        codes = getattr(batch, "q_start_codes", None)
        if (durations is None) != (codes is None):
            missing = "q_start_durations" if durations is None else "q_start_codes"
            raise ValueError(
                f"q_start_durations and q_start_codes must be given together (got {missing}=None)"
            )
        if durations is None:
            durations = torch.zeros_like(batch.q_durations)
            codes = torch.zeros_like(batch.q_bound_codes)
        return durations, codes

    def _window_embeds(self, batch) -> torch.Tensor:
        """Build the ``(B, K, H)`` window tokens from role-distinct start/end specs."""
        start_durations, start_codes = self._start_fields(batch)
        code_embeddings = self.HF_model.get_input_embeddings()

        start_duration = self.start_duration_embed((start_durations / 365.0).unsqueeze(-1))
        start_event = code_embeddings(start_codes).to(start_duration.dtype)
        start_event = start_event + self.start_marker.to(start_duration.dtype)
        start_spec = torch.where((start_codes > 0).unsqueeze(-1), start_event, start_duration)

        end_duration = self.end_duration_embed((batch.q_durations / 365.0).unsqueeze(-1))
        end_event = code_embeddings(batch.q_bound_codes).to(end_duration.dtype)
        end_event = end_event + self.bound_marker.to(end_duration.dtype)
        end_spec = torch.where((batch.q_bound_codes > 0).unsqueeze(-1), end_event, end_duration)

        n_windows = batch.q_durations.shape[1]
        block_idx = torch.arange(n_windows, device=batch.q_durations.device)
        window_type = self.token_type_embed.weight[TYPE_WINDOW].to(start_spec.dtype)
        block_pos = self.block_pos_embed(block_idx).to(start_spec.dtype)
        return start_spec + end_spec + window_type + block_pos.unsqueeze(0)

    def _query_tokens(self, batch) -> torch.Tensor:
        """Return the exact ``[W0,C0,A0,...,W(K-1)]`` token stream."""
        B, n_windows = batch.q_durations.shape
        H = self.HF_model_config.hidden_size
        windows = self._window_embeds(batch)
        stream = torch.empty(
            B,
            TOKENS_PER_WINDOW * n_windows - 2,
            H,
            dtype=windows.dtype,
            device=windows.device,
        )
        stream[:, 0::TOKENS_PER_WINDOW] = windows

        if n_windows > 1:
            code_embeddings = self.HF_model.get_input_embeddings()
            condition_codes = code_embeddings(batch.condition_codes).to(windows.dtype)
            condition_answers = self.answer_embed(batch.condition_answers.long()).to(windows.dtype)
            tt = self.token_type_embed.weight
            block_idx = torch.arange(n_windows - 1, device=windows.device)
            block_pos = self.block_pos_embed(block_idx).to(windows.dtype).unsqueeze(0)
            stream[:, 1::TOKENS_PER_WINDOW] = (
                condition_codes + tt[TYPE_CONDITION_CODE].to(windows.dtype) + block_pos
            )
            stream[:, 2::TOKENS_PER_WINDOW] = (
                condition_answers + tt[TYPE_CONDITION_ANSWER].to(windows.dtype) + block_pos
            )
        return stream

    def _position_ids(
        self,
        batch,
        n_patient: torch.Tensor,
        query_positions: torch.Tensor,
        total_len: int,
    ) -> torch.Tensor | None:
        """Construct clinical-time positions; query starts never advance clinical time."""
        time_pos = validate_rope_time_pair(self.use_rope_time, getattr(batch, "time_pos_ids", None))
        if time_pos is None:
            return None

        device = batch.code.device
        time_pos = time_pos.to(device)
        B, S = time_pos.shape
        last_idx = (n_patient - 1).clamp(min=0)
        last_hour = time_pos.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        last_hour = torch.where(n_patient > 0, last_hour, torch.zeros_like(last_hour))

        position_ids = torch.zeros(B, total_len, dtype=torch.long, device=device)
        position_ids[:, :S] = time_pos
        query_hours = last_hour.unsqueeze(1).expand(-1, query_positions.shape[1])
        position_ids.scatter_(1, query_positions, query_hours)
        return position_ids

    def window_hidden_states(self, batch) -> torch.Tensor:
        """Run the causal backbone and return the ``(B, K, H)`` hidden state at every window token.

        This is the whole of :meth:`forward` up to (and excluding) the tied-vocabulary projection,
        so the training forward and :meth:`score_final_query` read the *same* hidden states.  All
        the batch validation lives here: at least one window, ``K <= max_windows``, the paired
        start-field rule, the position budget, and the right-padding prefix rule.  The batch needs
        ``code``, ``q_durations``, ``q_bound_codes``, ``q_mask``, ``condition_codes`` /
        ``condition_answers`` (when ``K > 1``) and optionally the start fields and
        ``time_pos_ids``; ``targets`` is never read.
        """
        B, n_windows = batch.q_durations.shape
        if n_windows < 1:
            raise ValueError("ConditionalMultitaskARModel requires at least one window")
        if self.max_windows < n_windows:
            raise ValueError(f"Batch has K={n_windows} windows but max_windows={self.max_windows}")
        # Validate the paired legacy rule even before token construction.
        self._start_fields(batch)

        # ``wrap_tok_embeddings`` clears the per-forward mixed-table cache through a pre-hook on
        # ``forward``; ``score_final_query`` reaches these hidden states without going through
        # ``forward``, so clear here too.  Every lookup below and the readout after share one
        # product (and, in training, one autograd node) per pass.
        embedding = self.HF_model.get_input_embeddings()
        if isinstance(embedding, OntologyEmbedding):
            embedding.clear_cache()

        S = batch.code.shape[1]
        n_query_tokens = TOKENS_PER_WINDOW * n_windows - 2
        total_len = S + n_query_tokens
        if total_len > self.max_seq_len:
            raise ValueError(
                f"Combined sequence needs {total_len} positions ({S} patient + "
                f"{n_query_tokens} query tokens) but max_position_embeddings={self.max_seq_len}. "
                "The configured budget must cover max_seq_len + 3 * max_windows."
            )

        device = batch.code.device
        pad = batch.PAD_INDEX
        patient_mask = batch.code != pad
        n_patient = patient_mask.sum(dim=1)
        # The query tokens are scattered to positions n_patient.. and the patient prefix is
        # attended as ``code != PAD``, which is only right when every real token precedes every
        # PAD.  Left padding or an interior PAD would silently overwrite real tokens.
        if (patient_mask[:, 1:] & ~patient_mask[:, :-1]).any():
            raise ValueError(
                "patient_mask is not a prefix mask: batch.code has a real token after a PAD (left "
                "padding or an interior PAD). ConditionalMultitaskARModel requires right padding."
            )

        patient_emb = self.HF_model.get_input_embeddings()(batch.code)
        patient_emb = patient_emb + self.token_type_embed.weight[TYPE_PATIENT].to(patient_emb.dtype)
        patient_emb = patient_emb * patient_mask.unsqueeze(-1).to(patient_emb.dtype)
        query_tokens = self._query_tokens(batch).to(patient_emb.dtype)
        H = patient_emb.shape[-1]

        query_positions = n_patient.unsqueeze(1) + torch.arange(n_query_tokens, device=device).unsqueeze(0)
        inputs_embeds = torch.zeros(B, total_len, H, dtype=patient_emb.dtype, device=device)
        inputs_embeds[:, :S] = patient_emb
        inputs_embeds.scatter_(1, query_positions.unsqueeze(-1).expand(-1, -1, H), query_tokens)

        attention_mask = torch.zeros(B, total_len, dtype=torch.long, device=device)
        attention_mask[:, :S] = patient_mask.long()
        query_attn = batch.q_mask.repeat_interleave(TOKENS_PER_WINDOW, dim=1)[:, :n_query_tokens]
        attention_mask.scatter_(1, query_positions, query_attn.long())

        position_ids = self._position_ids(batch, n_patient, query_positions, total_len)
        hidden = self.HF_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        window_positions = n_patient.unsqueeze(1) + TOKENS_PER_WINDOW * torch.arange(
            n_windows, device=device
        ).unsqueeze(0)
        return hidden.gather(1, window_positions.unsqueeze(-1).expand(-1, -1, H))

    def forward(self, batch) -> tuple[torch.FloatTensor, ConditionalMultitaskOutput]:
        """Run one causal pass and return masked BCE loss plus all-vocabulary logits.

        ``batch.targets`` may be ``(B, K, V)`` leaf bits (the training sidecars' width) or, under
        an ontology, already ``(B, K, V_ext)``; the former is widened here, on the batch's device,
        before the projection.  Logits and ``valid_mask`` are always ``(B, K, vocab_size)``.
        """
        window_hidden = self.window_hidden_states(batch)
        B, n_windows = batch.q_durations.shape
        device = batch.code.device
        pad = batch.PAD_INDEX

        targets = self._widen_targets(batch.targets)
        embedding_weight = self._readout_weight()
        if embedding_weight.shape[0] != targets.shape[-1]:
            raise ValueError(
                f"Target vocabulary width V={targets.shape[-1]} does not match the tied "
                f"embedding table width V={embedding_weight.shape[0]}"
                + (
                    f" (leaf-only targets must be exactly base_vocab_size={self.base_vocab_size} wide)"
                    if self.base_vocab_size != self.vocab_size
                    else ""
                )
            )
        # Explicitly leave autocast: `.float()` alone is still downcast by bf16 autocast.
        with torch.autocast(device_type=window_hidden.device.type, enabled=False):
            logits = window_hidden.float() @ embedding_weight.float().T
            logits = logits + self.code_bias.float()

        vocab_not_pad = torch.arange(logits.shape[-1], device=device) != pad
        valid_mask = batch.q_mask.unsqueeze(-1) & vocab_not_pad.view(1, 1, -1)
        valid_mask = valid_mask.expand(B, n_windows, -1)
        per_element = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        loss = (per_element * valid_mask).sum() / valid_mask.sum().clamp_min(1)

        return loss, ConditionalMultitaskOutput(
            last_hidden_state=None,
            logits=logits,
            valid_mask=valid_mask,
        )

    @staticmethod
    def last_real_window(q_mask: torch.Tensor) -> torch.Tensor:
        """Index of each row's last real window, requiring a non-empty right-padded prefix mask.

        Examples:
            >>> ConditionalMultitaskARModel.last_real_window(
            ...     torch.tensor([[True, True, False], [True, False, False]])).tolist()
            [1, 0]
            >>> ConditionalMultitaskARModel.last_real_window(torch.tensor([[True, False, True]]))
            Traceback (most recent call last):
                ...
            ValueError: q_mask must be a right-padded prefix mask (True at every real window, then False)
            >>> ConditionalMultitaskARModel.last_real_window(torch.tensor([[False, False]]))
            Traceback (most recent call last):
                ...
            ValueError: every row needs at least one real window; q_mask row 0 is all False
        """
        if q_mask.dim() != 2 or q_mask.dtype != torch.bool:
            raise ValueError(f"q_mask must be a 2-D boolean tensor, got {tuple(q_mask.shape)} {q_mask.dtype}")
        if (q_mask[:, 1:] & ~q_mask[:, :-1]).any():
            raise ValueError(
                "q_mask must be a right-padded prefix mask (True at every real window, then False)"
            )
        n_real = q_mask.sum(dim=1)
        if (n_real == 0).any():
            row = int((n_real == 0).nonzero()[0].item())
            raise ValueError(f"every row needs at least one real window; q_mask row {row} is all False")
        return n_real - 1

    def score_final_query(self, batch, scored_codes: torch.Tensor) -> torch.Tensor:
        """Float32 logit ``(B,)`` of ``scored_codes[b]`` at row ``b``'s **last real** window.

        This is exactly ``forward(batch)[1].logits[b, last_b, scored_codes[b]]`` where
        ``last_b = q_mask[b].sum() - 1``: the same :meth:`window_hidden_states` (one backbone
        pass, identical inputs, positions and masks), the same tied input-embedding row and the
        same ``code_bias`` entry — just selected per row instead of projected onto the whole
        vocabulary.  Nothing of shape ``(B, K, V)`` is built and ``batch.targets`` is never
        read, so a batch without targets (the QuerySeq evaluation adapter's) is fine.

        The dot product is taken per row (``(h * e).sum(-1)``) rather than as a slice of the
        ``h @ E.T`` matmul, so the float32 accumulation order can differ from :meth:`forward`'s
        at the rounding level (~1e-6 relative); the two agree to ``allclose`` tolerance, not
        bit-for-bit.  Like the projection in :meth:`forward`, it runs outside autocast.

        Args:
            batch: Any batch :meth:`window_hidden_states` accepts; ``q_mask`` must be a
                right-padded prefix mask with at least one real window per row.
            scored_codes: ``(B,)`` int64 vocabulary indices, never PAD, each ``< vocab_size``.
        """
        # Validate the codes before paying for the backbone pass.
        B = batch.q_durations.shape[0]
        if scored_codes.shape != (B,) or scored_codes.dtype != torch.long:
            raise ValueError(
                f"scored_codes must be an int64 tensor of shape ({B},), got "
                f"{tuple(scored_codes.shape)} {scored_codes.dtype}"
            )
        if scored_codes.numel() and (scored_codes.min() < 0 or scored_codes.max() >= self.vocab_size):
            raise ValueError(
                f"scored_codes must lie in [0, {self.vocab_size}); got min {int(scored_codes.min())}, "
                f"max {int(scored_codes.max())}"
            )
        if (scored_codes == batch.PAD_INDEX).any():
            raise ValueError(f"scored_codes must never be PAD (index {batch.PAD_INDEX})")

        window_hidden = self.window_hidden_states(batch)
        H = window_hidden.shape[-1]
        last = self.last_real_window(batch.q_mask).to(window_hidden.device)
        selected = window_hidden.gather(1, last.view(B, 1, 1).expand(-1, 1, H)).squeeze(1)
        # The same effective table ``forward`` projects onto (the mixed one under an ontology), so an
        # ancestor code in ``[base_vocab_size, vocab_size)`` scores through the row its logit in the
        # dense forward reads.
        embedding_weight = self._readout_weight()
        # Explicitly leave autocast, as the full projection in ``forward`` does: `.float()`
        # alone is still downcast by bf16 autocast.
        with torch.autocast(device_type=selected.device.type, enabled=False):
            rows = embedding_weight[scored_codes].float()
            logits = (selected.float() * rows).sum(dim=-1) + self.code_bias[scored_codes].float()
        return logits
