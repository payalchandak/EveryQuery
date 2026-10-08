"""Embedding table whose rows are ontology-mixed averages of a raw table's rows.

Installed as the encoder's input-embedding module, this makes every lookup return

    ``(A @ W)[ids]``

where ``W`` is the ordinary learned table and ``A`` is the row-normalised ancestor-mix matrix
from :mod:`every_query.data.ontology`.  A rare leaf code's representation is therefore pulled
toward its better-estimated parents, and an ancestor node — which never appears in a patient
stream — still receives gradient through every descendant that does.

Substituting the module rather than changing call sites is what makes the feature compose: the
patient encoder, the query code slot and an event-bounded query's boundary code all reach the
same module through Hugging Face's ``get_input_embeddings()``, so each one inherits ontology
structure for free.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class OntologyEmbedding(torch.nn.Module):
    """Wraps a raw ``nn.Embedding`` with a sparse ancestor-mix matrix.

    The mixed table is **cached per forward pass** rather than recomputed per lookup.  Each
    lookup materialises the full dense ``(V_ext, H)`` product — 25 MB at a 16k vocabulary and
    ``H=384``, plus as much again in backward — and the conditional model touches the embedding
    table two to three times per forward (patient encoder, query codes, boundary codes).
    Recomputing it each time is cheap in FLOPs and dominant in activation memory,
    so the owning model clears the cache once per forward via a pre-hook and every lookup in
    between shares one product (and one autograd node).

    Args:
        raw: The underlying learned table, sized ``(V_ext, H)``.
        mix: Sparse ``(V_ext, V_ext)`` row-normalised mix matrix.

    Examples:
        A two-node ontology where node 1's row is the average of itself and node 0:

        >>> raw = torch.nn.Embedding(2, 3)
        >>> _ = raw.weight.data.copy_(torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]))
        >>> mix = torch.sparse_coo_tensor(
        ...     torch.tensor([[0, 1, 1], [0, 0, 1]]), torch.tensor([1.0, 0.5, 0.5]), (2, 2)
        ... ).coalesce()
        >>> emb = OntologyEmbedding(raw, mix)
        >>> emb(torch.tensor([0]))
        tensor([[1., 1., 1.]], grad_fn=<IndexBackward0>)
        >>> emb(torch.tensor([1]))  # (0.5 * 1 + 0.5 * 3) = 2
        tensor([[2., 2., 2.]], grad_fn=<IndexBackward0>)

        Multi-dimensional index tensors work, so a whole ``(B, L, K)`` block of ids can be
        looked up in one call:

        >>> emb(torch.zeros(2, 2, 2, dtype=torch.long)).shape
        torch.Size([2, 2, 2, 3])

        Gradient reaches the raw table through the mix, which is how an ancestor row gets
        trained by its descendants:

        >>> emb.clear_cache()
        >>> emb(torch.tensor([1])).sum().backward()
        >>> raw.weight.grad
        tensor([[0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000]])
    """

    def __init__(self, raw: torch.nn.Embedding, mix: torch.Tensor):
        super().__init__()
        self.tok = raw
        self.register_buffer("mix", mix, persistent=False)
        self._mixed: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        """The underlying **raw** table ``W`` — kept for code that introspects ``.weight``.

        Deliberately not the effective table: every lookup returns a row of
        ``mixed_weight() == A @ W``.  Reach for ``.weight`` to touch the learned parameter
        itself (an optimizer parameter group, a gradient check); call :meth:`mixed_weight`
        for the values the model actually sees.
        """
        return self.tok.weight

    @property
    def num_embeddings(self) -> int:
        return self.tok.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.tok.embedding_dim

    def clear_cache(self, *_args) -> None:
        """Drop the cached mixed table.  Called once per forward by the owning model's pre-hook.

        Also drops the autograd graph the cached tensor holds, so a cached product never survives into a
        second backward pass.
        """
        self._mixed = None

    def mixed_weight(self) -> torch.Tensor:
        """The full mixed table ``A @ W``, computed at most once per forward."""
        if self._mixed is None:
            # Force the sparse mm in the raw table's dtype: under autocast a bf16 dense operand
            # would not match the fp32 sparse matrix.
            with torch.autocast(device_type=self.tok.weight.device.type, enabled=False):
                self._mixed = torch.sparse.mm(self.mix.to(self.tok.weight.dtype), self.tok.weight)
        return self._mixed

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.mixed_weight()[ids]


def wrap_tok_embeddings(model: torch.nn.Module, mix: torch.Tensor) -> OntologyEmbedding:
    """Install an :class:`OntologyEmbedding` as the encoder's input-embedding module.

    Goes through Hugging Face's public ``get_input_embeddings`` / ``set_input_embeddings``
    contract rather than reaching into ModernBERT's ``embeddings.tok_embeddings``, so the
    shared-table claim is stated in terms every consumer can ask about: the patient encoder,
    the query code slot and an event-bounded query's boundary code all read whatever module
    is installed here.

    Registers a forward pre-hook on ``model`` that clears the per-forward cache, and returns the
    wrapper.  One outer forward contains every patient, query-code and boundary-code lookup, so
    clearing once before it is what lets them share a single mixed table.

    Raises unless the raw table has exactly ``V_ext`` rows: the sparse product
    ``(V_ext, V_ext) @ (V_model, H)`` is defined only when ``V_model == V_ext``.  Too few rows
    means the encoder was sized from the cohort vocabulary ``V``, leaving every ancestor index
    out of range; too many means it was never sized from the data at all — ModernBERT's own
    50k default, say — and ``torch.sparse.mm`` would fail later and less legibly.
    """
    encoder = model.HF_model
    raw = encoder.get_input_embeddings()
    v_ext = mix.shape[0]
    if raw.num_embeddings != v_ext:
        raise ValueError(
            f"Embedding table has {raw.num_embeddings} rows but the ontology needs exactly "
            f"{v_ext} (V_ext).  Size the encoder from the ontology: train.py does this "
            f"automatically when `lightning_module.model.ontology_dir` is set."
        )

    wrapper = OntologyEmbedding(raw, mix)
    encoder.set_input_embeddings(wrapper)
    model.register_forward_pre_hook(wrapper.clear_cache)
    logger.info("Ontology embeddings active: %d nodes, %d mix entries.", v_ext, mix._nnz())
    return wrapper
