"""One digest for every "is this artifact the one I labeled under" comparison.

Torch-free on purpose: the samplers' worker processes fingerprint their inputs and must not pay
for importing :mod:`every_query.data.ontology` (which imports torch) to do so, while the model
side digests the same closure table and has to get the same string.
"""

import hashlib
from collections.abc import Mapping

import polars as pl

#: Salt of :func:`vocab_fingerprint`.  Pinned independently of the multitask sampler's
#: ``FORMAT_VERSION`` so a manifest format bump does not move ``vocab_fingerprint`` and legacy
#: outputs still pass the cohort check.
VOCAB_FINGERPRINT_VERSION = 2


def vocab_fingerprint(code_to_index: Mapping[str, int]) -> str:
    """Digest of a ``code -> code/vocab_index`` mapping: the multitask manifest's ``vocab_fingerprint``.

    SHA-256 over ``"{index}\\t{code}\\n"`` rows in index order, prefixed by the width
    ``V = max(index) + 1``.  It is the identity of a cohort's vocabulary: two ``codes.parquet`` files
    with the same width but different codes, or the same codes at permuted indices, digest
    differently.  The multitask sampler records it in every manifest, the multitask dataset
    recomputes it from the cohort's ``codes.parquet``, and the ontology loader
    (``every_query.data.ontology.ontology_vocab_fingerprint``) computes it over an ontology's
    observed nodes, so all three compare like with like.

    Insertion order is irrelevant; only the pairs are:

    Examples:
        >>> vocab_fingerprint({"B": 2, "A": 1}) == vocab_fingerprint({"A": 1, "B": 2})
        True
        >>> vocab_fingerprint({"A": 2, "B": 1}) == vocab_fingerprint({"A": 1, "B": 2})
        False
        >>> vocab_fingerprint({"A": 1, "C": 2}) == vocab_fingerprint({"A": 1, "B": 2})
        False
        >>> vocab_fingerprint({})
        Traceback (most recent call last):
            ...
        ValueError: the vocabulary is empty
    """
    if not code_to_index:
        raise ValueError("the vocabulary is empty")
    ordered = sorted((int(i), str(c)) for c, i in code_to_index.items())
    if ordered[0][0] < 0:
        raise ValueError("code/vocab_index must be non-negative")
    if len({i for i, _ in ordered}) != len(ordered):
        raise ValueError("code/vocab_index values must be unique")
    size = ordered[-1][0] + 1
    h = hashlib.sha256()
    h.update(f"multitask-vocab-v{VOCAB_FINGERPRINT_VERSION}:{size}\n".encode())
    for i, c in ordered:
        h.update(f"{i}\t{c}\n".encode())
    return h.hexdigest()


def frame_digest(df: pl.DataFrame) -> str:
    """Serialization-independent digest of a frame's logical rows: ``"{height}:{hash16}"``.

    Polars' vectorized ``hash_rows`` summed over rows, combined with the row count.  Summing is
    order-independent and counts duplicates, so the digest depends only on the *multiset* of rows
    - exactly what decides a label - and a rewritten-but-identical parquet digests the same.  Not
    collision-proof; a collision only ever means a stale artifact is reused, never a wrong label.

    The same construction as ``sample_tasks._index_fingerprint`` for Stage 4's index partitions.

    Examples:
        >>> a = pl.DataFrame({"x": [1, 2], "y": ["p", "q"]})
        >>> frame_digest(a) == frame_digest(a.reverse())
        True
        >>> frame_digest(a) != frame_digest(a.head(1))
        True
        >>> frame_digest(a.clear())
        '0:0000000000000000'
    """
    total = int(df.hash_rows(seed=0).sum()) if df.height else 0
    return f"{df.height}:{total & 0xFFFFFFFFFFFFFFFF:016x}"
