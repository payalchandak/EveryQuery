"""Deterministic, cross-process-stable seed derivation helpers."""

import hashlib


def derive_seed(*parts: int | str) -> int:
    """Stable 31-bit int seed derived from a tuple of ints/strings via blake2b.

    Python's builtin ``hash`` is not stable across processes (uses a random per-interpreter salt),
    so Hydra multirun workers would draw inconsistent samples if their seeds depended on
    ``hash((seed, "tasks", shard_id))``.  Blake2b is cross-process stable and fast enough to be
    irrelevant at this scale.

    The ``0x7FFFFFFF`` mask keeps the result inside NumPy's legal 31-bit seed range so the return
    value can be passed directly to ``np.random.default_rng(...)`` without further narrowing.

    Examples:
        >>> derive_seed(1, "tasks", 0) == derive_seed(1, "tasks", 0)
        True
        >>> derive_seed(1, "tasks", 0) != derive_seed(1, "tasks", 1)
        True
        >>> derive_seed(1, "contexts", "shard_a", 0) != derive_seed(1, "contexts", "shard_b", 0)
        True
        >>> 0 <= derive_seed(1, "tasks", 0) < 2**31
        True
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")  # unit separator to avoid prefix collisions
    return int.from_bytes(h.digest(), "big") & 0x7FFFFFFF
