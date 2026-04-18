"""Shared helpers for the sample-codes scripts.

All three scripts in this submodule start from the same inputs — load a MEDS cohort's
``metadata/codes.parquet``, optionally filter out codes matching an exclude pattern (TIME
tokens by default) — and differ only in how they sample from the resulting code universe.
Keeps those steps in one place so a future schema change to ``codes.parquet`` is a one-line
update.
"""

import hashlib
from pathlib import Path

import polars as pl


def stable_hash_list(items: list[str], length: int = 12) -> str:
    """Order-sensitive, deterministic hash for a list of strings.

    Used to name output files by content so re-running with the same inputs produces an
    identically-named file (idempotent sampling run).

    Examples:
        Same list → same hash:

        >>> stable_hash_list(["A", "B"]) == stable_hash_list(["A", "B"])
        True

        Order matters:

        >>> stable_hash_list(["A", "B"]) != stable_hash_list(["B", "A"])
        True

        Length is configurable:

        >>> len(stable_hash_list(["A"], length=6))
        6
    """
    h = hashlib.sha256()
    for x in items:
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:length]


def load_filtered_codes(metadata_dir: Path, exclude_pattern: str | None = "TIME") -> list[str]:
    """Load the code universe from ``{metadata_dir}/codes.parquet`` and optionally filter.

    Args:
        metadata_dir: Path to a MEDS cohort's ``metadata/`` directory (or any directory
            containing a ``codes.parquet`` with a ``code`` column).  For a standard MEDS layout
            this is ``{COHORT_ROOT}/processed/metadata/`` — but any path with the right file
            structure works, which is the whole point of parameterizing.
        exclude_pattern: If given, any code containing this substring is dropped.  The default
            (``"TIME"``) strips time-derived MEDS tokens that aren't meaningful query targets.
            Pass ``None`` to disable filtering.

    Returns:
        Sorted list of unique codes.

    Raises:
        FileNotFoundError: If ``codes.parquet`` is not under ``metadata_dir``.
    """
    codes_fp = metadata_dir / "codes.parquet"
    if not codes_fp.is_file():
        raise FileNotFoundError(
            f"Expected MEDS codes table at {codes_fp}. "
            f"Pass --metadata-dir pointing at a directory that contains codes.parquet."
        )
    df = pl.read_parquet(codes_fp)
    codes = df["code"].unique().sort().to_list()
    if exclude_pattern is not None:
        codes = [c for c in codes if exclude_pattern not in c]
    return codes
