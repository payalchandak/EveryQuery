"""End-to-end tests for the ``EQ_process_data`` CLI.

These run the real console script against ``simple_static_MEDS`` (via the
``eq_preprocessed_dataset`` fixture in ``conftest.py``) and verify the output layout,
codes-metadata schema, and per-split tokenization outputs.

A differential test re-runs ``EQ_process_data`` with a stricter
``MIN_SUBJECTS_PER_CODE`` threshold and asserts the resulting vocabulary is strictly
smaller — catching silent flag-ignoring regressions that shape-only tests would miss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from conftest import run_and_check

if TYPE_CHECKING:
    from pathlib import Path


def test_expected_stage_dirs_exist(eq_preprocessed_dataset: Path) -> None:
    """The output dir contains the tensorization stages MTD consumers expect downstream."""
    for subdir in (
        "metadata",
        "tokenization",
        "tokenization/event_seqs",
        "tokenization/schemas",
        "normalization",
        "fit_normalization",
        "fit_vocabulary_indices",
    ):
        assert (eq_preprocessed_dataset / subdir).exists(), (
            f"expected preprocessing stage dir {subdir!r} is missing under {eq_preprocessed_dataset}"
        )


def test_codes_metadata_schema(eq_preprocessed_dataset: Path) -> None:
    """codes.parquet has the aggregation columns MTD + the model rely on, and is non-empty."""
    codes = pl.read_parquet(eq_preprocessed_dataset / "metadata" / "codes.parquet")
    required_cols = {
        "code",
        "code/n_occurrences",
        "code/n_subjects",
        "code/vocab_index",
    }
    assert required_cols.issubset(set(codes.columns)), (
        f"codes.parquet missing required columns.  have={set(codes.columns)}, need={required_cols}"
    )
    assert codes.height > 0, "codes.parquet is empty — preprocessing produced no vocabulary"
    # vocab_index is used as the token-id contract by the dataloader.  Must be dense and
    # contiguous starting from 1 (index 0 is reserved for padding by the tokenizer).
    null_idx = codes["code/vocab_index"].null_count()
    assert null_idx == 0, (
        f"codes.parquet has {null_idx} rows with null vocab_index — "
        f"some codes were not assigned a vocabulary slot"
    )
    idx = sorted(codes["code/vocab_index"].to_list())
    assert idx == list(range(1, len(idx) + 1)), f"vocab_index is not dense 1..N: {idx}"


def test_per_split_tokenization_outputs(eq_preprocessed_dataset: Path) -> None:
    """Each canonical MEDS split gets tokenized event_seqs + schema shards."""
    for split in ("train", "tuning", "held_out"):
        event_seqs = list((eq_preprocessed_dataset / "tokenization" / "event_seqs" / split).glob("*.parquet"))
        schemas = list((eq_preprocessed_dataset / "tokenization" / "schemas" / split).glob("*.parquet"))
        assert event_seqs, f"no tokenized event_seqs for split={split}"
        assert schemas, f"no schema shards for split={split}"

        seqs_df = pl.read_parquet(event_seqs[0])
        # MTD's tokenized shard contract — subject_id + integer code array.  Downstream dataloaders
        # depend on this structure, so a regression here would be silent and corrosive.
        assert "subject_id" in seqs_df.columns, f"tokenized shard missing subject_id: {seqs_df.columns}"
        assert "code" in seqs_df.columns, f"tokenized shard missing code: {seqs_df.columns}"


def test_min_subjects_filter_is_honored(
    eq_preprocessed_dataset: Path,
    simple_static_MEDS: Path,
    tmp_path_factory,
) -> None:
    """Raising MIN_SUBJECTS_PER_CODE should strictly shrink the vocabulary.

    Differential assertion: the demo run (MIN_SUBJECTS_PER_CODE=2, injected via
    ``do_demo=True``) produces ``N_demo`` codes; a second run with MIN=5 on the same input
    must produce ``N_strict < N_demo`` codes since some codes are guaranteed to fall below
    the stricter threshold.  A silent flag-drop (``MIN_SUBJECTS_PER_CODE`` ignored in the
    pipeline config) would yield ``N_strict == N_demo`` and fail this test.
    """
    root = tmp_path_factory.mktemp("eq_preprocessed_strict")
    intermediate = root / "intermediate"
    final = root / "final"
    # Don't pass do_demo=True — we want our env overrides to win.  Keep MIN_EVENTS_PER_SUBJECT=1
    # so we don't lose all subjects from the tiny dataset.
    run_and_check(
        [
            "EQ_process_data",
            f"input_dir={simple_static_MEDS!s}",
            f"intermediate_dir={intermediate!s}",
            f"output_dir={final!s}",
            "do_demo=False",
        ],
        env={"MIN_SUBJECTS_PER_CODE": "5", "MIN_EVENTS_PER_SUBJECT": "1"},
        timeout=300.0,
    )

    demo_codes = pl.read_parquet(eq_preprocessed_dataset / "metadata" / "codes.parquet")
    strict_codes = pl.read_parquet(final / "metadata" / "codes.parquet")

    assert strict_codes.height < demo_codes.height, (
        f"MIN_SUBJECTS_PER_CODE=5 did not shrink the vocabulary relative to MIN=2 "
        f"(strict={strict_codes.height}, demo={demo_codes.height}).  The filter may be silently ignored."
    )
