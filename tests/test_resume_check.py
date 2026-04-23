"""Unit tests for ``every_query.train.resume_check.validate_resume_directory``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

from every_query.train.resume_check import LEGACY_REMOVED_KEYS, validate_resume_directory

if TYPE_CHECKING:
    from pathlib import Path


def _write_cfg(path: Path, **overrides) -> None:
    base = {
        "seed": 140799,
        "datamodule": {"config": {"max_seq_len": 256}},
    }
    base.update(overrides)
    OmegaConf.save(OmegaConf.create(base), path)


def test_resume_ignores_legacy_query_stanza(tmp_path: Path) -> None:
    """An old run dir whose saved config still carries a top-level ``query`` stanza must resume cleanly after
    #64 removed the field."""
    assert "query" in LEGACY_REMOVED_KEYS

    saved = tmp_path / "config.yaml"
    _write_cfg(saved, query={"codes": ["A", "B", "C"]})

    new_cfg = OmegaConf.create(
        {
            "seed": 140799,
            "datamodule": {"config": {"max_seq_len": 256}},
        }
    )

    validate_resume_directory(tmp_path, new_cfg)


def test_resume_rejects_non_legacy_drift(tmp_path: Path) -> None:
    """A structural mismatch outside the allow-list / legacy set must still raise."""
    saved = tmp_path / "config.yaml"
    _write_cfg(saved, query={"codes": ["A"]})

    new_cfg = OmegaConf.create(
        {
            "seed": 42,  # drifted
            "datamodule": {"config": {"max_seq_len": 256}},
        }
    )

    with pytest.raises(ValueError, match="seed"):
        validate_resume_directory(tmp_path, new_cfg)
