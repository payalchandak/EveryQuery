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


def test_resume_survives_the_amp_precision_move(tmp_path: Path) -> None:
    """A run dir predating the AMP move — no ``trainer.precision``, old model value — still resumes."""
    saved = tmp_path / "config.yaml"
    _write_cfg(
        saved,
        trainer={"accelerator": "auto"},
        lightning_module={"model": {"precision": "16-mixed"}},
    )

    new_cfg = OmegaConf.create(
        {
            "seed": 140799,
            "datamodule": {"config": {"max_seq_len": 256}},
            "trainer": {"accelerator": "auto", "precision": "bf16-mixed"},
            "lightning_module": {"model": {"precision": "bf16-mixed"}},
        }
    )

    validate_resume_directory(tmp_path, new_cfg)


def test_resume_still_rejects_precision_drift_after_the_amp_move(tmp_path: Path) -> None:
    """Once a run dir has ``trainer.precision``, precision drift raises again.

    The legacy exemption must not become a blanket allow-list: ``bf16-true`` casts weights to bf16
    via Lightning's ``HalfPrecision.convert_module``, so silently accepting it would resume with
    half-precision masters.
    """
    saved = tmp_path / "config.yaml"
    _write_cfg(
        saved,
        trainer={"accelerator": "auto", "precision": "bf16-mixed"},
        lightning_module={"model": {"precision": "bf16-mixed"}},
    )

    new_cfg = OmegaConf.create(
        {
            "seed": 140799,
            "datamodule": {"config": {"max_seq_len": 256}},
            "trainer": {"accelerator": "auto", "precision": "bf16-true"},
            "lightning_module": {"model": {"precision": "bf16-true"}},
        }
    )

    with pytest.raises(ValueError, match="precision"):
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


def test_resume_survives_an_opt_in_knob_that_did_not_exist_yet(tmp_path: Path) -> None:
    """A run dir predating an opt-in feature flag resumes while the flag is left unset, at any depth.

    Ontology support (PR #32) added ``lightning_module.model.ontology_dir`` /
    ``cohort_vocab_fingerprint`` and ``datamodule.dataset_kwargs.ontology_dir`` to the shipped
    multitask config, all defaulting to ``null``.  Without this exemption every run dir started
    before that release would refuse to resume, naming keys whose value turns the feature off.
    """
    saved = tmp_path / "config.yaml"
    _write_cfg(saved, lightning_module={"model": {"max_windows": 5}})

    unset = OmegaConf.create(
        {
            "seed": 140799,
            "datamodule": {"config": {"max_seq_len": 256}, "dataset_kwargs": None},
            "lightning_module": {
                "model": {"max_windows": 5, "ontology_dir": None, "cohort_vocab_fingerprint": None}
            },
        }
    )
    validate_resume_directory(tmp_path, unset)

    # Turning the new knob on *is* structural drift: the resumed run would train a wider table
    # against ancestor-mixed rows the original never saw.
    turned_on = OmegaConf.create(
        {
            "seed": 140799,
            "datamodule": {"config": {"max_seq_len": 256}},
            "lightning_module": {"model": {"max_windows": 5, "ontology_dir": "/onto"}},
        }
    )
    with pytest.raises(ValueError, match="ontology_dir"):
        validate_resume_directory(tmp_path, turned_on)
