"""Pure leaf helpers shared across the pipeline (not tied to a single stage).

- ``TestPrimitives`` — ``derive_seed`` axis separation, ``read_query_codes`` resolution branches.
- ``TestResolveTrainingTaskPaths`` — the three path roots resolve from required Hydra keys (#235).
- ``TestArtifactLayout`` — the two-root *safety* invariants (the trivial path-string equalities
  are covered by the source doctests and intentionally not duplicated here).
- ``TestAtomicWrite`` — round-trip + the failure path leaves no orphan temp.
- ``TestResolveWorkers`` — Stage 4 worker-pool sizing (SLURM env precedence, cpu_count fallback).
- ``TestRedesignConfigFile`` — the config exposes the spec's keys with the right shape (presence +
  type, not pinned literal default values).
"""

import os
from importlib.resources import files
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

from every_query.generate_tasks import sample_tasks as st
from every_query.generate_tasks.sample_tasks import (
    default_artifacts_dir,
    final_output_path,
    index_path,
    prediction_time_counts_path,
    prediction_times_path,
    resolve_training_task_paths,
    resolve_workers,
)
from every_query.utils.seeds import derive_seed


class TestPrimitives:
    def test_derive_seed_is_stable_across_calls(self):
        assert derive_seed(1, "tasks", 0) == derive_seed(1, "tasks", 0)
        assert derive_seed(1, "contexts", "shard_a", 7) == derive_seed(1, "contexts", "shard_a", 7)

    def test_derive_seed_separates_axes(self):
        assert derive_seed(1, "tasks", 0) != derive_seed(1, "tasks", 1)
        assert derive_seed(1, "contexts", "a", 0) != derive_seed(1, "contexts", "b", 0)
        assert derive_seed(1, "contexts", "a", 0) != derive_seed(1, "contexts", "a", 1)
        assert derive_seed(1, "tasks", 0) != derive_seed(2, "tasks", 0)

    def test_read_query_codes_reads_metadata_dir(self, tmp_path, synthetic_query_codes):
        codes_dir = tmp_path / "processed"
        metadata_dir = codes_dir / "metadata"
        metadata_dir.mkdir(parents=True)
        pl.DataFrame({"code": synthetic_query_codes}).write_parquet(metadata_dir / "codes.parquet")

        assert st.read_query_codes(codes_dir) == sorted(synthetic_query_codes)

    def test_read_query_codes_requires_value_when_unset(self):
        with pytest.raises(ValueError, match="query_codes is unset"):
            st.read_query_codes(None)

    def test_read_query_codes_reads_codes_key_yaml(self, tmp_path):
        codes_yaml = tmp_path / "allowed_codes.yaml"
        codes_yaml.write_text("codes:\n  - A\n  - B\n  - A\n")

        assert st.read_query_codes(codes_yaml) == ["A", "B"]

    def test_read_query_codes_reads_flat_yaml_list(self, tmp_path):
        codes_yaml = tmp_path / "allowed_codes.yaml"
        codes_yaml.write_text("- A\n- B\n- A\n")

        assert st.read_query_codes(codes_yaml) == ["A", "B"]

    def test_read_query_codes_rejects_yaml_mapping_without_codes(self, tmp_path):
        codes_yaml = tmp_path / "allowed_codes.yaml"
        codes_yaml.write_text("query_codes:\n  - A\n  - B\n")

        with pytest.raises(ValueError, match="list of codes"):
            st.read_query_codes(codes_yaml)

    def test_read_query_codes_rejects_non_list_yaml(self, tmp_path):
        codes_yaml = tmp_path / "allowed_codes.yaml"
        codes_yaml.write_text("codes: A\n")

        with pytest.raises(ValueError, match="list of codes"):
            st.read_query_codes(codes_yaml)

    def test_read_query_codes_deduplicates_inline_list(self):
        assert st.read_query_codes(["A", "B", "A"]) == ["A", "B"]


class TestResolveTrainingTaskPaths:
    """The redesigned sampler's three path roots, resolved from required Hydra keys (#235).

    Covers the sibling artifacts-root derivation, that the artifacts root has no key of its own, that
    the keys resolve straight from cfg (no env fallback), and that a missing (``???``) required root
    raises OmegaConf's ``MissingMandatoryValue``.
    """

    def test_default_artifacts_dir_is_a_sibling(self):
        assert default_artifacts_dir(Path("/x/y/tasks")) == Path("/x/y/tasks_artifacts")
        # Sibling, never nested under the final-output root (invariant 7).
        assert default_artifacts_dir(Path("/x/y/tasks")).parent == Path("/x/y/tasks").parent

    def test_resolves_from_cfg_keys(self):
        cfg = OmegaConf.create({"data_dir": "/cli/data", "out_dir": "/cli/tasks"})
        data, tasks, arts = resolve_training_task_paths(cfg)
        assert data == Path("/cli/data")
        assert tasks == Path("/cli/tasks")
        # artifacts root derives from out_dir and has no key/env var of its own.
        assert arts == Path("/cli/tasks_artifacts")

    def test_missing_required_root_raises(self):
        # `???` is OmegaConf MISSING; it must raise a clear ValueError rather than fall back to any
        # env var or slip through as a literal `None`/empty path.
        cfg = OmegaConf.create({"data_dir": "???", "out_dir": "/cli/tasks"})
        with pytest.raises(ValueError, match="data_dir is unset or empty"):
            resolve_training_task_paths(cfg)

    def test_empty_override_raises_not_silent_none(self):
        # `data_dir=$VAR` with an unexported $VAR expands to an empty override that Hydra parses as
        # None, overriding `???`.  Must raise up front, not become Path("None") and fail in Stage 0.
        for bad in (None, "", "   "):
            cfg = OmegaConf.create({"data_dir": bad, "out_dir": "/cli/tasks"})
            with pytest.raises(ValueError, match="data_dir is unset or empty"):
                resolve_training_task_paths(cfg)


class TestArtifactLayout:
    """The two-root, never-nested on-disk layout's *safety* invariants (issue #204).

    The exact path strings are pinned by the source doctests on ``final_output_path`` /
    ``prediction_time_counts_path`` / ``prediction_times_path`` / ``index_path``; here we assert only
    the safety contract those strings must satisfy — the final root holds nothing ``_``-prefixed,
    every intermediate lives under the disjoint artifacts root, and ``rm -rf`` of either can't touch
    the other (invariant 7).
    """

    TASKS = Path("/x/tasks")
    ARTS = Path("/x/tasks_artifacts")

    def test_final_root_split_dir_holds_only_shard_parquets(self):
        # The final-output split dir must be directly glob-consumable as `{split}/*.parquet`: no
        # `_`-prefixed entries (those all live under the artifacts root).
        name = final_output_path(self.TASKS, "train", "0").name
        assert name == "0.parquet"
        assert not name.startswith("_")

    def test_every_intermediate_is_under_the_artifacts_root_not_the_dataset(self):
        # Invariant 7: no intermediate may resolve inside the final-output root.
        intermediates = [
            prediction_time_counts_path(self.ARTS, "train"),
            prediction_times_path(self.ARTS, "train", "0"),
            index_path(self.ARTS, "train", "0"),
        ]
        for p in intermediates:
            assert self.ARTS in p.parents
            assert self.TASKS not in p.parents

    def test_default_artifacts_root_is_disjoint_so_rm_rf_cannot_touch_dataset(self):
        # The sibling default (from #203) is what makes `rm -rf {artifacts}` safe: the artifacts root
        # is neither equal to nor nested under the dataset root, and vice versa.
        arts = default_artifacts_dir(self.TASKS)
        assert arts != self.TASKS
        assert self.TASKS not in arts.parents
        assert arts not in self.TASKS.parents


class TestAtomicWrite:
    """``_atomic_write_parquet`` writes via a unique sibling temp + ``os.replace`` so a present file is always
    complete and a failed write leaves no orphan."""

    def test_atomic_write_parquet_round_trip(self, tmp_path):
        fp = tmp_path / "out.parquet"
        df = pl.DataFrame({"a": [1, 2, 3]})
        st._atomic_write_parquet(df, fp)
        assert fp.exists()
        loaded = pl.read_parquet(fp)
        assert loaded.equals(df)
        orphans = [p for p in tmp_path.iterdir() if p.name != "out.parquet"]
        assert orphans == []

    def test_failure_leaves_no_orphan_temp(self, tmp_path, monkeypatch):
        """If the commit (``os.replace``) fails, the helper must clean up its sibling temp and re-raise — no
        partial ``.out.parquet.tmp.*`` is left behind and no final file appears."""
        fp = tmp_path / "out.parquet"
        df = pl.DataFrame({"a": [1, 2, 3]})

        def _boom(*args, **kwargs):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(st.os, "replace", _boom)
        with pytest.raises(OSError, match="simulated replace failure"):
            st._atomic_write_parquet(df, fp)

        assert not fp.exists()
        assert list(tmp_path.glob(".out.parquet.tmp.*")) == []

    def test_unique_tmp_path_matches_clean_glob(self, tmp_path):
        """The temp name produced by ``_unique_tmp_path`` must match the ``_clean_stale_temps`` glob.

        Regression for the bug where ``mkstemp(suffix=".tmp")`` emitted ``.{name}.<random>.tmp`` while
        cleanup globbed ``.{name}.tmp.*`` — the two never agreed, so orphaned temps were never cleaned.
        Pin the contract: a real ``_unique_tmp_path`` temp is found by the cleanup glob and removed.
        """
        fp = tmp_path / "0.parquet"
        tmp = st._unique_tmp_path(fp)
        try:
            assert tmp.parent == fp.parent
            assert tmp in set(tmp_path.glob(f".{fp.name}.tmp.*"))
        finally:
            tmp.unlink(missing_ok=True)

        # And it does NOT collide with the final-root ``*.parquet`` safety glob (invariant 7).
        recreated = st._unique_tmp_path(fp)
        try:
            assert recreated not in set(tmp_path.glob("*.parquet"))
            assert st._clean_stale_temps(tmp_path, "0") == 1
            assert not recreated.exists()
        finally:
            recreated.unlink(missing_ok=True)


class TestResolveWorkers:
    """Stage 4 worker-pool sizing: SLURM env precedence, cpu_count fallback, downward-only cap."""

    @pytest.fixture(autouse=True)
    def _clear_slurm_env(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    def test_prefers_cpus_per_task_over_cpus_on_node(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
        assert resolve_workers() == 4

    def test_falls_back_to_cpus_on_node(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
        assert resolve_workers() == 8

    def test_falls_back_to_os_cpu_count_without_slurm(self):
        assert resolve_workers() == (os.cpu_count() or 1)

    def test_max_workers_caps_downward_only(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        assert resolve_workers(2) == 2  # lower cap applies
        assert resolve_workers(16) == 8  # higher cap is ignored (never exceed cores)

    def test_max_workers_none_returns_cores(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
        assert resolve_workers(None) == 6


class TestRedesignConfigFile:
    """``sample_training_tasks_config.yaml`` exposes the spec's keys with the right *shape*.

    We assert key presence and type, not pinned literal default values (a brittle change-detector): the two
    input roots are mandatory (``???``) Hydra args, the derived roots are never config keys, and the sampling
    knobs exist with their expected types.
    """

    @staticmethod
    def _load():
        path = files("every_query") / "generate_tasks" / "configs" / "sample_training_tasks_config.yaml"
        return OmegaConf.load(str(path))

    def test_required_keys_present_and_typed(self):
        cfg = self._load()
        # The two input roots are mandatory (`???`) Hydra args — no env fallback (#235).
        assert OmegaConf.is_missing(cfg, "data_dir")
        assert OmegaConf.is_missing(cfg, "out_dir")
        # The derived roots are never config keys.
        for derived_key in ("path_to_data", "training_tasks_dir", "training_task_artifacts_dir"):
            assert derived_key not in cfg
        # Core sampling knobs exist with the expected shape.
        assert isinstance(cfg.min_prediction_times_per_subject, int)
        assert isinstance(cfg.duration_distribution, str)
        assert cfg.max_workers is None  # uncapped by default
