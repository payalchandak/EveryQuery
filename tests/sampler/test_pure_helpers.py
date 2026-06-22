"""Pure leaf helpers shared across the pipeline (not tied to a single stage).

- ``TestPrimitives`` — ``derive_seed`` axis separation, ``read_query_codes`` resolution branches,
  ``compute_max_time_per_subject``.
- ``TestResolvePath`` / ``TestResolveTrainingTaskPaths`` — the ``override > env > raise`` path
  contract (the wrapper is trimmed to the cases that add information beyond the primitive).
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
    compute_max_time_per_subject,
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

    def test_read_query_codes_uses_processed_fallback(self, monkeypatch, tmp_path, synthetic_query_codes):
        processed = tmp_path / "processed"
        metadata_dir = processed / "metadata"
        metadata_dir.mkdir(parents=True)
        pl.DataFrame({"code": synthetic_query_codes}).write_parquet(metadata_dir / "codes.parquet")

        monkeypatch.setenv("PROCESSED", str(processed))

        assert st.read_query_codes(None) == sorted(synthetic_query_codes)

    def test_read_query_codes_requires_processed_when_null(self, monkeypatch):
        monkeypatch.delenv("PROCESSED", raising=False)

        with pytest.raises(ValueError, match="query_codes is null"):
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

    def test_compute_max_time_per_subject(self, synthetic_events):
        max_df = compute_max_time_per_subject(synthetic_events)
        assert set(max_df.columns) == {"subject_id", "max_time"}
        # Every subject has 30 events; max_time is the 30th event time per subject.
        for subj in [1, 2, 3]:
            subj_max = synthetic_events.filter(pl.col("subject_id") == subj)["time"].max()
            row = max_df.filter(pl.col("subject_id") == subj).row(0, named=True)
            assert row["max_time"] == subj_max


class TestResolvePath:
    """``_resolve_path`` is how ``main()`` threads explicit path roots / env fallbacks / required.

    Directly exercising the helper lets us pin the fallback matrix without spinning up a full Hydra run per
    case.
    """

    def test_explicit_cfg_value_wins(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "/from/env")
        result = st._resolve_path("/from/cfg", "MY_VAR", "data_dir")
        assert result == Path("/from/cfg")

    def test_env_fallback_when_cfg_is_none(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "/from/env")
        result = st._resolve_path(None, "MY_VAR", "data_dir")
        assert result == Path("/from/env")

    def test_raises_when_both_unset(self, monkeypatch):
        monkeypatch.delenv("MY_VAR", raising=False)
        with pytest.raises(ValueError, match="data_dir must be set"):
            st._resolve_path(None, "MY_VAR", "data_dir")

    def test_error_message_mentions_env_var_and_dotenv(self, monkeypatch):
        monkeypatch.delenv("INTERMEDIATE", raising=False)
        with pytest.raises(ValueError) as excinfo:
            st._resolve_path(None, "INTERMEDIATE", "data_dir")
        msg = str(excinfo.value)
        assert "INTERMEDIATE" in msg
        assert ".env" in msg  # dotenv hint

    def test_empty_env_var_is_treated_as_unset(self, monkeypatch):
        """An explicitly-empty env var should not be taken as a valid path."""
        monkeypatch.setenv("MY_VAR", "")
        with pytest.raises(ValueError, match="must be set"):
            st._resolve_path(None, "MY_VAR", "data_dir")


class TestResolveTrainingTaskPaths:
    """The redesigned sampler's three path roots (``override > env > raise``).

    Trimmed to the cases the wrapper adds *on top of* ``_resolve_path``: the sibling artifacts-root
    derivation, that the artifacts root has no env var of its own, the override-beats-env precedence,
    and that each missing required root raises.  The bare env-fallback / null-override cases are
    already proven by ``TestResolvePath`` and are not re-tested through the wrapper.
    """

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        # Isolate from the developer's own .env / shell so assertions are deterministic.
        monkeypatch.delenv("INTERMEDIATE", raising=False)
        monkeypatch.delenv("TRAINING_TASKS_DIR", raising=False)

    def test_default_artifacts_dir_is_a_sibling(self):
        assert default_artifacts_dir(Path("/x/y/tasks")) == Path("/x/y/tasks_artifacts")
        # Sibling, never nested under the final-output root (invariant 7).
        assert default_artifacts_dir(Path("/x/y/tasks")).parent == Path("/x/y/tasks").parent

    def test_artifacts_dir_has_no_env_var(self, monkeypatch):
        # Even if someone exports it, the resolver ignores it and uses the sibling default.
        monkeypatch.setenv("INTERMEDIATE", "/env/data")
        monkeypatch.setenv("TRAINING_TASKS_DIR", "/env/tasks")
        monkeypatch.setenv("TRAINING_TASK_ARTIFACTS_DIR", "/env/should_be_ignored")
        _, _, arts = resolve_training_task_paths()
        assert arts == Path("/env/tasks_artifacts")

    def test_missing_path_to_data_raises(self, monkeypatch):
        monkeypatch.delenv("INTERMEDIATE", raising=False)
        monkeypatch.setenv("TRAINING_TASKS_DIR", "/env/tasks")
        with pytest.raises(ValueError, match="data_dir"):
            resolve_training_task_paths()

    def test_missing_training_tasks_dir_raises(self, monkeypatch):
        monkeypatch.setenv("INTERMEDIATE", "/env/data")
        monkeypatch.delenv("TRAINING_TASKS_DIR", raising=False)
        with pytest.raises(ValueError, match="out_dir"):
            resolve_training_task_paths()

    def test_cfg_overrides_take_precedence_over_env(self, monkeypatch):
        # override > env: cfg.data_dir / cfg.out_dir win even when the env vars are set, and the
        # artifacts root derives from the *override* out_dir.
        monkeypatch.setenv("INTERMEDIATE", "/env/data")
        monkeypatch.setenv("TRAINING_TASKS_DIR", "/env/tasks")
        cfg = OmegaConf.create({"data_dir": "/cli/data", "out_dir": "/cli/tasks"})
        data, tasks, arts = resolve_training_task_paths(cfg)
        assert data == Path("/cli/data")
        assert tasks == Path("/cli/tasks")
        assert arts == Path("/cli/tasks_artifacts")


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
    input roots are optional null overrides, the derived roots are never config keys, and the sampling knobs
    exist with their expected types.
    """

    @staticmethod
    def _load():
        path = files("every_query") / "generate_tasks" / "configs" / "sample_training_tasks_config.yaml"
        return OmegaConf.load(str(path))

    def test_required_keys_present_and_typed(self):
        cfg = self._load()
        # The two input roots are optional CLI overrides defaulting to null (override > env > raise).
        assert "data_dir" in cfg and cfg.data_dir is None
        assert "out_dir" in cfg and cfg.out_dir is None
        # The derived roots are never config keys.
        for derived_key in ("path_to_data", "training_tasks_dir", "training_task_artifacts_dir"):
            assert derived_key not in cfg
        # Core sampling knobs exist with the expected shape.
        assert isinstance(cfg.min_prediction_times_per_subject, int)
        assert isinstance(cfg.duration_distribution, str)
        assert cfg.max_workers is None  # uncapped by default
