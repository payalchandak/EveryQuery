"""Regression tests for the **real** train config's Hydra run/sweep dir wiring (#251).

These deliberately don't run training. Hydra creates `run.dir` / `sweep.dir` *before* `main()`
executes, so pointing the data dirs at non-existent paths lets `main()` fail fast (in
`validate_training_config`) while still having exercised every interpolation in `config.yaml`'s
hydra block: `${output_dir}/${now:}`, the per-job `sweep.subdir`, and `override_dirname.exclude_keys`.
The integration suites all use `_demo_train.yaml`, which hardwires `default_root_dir: ${output_dir}`
and has no hydra block — so without these, the real config's uniqueness machinery ships untested.
"""

import subprocess
import sys
from pathlib import Path


def _run_real_config(output_dir: Path, *extra: str, multirun: bool = False) -> subprocess.CompletedProcess:
    """Invoke the real (default) train config with bogus data dirs so it fails fast after Hydra has already
    created the run/sweep dir."""
    cmd = [sys.executable, "-m", "every_query.train.train"]
    if multirun:
        cmd.append("-m")
    cmd += [
        f"output_dir={output_dir}",
        "datamodule.config.tensorized_cohort_dir=/eq_nonexistent_cohort",
        "datamodule.config.task_labels_dir=/eq_nonexistent_tasks",
        "trainer.logger=false",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180)


def test_single_run_creates_one_timestamped_subdir(tmp_path):
    base = tmp_path / "out"
    _run_real_config(base)  # expected to fail on the bogus data dir; we only check dir creation

    days = list(base.iterdir())
    assert len(days) == 1, f"expected one <date> dir under {base}, got {[d.name for d in days]}"
    times = list(days[0].iterdir())
    assert len(times) == 1, f"expected one <time> dir under {days[0]}, got {[t.name for t in times]}"
    # run.dir = ${output_dir}/<YYYY-MM-DD>/<HH-MM-SS>
    assert len(days[0].name) == len("2026-06-24")
    assert len(times[0].name) == len("14-03-09")


def test_sweep_creates_one_numbered_subdir_per_job(tmp_path):
    base = tmp_path / "sweep"
    _run_real_config(base, "lightning_module.optimizer.lr=1e-5,1e-4", multirun=True)

    # All jobs of one launch share a single timestamped folder (${now:} fixed per launch).
    days = list(base.iterdir())
    assert len(days) == 1, f"sweep should share one <date> dir, got {[d.name for d in days]}"
    times = list(days[0].iterdir())
    assert len(times) == 1, f"sweep should share one <time> dir, got {[t.name for t in times]}"

    subdirs = [p.name for p in times[0].iterdir() if p.is_dir()]
    # subdir is ${hydra.job.num}: one numbered dir per sweep job, no override args in the name.
    assert sorted(subdirs) == ["0", "1"], f"expected one job.num subdir per swept lr, got {subdirs}"
