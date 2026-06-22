"""End-to-end tests for the ``EQ_train`` CLI.

Covers the ``eq_trained_model_dir`` fixture (from #105) plus a resume-differential that
catches silent ``do_resume=True`` regressions — a reinitialize-on-resume bug would leave
the global_step unchanged and fail this test.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import torch
from omegaconf import OmegaConf

from conftest import run_and_check

if TYPE_CHECKING:
    from pathlib import Path


def _highest_step_checkpoint(output_dir: Path) -> tuple[Path, dict]:
    """Return the (path, loaded_ckpt_dict) for the checkpoint with the largest ``global_step``.

    Lightning's ``ModelCheckpoint(save_last=True)`` versions ``last.ckpt`` as ``last-v1.ckpt``
    when resuming into a directory that already has a ``last.ckpt`` — it keeps the old one in
    place.  Scanning all ``*.ckpt`` and picking the max-step one is robust to that behavior.
    """
    ckpts = sorted((output_dir / "checkpoints").glob("*.ckpt"))
    assert ckpts, f"no checkpoints under {output_dir / 'checkpoints'}"
    best_path, best_ckpt, best_step = None, None, -1
    for fp in ckpts:
        c = torch.load(fp, map_location="cpu", weights_only=False)
        if c["global_step"] > best_step:
            best_path, best_ckpt, best_step = fp, c, c["global_step"]
    return best_path, best_ckpt


def test_resolved_config_captures_overrides(
    eq_trained_model_dir: Path,
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
) -> None:
    """`resolved_config.yaml` records the fully-interpolated config with CLI overrides baked in.

    Catches regressions where Hydra overrides are silently dropped between CLI parse and on-disk persistence —
    downstream tools (evaluate, reproduce-this-run scripts) depend on this file being faithful.
    """
    resolved = eq_trained_model_dir / "resolved_config.yaml"
    assert resolved.exists(), f"resolved_config.yaml not written to {eq_trained_model_dir}"
    cfg = OmegaConf.load(resolved)
    # output_dir override → must reflect the actual tmp run dir, not the ??? sentinel.
    assert cfg.output_dir == str(eq_trained_model_dir), (
        f"resolved_config.output_dir={cfg.output_dir!r} does not match actual run dir "
        f"{eq_trained_model_dir!r}"
    )
    # task_labels_dir / tensorized_cohort_dir overrides must land the *actual* fixture
    # paths on disk, not a truthy-but-useless sentinel like '???'.  A pure truthiness
    # check would silently pass if Hydra dropped the override between CLI parse and
    # save (since '???' is truthy), so assert equality with the expected fixture paths.
    assert cfg.datamodule.config.task_labels_dir == str(eq_sampled_tasks_dir), (
        f"resolved_config.task_labels_dir={cfg.datamodule.config.task_labels_dir!r} does "
        f"not match fixture path {str(eq_sampled_tasks_dir)!r}"
    )
    assert cfg.datamodule.config.tensorized_cohort_dir == str(eq_preprocessed_dataset), (
        f"resolved_config.tensorized_cohort_dir={cfg.datamodule.config.tensorized_cohort_dir!r} "
        f"does not match fixture path {str(eq_preprocessed_dataset)!r}"
    )


def test_checkpoint_metadata(eq_trained_model_dir: Path) -> None:
    """Lightning checkpoint carries the expected training-state keys and matches ``max_steps=2``."""
    _, ckpt = _highest_step_checkpoint(eq_trained_model_dir)
    for key in ("state_dict", "optimizer_states", "global_step", "epoch", "hyper_parameters"):
        assert key in ckpt, f"checkpoint missing expected key {key!r}"
    # Demo config has max_steps=2; global_step advances once per optimizer step.
    assert ckpt["global_step"] == 2, (
        f"demo training ran for {ckpt['global_step']} steps, expected 2 per _demo_train.yaml"
    )
    # State dict is non-empty and carries the model-parameter keys Lightning persists
    # under the `HF_model.` / `model.` prefixes (loose check rather than a specific
    # architecture prefix — this test's job is to catch "checkpoint has no weights" as a
    # regression, not to validate the ModernBERT module tree).
    assert len(ckpt["state_dict"]) > 0
    assert any("HF_model" in k or "model." in k for k in ckpt["state_dict"]), (
        f"state_dict doesn't appear to contain model-related parameters: keys={list(ckpt['state_dict'])[:3]}"
    )


def _stage_resume_dir(src: Path, dst: Path) -> None:
    """Copy ``config.yaml`` / ``resolved_config.yaml`` / ``checkpoints/`` into a fresh dir.

    ``config.yaml`` is load-bearing — ``train.py`` uses its presence (``cfg_path.exists()``)
    to classify the output dir as populated and to enter the ``do_resume`` branch.
    Without it, ``do_resume=True`` would be silently ignored and a fresh training run would
    start instead.
    """
    for item in ("config.yaml", "resolved_config.yaml", "checkpoints"):
        if (src / item).is_dir():
            shutil.copytree(src / item, dst / item)
        else:
            shutil.copy2(src / item, dst / item)


def test_resume_actually_loads_checkpoint_state(
    eq_trained_model_dir: Path,
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
    tmp_path_factory,
) -> None:
    """``do_resume=True`` actually loads the prior checkpoint's state — not a silent restart.

    Two differentials, together, narrow down to "resume is really resuming":

    1. **No-op resume (``max_steps`` == starting ``global_step``)**: a real resume immediately
       exits ``trainer.fit`` because the max-steps budget is already spent — so both
       ``global_step`` and the weight tensors stay identical.  A silently-dropped resume
       that started fresh training would run 2 new optimizer steps against random weights
       and would land the state_dict somewhere completely different, with ``global_step``
       back at 2 but tensors mismatching the pre-run snapshot.  Verifying **weights
       unchanged** here is what distinguishes "resume" from "coincidentally-same-step
       fresh run" (both advance ``global_step`` to 2; only a real resume keeps the weights).
    2. **Advancing resume (``max_steps`` > starting ``global_step``)**: after a real resume
       from step 2 with ``max_steps=4``, at least one parameter tensor has to differ from
       the starting snapshot (otherwise the optimizer isn't training anything — zero-lr
       bug, frozen params, or ``trainer.fit`` not actually running).

    Bundling both into one test keeps them on the same staged resume-dir fixture so setup
    cost amortizes.  Pre-#86-style reinit-on-resume bugs fail the first check; zero-lr /
    gradient-path regressions fail the second.
    """
    # Stage the fixture's outputs into a fresh dir so we can mutate without polluting the
    # session-scoped fixture for other tests.
    resume_dir = tmp_path_factory.mktemp("eq_train_resume")
    _stage_resume_dir(eq_trained_model_dir, resume_dir)

    _, start_ckpt = _highest_step_checkpoint(resume_dir)
    starting_step = start_ckpt["global_step"]
    assert starting_step == 2, f"sanity: fixture trained to step 2, got {starting_step}"
    start_sd = start_ckpt["state_dict"]

    # ── (1) No-op resume — max_steps == starting global_step ─────────────────────
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train",
            f"output_dir={resume_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={eq_sampled_tasks_dir!s}",
            "do_overwrite=False",
            "do_resume=True",
            f"trainer.max_steps={starting_step}",
        ],
        timeout=300.0,
    )
    _, noop_ckpt = _highest_step_checkpoint(resume_dir)
    assert noop_ckpt["global_step"] == starting_step, (
        f"no-op resume (max_steps={starting_step}) advanced global_step "
        f"{starting_step} → {noop_ckpt['global_step']}; resume likely ignored and fresh "
        f"training started."
    )
    noop_sd = noop_ckpt["state_dict"]
    assert set(noop_sd.keys()) == set(start_sd.keys()), (
        "state_dict key set changed across no-op resume — architecture drift?"
    )
    differing = [k for k in start_sd if not torch.equal(start_sd[k], noop_sd[k])]
    assert not differing, (
        f"no-op resume (max_steps == starting_step) mutated {len(differing)} parameter "
        f"tensor(s): {differing[:3]}.  Weights must be identical if resume actually loaded "
        f"the prior checkpoint — if they differ, a fresh training run ran instead."
    )

    # ── (2) Advancing resume — max_steps > starting global_step ──────────────────
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train",
            f"output_dir={resume_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={eq_sampled_tasks_dir!s}",
            "do_overwrite=False",
            "do_resume=True",
            "trainer.max_steps=4",
        ],
        timeout=300.0,
    )
    _, resumed_ckpt = _highest_step_checkpoint(resume_dir)
    resumed_sd = resumed_ckpt["state_dict"]
    assert set(resumed_sd.keys()) == set(start_sd.keys()), (
        "state_dict key set changed across resume — architecture drift?"
    )
    changed = [k for k in start_sd if not torch.equal(start_sd[k], resumed_sd[k])]
    assert changed, (
        "advancing resume did not change any parameter tensors.  Likely a gradient-path "
        "regression (zero-lr, frozen params, or trainer.fit not actually training)."
    )
