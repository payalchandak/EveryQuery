import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
from omegaconf import open_dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def values_as_list(**kwargs: Any) -> list[Any]:
    return list(kwargs.values())


def load_run_config(run_dir: Path):
    """
    Load the resolved config from a finished training run.
    Prefers resolved_config.yaml (all interpolations resolved).
    Falls back to config.yaml if needed.
    """
    resolved_cfg_path = run_dir / "resolved_config.yaml"
    cfg_path = run_dir / "config.yaml"

    if resolved_cfg_path.is_file():
        logger.info(f"Loading resolved config from {resolved_cfg_path}")
        cfg = OmegaConf.load(resolved_cfg_path)
    elif cfg_path.is_file():
        logger.info(f"Loading config from {cfg_path}")
        cfg = OmegaConf.load(cfg_path)
    else:
        raise FileNotFoundError(
            f"Could not find resolved_config.yaml or config.yaml in {run_dir}"
        )

    return cfg


def find_best_checkpoint(run_dir: Path) -> Path:
    """
    Use best_model.ckpt if present; otherwise fall back to last.ckpt.
    """
    best_model_ckpt = run_dir / "best_model.ckpt"
    if best_model_ckpt.is_file():
        return best_model_ckpt

    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt

    raise FileNotFoundError(
        f"No best_model.ckpt or last.ckpt found in {run_dir}."
    )


def _json_default(obj: Any) -> Any:
    """Helper to make test results JSON-serializable."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        # scalar → item, otherwise list
        return obj.item() if obj.ndim == 0 else obj.tolist()
    return str(obj)


def pretty_print_results(results: list[dict[str, Any]]) -> None:
    """Print test results in a clean, readable way."""
    if not results:
        print("No results returned from trainer.test()")
        return

    if len(results) == 1:
        print("\n=== Held-out metrics ===")
        for k in sorted(results[0].keys()):
            v = results[0][k]
            if isinstance(v, (int, float)):
                print(f"{k:30s}: {v:.6f}")
            else:
                print(f"{k:30s}: {v}")
        print()
    else:
        # Uncommon, but handle multiple test loaders
        print("\n=== Held-out metrics (multiple loaders) ===")
        print(json.dumps(results, indent=2, default=_json_default))
        print()


def main(run_dir_str: str) -> None:
    run_dir = Path(run_dir_str).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"{run_dir} is not a directory")

    # 1) Load config from this run
    cfg = load_run_config(run_dir)

    # Make extra sure we don't overwrite anything by accident
    cfg.do_overwrite = False
    cfg.do_resume = False
    cfg.output_dir = str(run_dir)

    # 2) Make eval reproducible: same seed & matmul precision as training
    seed = cfg.get("seed", None)
    if seed is not None:
        logger.info(f"Seeding with seed={seed}")
        seed_everything(seed, workers=True)

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    # 3) Disable W&B (and any other logger) for this eval
    #    - kill Trainer.logger
    #    - also hard-disable wandb via env as a safety net
    os.environ.setdefault("WANDB_MODE", "disabled")
    if "trainer" in cfg and hasattr(cfg.trainer, "logger"):
        logger.info("Disabling trainer logger for evaluation (no new W&B runs).")
        cfg.trainer.logger = None

    # 4) Instantiate datamodule, lightning_module, trainer from saved cfg
    logger.info("Instantiating datamodule...")
    datamodule = instantiate(cfg.datamodule)

    logger.info("Instantiating lightning module (architecture only)...")
    lightning_module = instantiate(cfg.lightning_module)

    logger.info("Instantiating trainer...")
    trainer = instantiate(cfg.trainer)

    # 5) Find checkpoint from this run and run test (held_out split = test)
    best_ckpt_path = find_best_checkpoint(run_dir)
    logger.info(f"Evaluating checkpoint: {best_ckpt_path}")

    results = trainer.test(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=str(best_ckpt_path),
    )

    # 6) Print clean summary to stdout
    pretty_print_results(results)

    # 7) Save full results JSON next to the run
    out_path = run_dir / "heldout_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    logger.info(f"Held-out results saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python eval_held_out.py /path/to/run_dir\n"
            "where /path/to/run_dir contains best_model.ckpt and resolved_config.yaml"
        )
        sys.exit(1)

    main(sys.argv[1])

