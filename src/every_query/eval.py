import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf
from train import collate_tasks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def values_as_list(**kwargs: Any) -> list[Any]:
    return list(kwargs.values())


def load_run_config(run_dir: Path):
    """Load the resolved config from a finished training run.

    Prefers resolved_config.yaml (all interpolations resolved). Falls back to config.yaml if needed.
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
        raise FileNotFoundError(f"Could not find resolved_config.yaml or config.yaml in {run_dir}")

    return cfg


def find_best_checkpoint(run_dir: Path) -> Path:
    """Use best_model.ckpt if present; otherwise fall back to last.ckpt."""
    best_model_ckpt = run_dir / "best_model.ckpt"
    if best_model_ckpt.is_file():
        return best_model_ckpt

    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt

    raise FileNotFoundError(f"No best_model.ckpt or last.ckpt found in {run_dir}.")


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
            if isinstance(v, int | float):
                print(f"{k:30s}: {v:.6f}")
            else:
                print(f"{k:30s}: {v}")
        print()
    else:
        # Uncommon, but handle multiple test loaders
        print("\n=== Held-out metrics (multiple loaders) ===")
        print(json.dumps(results, indent=2, default=_json_default))
        print()


@hydra.main(version_base="1.3", config_path="", config_name="eval_config.yaml")
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"{run_dir} is not a directory")

    # for OOD: Collate new tasks on held_out split if OOD queries set in eval_config
    # for ID: Or points to held_out tasks if query list has already been collated from training run
    task_dir = collate_tasks(cfg)
    if cfg.only_preprocess:
        print("Collate tasks complete. Existing.")
        sys.exit(0)

    # Load training config
    train_cfg = OmegaConf.load(run_dir / "resolved_config.yaml")

    # Point the train_fg at the task files created by collate
    train_cfg.datamodule.config.task_labels_dir = task_dir

    seed = train_cfg.get("seed", None)
    if seed is not None:
        logger.info(f"Seeding with seed={seed}")
        seed_everything(seed, workers=True)

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    # Instantiate datamodule, lightning_module, trainer from saved train_cfg
    logger.info("Instantiating datamodule...")
    D = instantiate(train_cfg.datamodule)

    logger.info("Instantiating lightning module (architecture only)...")
    M = instantiate(train_cfg.lightning_module)

    logger.info("Instantiating trainer...")
    trainer = instantiate(train_cfg.trainer)

    # Find checkpoint from this run and run test (held_out split = test)
    best_ckpt_path = find_best_checkpoint(run_dir)
    logger.info(f"Evaluating checkpoint: {best_ckpt_path}")

    results = trainer.test(
        model=M,
        datamodule=D,
        ckpt_path=str(best_ckpt_path),
    )

    # Print clean summary to stdout
    pretty_print_results(results)

    # Save full results JSON next to the run
    out_path = run_dir / "heldout_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    logger.info(f"Held-out results saved to {out_path}")


if __name__ == "__main__":
    main()
