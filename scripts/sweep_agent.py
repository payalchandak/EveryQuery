"""W&B Sweep agent that translates sweep params into Hydra overrides and runs train.py.

Usage (called automatically by `wandb agent`):
    python scripts/sweep_agent.py

The script:
  1. Initialises a wandb run to receive the sweep config
  2. Maps sweep parameters to Hydra overrides
  3. Computes batch_size and accumulate_grad_batches from effective_batch_size
  4. Launches train.py as a subprocess with the correct Hydra overrides
"""

import subprocess
import sys

import wandb

# Fixed per-GPU batch size options to try. We pick the largest that evenly
# divides the effective batch size so gradient accumulation stays integral.
BASE_BATCH_SIZES = [40, 32, 24, 16, 8]


def compute_batch_accum(effective_batch_size: int) -> tuple[int, int]:
    for bs in BASE_BATCH_SIZES:
        if effective_batch_size % bs == 0:
            return bs, effective_batch_size // bs
    # Fallback: just use effective as batch_size with no accumulation
    return effective_batch_size, 1


def main() -> None:
    run = wandb.init()
    cfg = dict(run.config)

    lr = cfg["lr"]
    num_hidden_layers = cfg["num_hidden_layers"]
    effective_batch_size = cfg["effective_batch_size"]

    batch_size, accum = compute_batch_accum(effective_batch_size)

    overrides = [
        f"lightning_module.optimizer.lr={lr}",
        f"lightning_module.model.num_hidden_layers={num_hidden_layers}",
        f"datamodule.batch_size={batch_size}",
        f"trainer.accumulate_grad_batches={accum}",
    ]

    cmd = [
        sys.executable,
        "src/every_query/train.py",
        *overrides,
    ]

    print(f"[sweep_agent] effective_batch_size={effective_batch_size} -> "
          f"batch_size={batch_size}, accumulate_grad_batches={accum}")
    print(f"[sweep_agent] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    run.finish(exit_code=result.returncode)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
