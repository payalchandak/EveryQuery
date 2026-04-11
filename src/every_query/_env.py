"""Load .env and validate that required environment variables are set.

Imported early by both train.py and tasks.py so that a missing .env on a fresh
machine surfaces a single clear error rather than a mid-run KeyError or Hydra
InterpolationResolutionError.
"""

import os
import sys

from dotenv import load_dotenv

REQUIRED_ENV_VARS = (
    "PROJECT_DIR",
    "OUTPUT_DIR",
    "TASK_DIR",
    "PROCESSED",
    "INTERMEDIATE",
    "FINAL_DATA_DIR",
    "WANDB_ENTITY",
)


def ensure_env() -> None:
    load_dotenv()
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        joined = ", ".join(missing)
        print(
            f"ERROR: required environment variables not set: {joined}\n"
            "Copy .env.example to .env and fill in machine-specific paths, "
            "or export these variables before running.",
            file=sys.stderr,
        )
        sys.exit(1)
