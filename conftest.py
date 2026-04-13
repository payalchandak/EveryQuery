"""Root conftest: set dummy env vars so ensure_env() does not sys.exit during test collection."""

import os

_DUMMY_ENV = {
    "PROJECT_DIR": "/tmp/eq-test",
    "OUTPUT_DIR": "/tmp/eq-test/results",
    "TASK_DIR": "/tmp/eq-test/tasks",
    "PROCESSED": "/tmp/eq-test/processed",
    "INTERMEDIATE": "/tmp/eq-test/intermediate",
    "FINAL_DATA_DIR": "/tmp/eq-test/processed",
    "WANDB_ENTITY": "test",
}

for key, val in _DUMMY_ENV.items():
    os.environ.setdefault(key, val)
