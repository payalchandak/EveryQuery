"""Load ``.env`` so that ``${oc.env:...}`` interpolations in the Hydra configs can resolve.

This module used to additionally gate a fixed list of "required" environment variables for
*presence* before the Hydra config was composed.  That check was too blunt: it could not see
CLI overrides (``datamodule.config.task_labels_dir=/path`` makes the backing ``${oc.env:TASK_DIR}``
interpolation never evaluate, yet the gate still demanded the env var), and it required
``WANDB_ENTITY`` even when the logger was disabled.  Path/entity validation now happens
*post-compose* against the resolved config in ``every_query.train.train.validate_training_config``,
which can tell whether a node was overridden.  See #184.

``train.py::_init_env`` is the only caller; it loads ``.env`` here so the values are visible to the
lazy ``${oc.env:...}`` interpolations Hydra resolves later.
"""

from dotenv import load_dotenv


def ensure_env() -> None:
    """Load ``.env`` (if present) so ``${oc.env:...}`` interpolations can resolve."""
    load_dotenv()
