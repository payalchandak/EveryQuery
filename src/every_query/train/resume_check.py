"""Validate that a ``do_resume=True`` invocation matches the resumed-from run's config.

Ported from [MEDS_EIC_AR's ``training/files.py``](https://github.com/mmcdermott/MEDS_EIC_AR/blob/main/src/MEDS_EIC_AR/training/files.py)
with one substantive EQ-side tweak: the ``ALLOWED_DIFFERENCE_KEYS`` allow-list adds a handful
of training-schedule knobs (``trainer.max_steps``, ``trainer.max_epochs``,
``trainer.callbacks.early_stopping.patience``, ``datamodule.pin_memory``) that are *legitimate*
things to override on resume — bumping step count to train longer than originally planned is a
common workflow.

See #91 for the motivation and #31 for the original ``do_overwrite`` config-preservation bug
this complements.
"""

from pathlib import Path
from typing import Any

from meds_torchdata.types import PaddingSide, StaticInclusionMode, SubsequenceSamplingStrategy
from omegaconf import DictConfig, OmegaConf

ALLOWED_DIFFERENCE_KEYS = {
    # Lifecycle flags — by definition different on resume.
    "do_resume",
    "do_overwrite",
    # Paths that legitimately differ between runs of the same config.
    "output_dir",
    "log_dir",
    "trainer.logger.save_dir",
    "trainer.callbacks.model_checkpoint.dirpath",
    "trainer.default_root_dir",
    # Machine-local performance knobs — no effect on model behavior.
    "datamodule.num_workers",
    "datamodule.pin_memory",
    # Training-schedule overrides that are legitimate things to bump on resume.
    "trainer.max_steps",
    "trainer.max_epochs",
    "trainer.callbacks.early_stopping.patience",
}

STR_ENUM_PARAMS = {
    "seq_sampling_strategy": SubsequenceSamplingStrategy,
    "padding_side": PaddingSide,
    "static_inclusion_mode": StaticInclusionMode,
}


def resolve_enum_value(value: Any, enum_type: type) -> Any:
    """Resolves a value to its corresponding enum type, handling both string and enum inputs.

    Args:
        value: The value to resolve, which can be a string or an instance of the enum type.
        enum_type: The enum type to resolve the value to.

    Returns:
        The resolved enum value.

    Raises:
        ValueError: If the value cannot be resolved to the enum type.

    Examples:
        >>> resolve_enum_value("left", PaddingSide)
        <PaddingSide.LEFT: 'left'>
        >>> resolve_enum_value("LEFT", PaddingSide)
        <PaddingSide.LEFT: 'left'>
        >>> resolve_enum_value("RANDOM", SubsequenceSamplingStrategy)
        <SubsequenceSamplingStrategy.RANDOM: 'random'>
        >>> resolve_enum_value(SubsequenceSamplingStrategy.RANDOM, SubsequenceSamplingStrategy)
        <SubsequenceSamplingStrategy.RANDOM: 'random'>
        >>> resolve_enum_value(34, PaddingSide)
        Traceback (most recent call last):
            ...
        ValueError: Expected 34 to be a string or PaddingSide, got int.
    """
    match value:
        case str():
            return enum_type(value.lower())
        case enum_type():
            return value
        case _:
            raise ValueError(
                f"Expected {value} to be a string or {enum_type.__name__}, got {type(value).__name__}."
            )


def diff_configs(new_config: dict[str, Any], old_config: dict[str, Any]) -> dict[str, str]:
    """Compares two configurations and returns a dictionary mapping key names to any differences found.

    Handles the case-insensitive-enum edge case for ``STR_ENUM_PARAMS`` — values that differ only in
    casing or string-vs-enum representation are treated as equal so we don't false-positive on Hydra /
    OmegaConf / StrEnum round-trips.

    Args:
        new_config: The new configuration (the one the user just invoked).
        old_config: The old configuration (loaded from the resumed-from output dir).

    Returns:
        ``{dotted_key: human_diff_message}`` — empty iff the configs agree on every key that matters.

    Examples:
        >>> old_cfg = {"c": 1, "b": 3}
        >>> new_cfg = {"c": 1, "b": 2}
        >>> diff_configs(new_cfg, old_cfg)
        {'b': 'has value 3 (int) in the old config, but 2 (int) in the new config.'}
        >>> old_cfg = {"c": 1, "foo": {"bar": 3}}
        >>> new_cfg = {"c": 1, "b": 2, "foo": {}}
        >>> diff_configs(new_cfg, old_cfg)
        {'foo.bar': 'is present in the old config, but not in the new config.',
         'b': 'is not present in the old config, but is in the new config.'}

        Enum parameters with casing / string-vs-enum drift are not flagged:

        >>> old_cfg = {
        ...     "padding_side": "left", "seq_sampling_strategy": "RANDOM", "static_inclusion_mode": "omit"
        ... }
        >>> new_cfg = {
        ...     "padding_side": "LEFT", "seq_sampling_strategy": SubsequenceSamplingStrategy.RANDOM,
        ...     "static_inclusion_mode": "omit"
        ... }
        >>> diff_configs(new_cfg, old_cfg)
        {}
    """
    differences = {}
    for key, old_val in old_config.items():
        if key not in new_config:
            differences[key] = "is present in the old config, but not in the new config."
            continue

        new_val = new_config[key]

        if isinstance(old_val, dict) and isinstance(new_config[key], dict):
            nested_diffs = diff_configs(new_val, old_val)
            for nested_key, nested_diff in nested_diffs.items():
                differences[f"{key}.{nested_key}"] = nested_diff
            continue

        different = old_val != new_val

        if key in STR_ENUM_PARAMS:
            enum_type = STR_ENUM_PARAMS[key]
            old_val_enum = resolve_enum_value(old_val, enum_type)
            new_val_enum = resolve_enum_value(new_val, enum_type)
            different = old_val_enum != new_val_enum

        if different:
            differences[key] = (
                f"has value {old_val} ({type(old_val).__name__}) in the old config, "
                f"but {new_val} ({type(new_val).__name__}) in the new config."
            )

    for key in new_config:
        if key not in old_config:
            differences[key] = "is not present in the old config, but is in the new config."

    return differences


def validate_resume_directory(output_dir: Path, cfg: DictConfig) -> None:
    """Raise if the config in ``output_dir/config.yaml`` disagrees with ``cfg`` on any key that matters.

    "Matters" = any key not listed in ``ALLOWED_DIFFERENCE_KEYS``.  The allow-list covers lifecycle
    flags (``do_resume``, ``do_overwrite``), path fields that legitimately differ between runs, local
    performance knobs, and training-schedule overrides that are reasonable to bump on resume.  Any
    structural drift (model hyperparameters, dataset shape, seed, query codes) raises ``ValueError``
    with a per-key message.

    Args:
        output_dir: The run directory being resumed from.
        cfg: The configuration the user just invoked with ``do_resume=True``.

    Raises:
        FileNotFoundError: If ``output_dir/config.yaml`` does not exist.
        ValueError: If structural drift is detected.
    """
    old_cfg_fp = output_dir / "config.yaml"
    if not old_cfg_fp.is_file():
        raise FileNotFoundError(f"Configuration file {old_cfg_fp} does not exist in the output directory.")

    old_cfg = OmegaConf.load(old_cfg_fp)
    old_cfg = OmegaConf.to_container(old_cfg, resolve=True)
    new_cfg = OmegaConf.to_container(cfg, resolve=True)

    differences = diff_configs(new_cfg, old_cfg)

    err_lines = []
    for key, diff in differences.items():
        if key in ALLOWED_DIFFERENCE_KEYS:
            continue
        err_lines.append(f"  - key '{key}' {diff}")

    if err_lines:
        err_lines_str = "\n".join(err_lines)
        raise ValueError(
            f"Refusing to resume: the configuration at {old_cfg_fp} disagrees with the current "
            f"invocation on the following keys:\n{err_lines_str}\n"
            f"Either pass ``do_overwrite=True`` to start a fresh run, or align your overrides "
            f"with the original run's config."
        )
