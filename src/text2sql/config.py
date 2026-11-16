"""Training config loading/validation. Needs PyYAML only (no GPU deps), so it is
unit-testable on CPU."""

from __future__ import annotations

from pathlib import Path

REQUIRED_SECTIONS = ("model", "quantization", "lora", "data", "training")
REQUIRED_KEYS = {
    "model": ("name", "max_length"),
    "lora": ("r", "alpha", "dropout", "target_modules"),
    "data": ("dataset", "train_split", "eval_split"),
    "training": ("output_dir",),
}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply dotted `key.path=value` overrides in place (values parsed as YAML),
    e.g. `training.bf16=false` or `data.max_train_samples=2000`. Re-validates."""
    import yaml

    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        parts = key.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot set {key!r}: {part!r} is not a mapping")
        node[parts[-1]] = yaml.safe_load(raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping")
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"missing sections: {missing}")
    for section, keys in REQUIRED_KEYS.items():
        absent = [k for k in keys if k not in cfg[section]]
        if absent:
            raise ConfigError(f"missing keys in '{section}': {absent}")
    # These are controlled by the script, not the YAML, to keep train/eval consistent.
    for reserved in ("max_length", "completion_only_loss", "dataset_text_field"):
        if reserved in cfg["training"]:
            raise ConfigError(f"'training.{reserved}' is set by train.py; remove it from the YAML")
