from pathlib import Path

import pytest

pytest.importorskip("yaml")

from text2sql.config import ConfigError, apply_overrides, load_config, validate_config  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qlora_llama32_3b.yaml"


def test_shipped_config_is_valid():
    cfg = load_config(CONFIG)
    assert cfg["model"]["name"] == "unsloth/Llama-3.2-3B-Instruct"
    assert cfg["training"]["report_to"] == "wandb"
    assert cfg["seed"] == 42


def test_missing_section():
    cfg = load_config(CONFIG)
    del cfg["lora"]
    with pytest.raises(ConfigError, match="lora"):
        validate_config(cfg)


def test_missing_key():
    cfg = load_config(CONFIG)
    del cfg["data"]["dataset"]
    with pytest.raises(ConfigError, match="dataset"):
        validate_config(cfg)


def test_overrides_parse_yaml_values():
    cfg = apply_overrides(
        load_config(CONFIG),
        ["training.bf16=false", "data.max_train_samples=2000", "training.learning_rate=1.0e-4",
         "data.dataset=me/data", "run_name=smoke", "wandb.entity=team"],
    )
    assert cfg["training"]["bf16"] is False
    assert cfg["data"]["max_train_samples"] == 2000
    assert cfg["training"]["learning_rate"] == pytest.approx(1e-4)
    assert cfg["data"]["dataset"] == "me/data"
    assert cfg["run_name"] == "smoke"
    assert cfg["wandb"]["entity"] == "team"


@pytest.mark.parametrize("bad", ["no_equals", "=1", "seed.x=1"])
def test_bad_overrides(bad):
    with pytest.raises(ConfigError):
        apply_overrides(load_config(CONFIG), [bad])


def test_override_cannot_set_reserved_key():
    with pytest.raises(ConfigError, match="max_length"):
        apply_overrides(load_config(CONFIG), ["training.max_length=1024"])


def test_reserved_training_key():
    cfg = load_config(CONFIG)
    cfg["training"]["max_length"] = 1024
    with pytest.raises(ConfigError, match="max_length"):
        validate_config(cfg)
