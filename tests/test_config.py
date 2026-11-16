from pathlib import Path

import pytest

pytest.importorskip("yaml")

from text2sql.config import (  # noqa: E402
    ConfigError,
    adapt_training_args,
    apply_overrides,
    load_config,
    validate_config,
)

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


V5_FIELDS = {"output_dir", "warmup_steps", "eval_strategy", "learning_rate", "max_length", "completion_only_loss"}
V4_FIELDS = V5_FIELDS | {"warmup_ratio"}


def test_warmup_ratio_translated_for_v5():
    kwargs, notes = adapt_training_args({"output_dir": "o", "warmup_ratio": 0.03}, V5_FIELDS)
    assert kwargs == {"output_dir": "o", "warmup_steps": 0.03}
    assert notes and "warmup_steps" in notes[0]


def test_warmup_ratio_kept_when_supported():
    kwargs, notes = adapt_training_args({"output_dir": "o", "warmup_ratio": 0.03}, V4_FIELDS)
    assert kwargs == {"output_dir": "o", "warmup_ratio": 0.03} and notes == []


def test_warmup_ratio_dropped_if_steps_set():
    kwargs, _ = adapt_training_args({"output_dir": "o", "warmup_ratio": 0.03, "warmup_steps": 10}, V5_FIELDS)
    assert kwargs == {"output_dir": "o", "warmup_steps": 10}


def test_eval_strategy_renamed_for_old_versions():
    old = {"output_dir", "evaluation_strategy"}
    kwargs, _ = adapt_training_args({"output_dir": "o", "eval_strategy": "steps"}, old)
    assert kwargs == {"output_dir": "o", "evaluation_strategy": "steps"}


def test_all_unknown_keys_reported_together():
    with pytest.raises(ConfigError, match=r"\['bogus_a', 'bogus_b'\]"):
        adapt_training_args({"output_dir": "o", "bogus_b": 1, "bogus_a": 2}, V5_FIELDS)


def test_shipped_config_adapts_cleanly_for_v5():
    cfg = load_config(CONFIG)
    fields = V5_FIELDS | (set(cfg["training"]) - {"warmup_ratio"})
    kwargs, _ = adapt_training_args(cfg["training"], fields)
    assert "warmup_ratio" not in kwargs and kwargs["warmup_steps"] == 0.03


def test_reserved_training_key():
    cfg = load_config(CONFIG)
    cfg["training"]["max_length"] = 1024
    with pytest.raises(ConfigError, match="max_length"):
        validate_config(cfg)
