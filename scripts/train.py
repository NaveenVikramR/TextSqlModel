"""QLoRA supervised fine-tuning with TRL SFTTrainer + PEFT. GPU only.

Usage (on the GPU box):
    pip install -r requirements-train.txt
    wandb login && hf auth login
    python scripts/train.py --config configs/qlora_llama32_3b.yaml
    python scripts/train.py --config configs/qlora_llama32_3b.yaml --max-train-samples 2000  # smoke test

Prompts are built with text2sql.prompt.build_prompt_completion, so loss is computed on
the SQL only and the prompt matches what scripts/generate.py and serving send.
The final LoRA adapter + tokenizer are saved to <output_dir>/final.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.env import load_env  # noqa: E402
from text2sql.config import ConfigError, adapt_training_args, apply_overrides, load_config  # noqa: E402
from text2sql.data_loading import load_split, validate_dataset_ref  # noqa: E402
from text2sql.prompt import build_prompt_completion  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--max-train-samples", type=int, help="Override data.max_train_samples")
    p.add_argument("--output-dir", help="Override training.output_dir")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.bf16=false (repeatable)",
    )
    return p.parse_args()


def to_prompt_completion(ds):
    return ds.map(
        lambda ex: build_prompt_completion(ex["question"], ex["context"], ex["answer"]),
        remove_columns=ds.column_names,
    )


def main() -> None:
    load_env()  # .env secrets (HF_TOKEN, WANDB_API_KEY); real env vars win
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    try:
        validate_dataset_ref(cfg["data"]["dataset"])  # fail before importing torch / loading weights
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        sys.exit("error: no CUDA GPU found. Training runs on Colab / EC2, not on the CPU dev machine.")

    set_seed(cfg.get("seed", 42))
    train_cfg = dict(cfg["training"])
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir
    if args.no_wandb:
        train_cfg["report_to"] = "none"
    elif "wandb" in cfg:
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])
        if cfg["wandb"].get("entity"):
            os.environ.setdefault("WANDB_ENTITY", cfg["wandb"]["entity"])

    # --- trainer config: built first so version mismatches fail before any download
    sft_kwargs = {
        **train_cfg,
        "run_name": cfg.get("run_name"),
        "seed": cfg.get("seed", 42),
        "max_length": cfg["model"]["max_length"],
        "completion_only_loss": True,
    }
    try:
        sft_kwargs, notes = adapt_training_args(sft_kwargs, {f.name for f in dataclasses.fields(SFTConfig)})
    except ConfigError as exc:
        sys.exit(f"error: {exc}")
    for note in notes:
        print(f"[config] {note}")
    sft_config = SFTConfig(**sft_kwargs)

    # --- data ---------------------------------------------------------------
    data = cfg["data"]
    train_limit = args.max_train_samples or data.get("max_train_samples")
    train_ds = to_prompt_completion(load_split(data["dataset"], data["train_split"], train_limit))
    eval_ds = to_prompt_completion(load_split(data["dataset"], data["eval_split"], data.get("max_eval_samples")))
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    # --- model --------------------------------------------------------------
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Llama 3 ships a dedicated pad token; fall back to EOS for other models.
        pad = "<|finetune_right_pad_id|>"
        tokenizer.pad_token = pad if pad in tokenizer.get_vocab() else tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        device_map={"": 0},
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    final_dir = Path(sft_config.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    import yaml

    # Save the effective config (after --set overrides) next to the adapter.
    (final_dir / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
