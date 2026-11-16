"""Generate SQL predictions for a dataset split. GPU only.

Loads the base model (optionally in 4-bit) plus an optional LoRA adapter, prompts it
with text2sql.prompt.build_messages (same format as training), decodes greedily, and
writes a predictions .jsonl. Score it on any machine (CPU is fine) with
scripts/evaluate.py.

Usage:
    # zero-shot baseline
    python scripts/generate.py --dataset <hf-user>/sql-create-context-clean --split test \
        --output outputs/preds_base.jsonl
    # fine-tuned
    python scripts/generate.py --dataset <hf-user>/sql-create-context-clean --split test \
        --adapter outputs/llama32-3b-qlora-r16/final --output outputs/preds_qlora.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.env import load_env  # noqa: E402
from text2sql.data_loading import load_split, validate_adapter_ref, validate_dataset_ref  # noqa: E402
from text2sql.jsonl import write_jsonl  # noqa: E402
from text2sql.prompt import build_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    p.add_argument("--adapter", help="Path or Hub id of a LoRA adapter (omit for the base model)")
    p.add_argument("--dataset", required=True, help="Hub repo id or local dir with <split>.jsonl")
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit NF4")
    return p.parse_args()


def main() -> None:
    load_env()  # .env secrets (HF_TOKEN, WANDB_API_KEY); real env vars win
    args = parse_args()
    # Validate cheap inputs before loading any model weights.
    try:
        validate_dataset_ref(args.dataset)
        if args.adapter:
            validate_adapter_ref(args.adapter)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"error: {exc}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        sys.exit("error: no CUDA GPU found. Run generation on Colab / EC2, then score locally.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer_src = args.adapter or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    tokenizer.padding_side = "left"  # decoder-only batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, quantization_config=quant, device_map={"": 0}
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    ds = load_split(args.dataset, args.split, args.limit)
    rows = ds.to_list()
    outputs = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(r["question"], r["context"]), add_generation_prompt=True, tokenize=False
            )
            for r in batch
        ]
        # The chat template already contains BOS.
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for r, text in zip(batch, texts):
            outputs.append({**r, "prediction": text.strip()})
        print(f"{min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    meta = {"base_model": args.base_model, "adapter": args.adapter, "split": args.split}
    write_jsonl(args.output, [{**o, "meta": meta} for o in outputs])
    print(f"wrote {len(outputs)} predictions to {args.output}")


if __name__ == "__main__":
    main()
