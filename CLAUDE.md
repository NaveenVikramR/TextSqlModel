# CLAUDE.md

Production-style Text-to-SQL fine-tuning project (portfolio). See README.md for the user-facing overview.

## Plan

- **Dataset**: `b-mc2/sql-create-context` (fields: `question`, `context` = CREATE TABLE statements, `answer` = SQL).
  Clean: strip fields, drop empties, dedupe on (question, context), drop rows whose schema or SQL fails
  in an in-memory SQLite DB. Split with seed 42 into train / validation (500) / test (500).
  Push to the user's HF account as a **private** dataset.
- **Base models**: `unsloth/Llama-3.2-3B-Instruct` first, then `Llama-3.1-8B-Instruct`.
- **Metrics**: valid-SQL % (executes on the schema in SQLite) and normalized exact-match %.
- **Training**: QLoRA driven by a YAML config in `configs/` (Axolotl or TRL, not decided yet), logged to Weights & Biases.
- **Serving**: merge LoRA → convert to GGUF → run in Ollama → FastAPI endpoint in front.

## Hard constraints

- The local dev PC has **no usable GPU**. Data prep, eval, serving and tests must run on CPU.
- Training runs elsewhere (Colab or AWS EC2 g6.xlarge). Keep training code and deps
  (`requirements-train.txt`) separate; never import torch/transformers from the CPU modules.
- **Ask the user before installing anything large** (torch, CUDA, model weights, llama.cpp builds, etc.).
- Never commit data, checkpoints, GGUF files or W&B runs (see .gitignore). Data lives on the HF Hub.

## Code conventions

- Library code in `src/text2sql/`, thin CLIs in `scripts/` (they add `src` to `sys.path`).
- Core modules (`sql_utils`, `prompt`, `data_prep`, `evaluate`, `jsonl`) are **stdlib-only**;
  heavy imports (`datasets`, etc.) happen lazily inside scripts.
- `src/text2sql/prompt.py` is the single source of truth for the prompt format. Training,
  eval and serving must all use `build_messages` / `extract_sql`.
- `sql_utils.check_sql` builds a fresh DB per call and denies ATTACH/DETACH/PRAGMA. The serving
  layer gets user-supplied schemas and needs extra sandboxing (size limits, statement allow-list).

## Commands

```bash
pytest                                                    # unit tests (pythonpath configured in pyproject.toml)
python scripts/prepare_data.py --no-push --limit 2000     # quick data-prep dry run
python scripts/prepare_data.py --hub-repo <user>/<name>   # full run + private push
python scripts/evaluate.py --predictions preds.jsonl --report report.json
```

## Status

- [x] Repo scaffold, data-prep module + script, evaluation module + script, unit tests
- [ ] Run data prep and push dataset
- [ ] Baseline zero-shot eval of the base model
- [ ] QLoRA training config + Colab/EC2 notebook
- [ ] Merge → GGUF → Ollama, FastAPI service
