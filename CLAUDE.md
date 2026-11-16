# CLAUDE.md

Production-style Text-to-SQL fine-tuning project (portfolio). See README.md for the user-facing overview.

## Plan

- **Dataset**: `b-mc2/sql-create-context` (fields: `question`, `context` = CREATE TABLE statements, `answer` = SQL).
  Clean: strip fields, drop empties, dedupe on (question, context), drop rows whose schema or SQL fails
  in an in-memory SQLite DB. Add `num_tables`. Split with seed 42, stratified by table count:
  test = 500 single + 100 multi, validation = 500 single + 100 multi, train = rest.
  Push to the user's HF account as a **private** dataset.
- **Base models**: `unsloth/Llama-3.2-3B-Instruct` first, then `Llama-3.1-8B-Instruct`.
- **Metrics**: valid-SQL % (a **read-only** query, SELECT / WITH ... SELECT, that executes on the schema in
  SQLite) and normalized exact-match %. Normalization is final; don't change it without asking.
- **Training**: TRL `SFTTrainer` + PEFT QLoRA, driven by `configs/qlora_llama32_3b.yaml` via `scripts/train.py`,
  logged to Weights & Biases. Conversational prompt-completion format → loss on SQL only.
- **Generation vs scoring**: `scripts/generate.py` (GPU) writes `predictions.jsonl`; `scripts/evaluate.py`
  (CPU) scores it.
- **Serving**: merge LoRA → convert to GGUF → run in Ollama → FastAPI endpoint in front.

## Hard constraints

- The local dev PC has **no usable GPU**. Data prep, eval, serving and tests must run on CPU.
- Training runs elsewhere (Colab or AWS EC2 g6.xlarge). Keep training code and deps
  (`requirements-train.txt`) separate; never import torch/transformers from the CPU modules.
- **Ask the user before installing anything large** (torch, CUDA, model weights, llama.cpp builds, etc.).
- Never commit data, checkpoints, GGUF files or W&B runs (see .gitignore). Data lives on the HF Hub.
- Secrets live in a gitignored `.env` (names in `.env.example`: `HF_TOKEN`, `WANDB_API_KEY`, `GH_COLAB_TOKEN`).
  Scripts call `text2sql.env.load_env()` (override=False). **Never print or read secret values**; check
  presence only.
- GitHub repo: **NaveenVikramR/TextSqlModel** (public; https://github.com/NaveenVikramR/TextSqlModel).
  The local folder is named `textToSqlModel`, but the repo name is `TextSqlModel`; use the repo name in
  docs, notebook and commands.
- Colab notebook reads the same names from Colab Secrets via `get_secret()`, falling back to `getpass`.

## Code conventions

- Library code in `src/text2sql/`, thin CLIs in `scripts/` (they add `src` to `sys.path`).
- Core modules (`sql_utils`, `prompt`, `data_prep`, `evaluate`, `jsonl`) are **stdlib-only**;
  `config` needs PyYAML, `data_loading` imports `datasets` lazily. torch/transformers/trl/peft are
  imported only inside `scripts/train.py` and `scripts/generate.py`.
- `src/text2sql/prompt.py` is the single source of truth for the prompt format. Training
  (`build_prompt_completion`), generation/serving (`build_messages`) and eval (`extract_sql`) all use it.
  When building the Ollama Modelfile, its TEMPLATE must reproduce the Llama 3 chat template + this system prompt.
- `sql_utils.check_sql(read_only=True)` (default) = prefix check (SELECT/WITH) + SQLite authorizer that only
  allows read actions; it builds a fresh DB per call and always denies ATTACH/DETACH/PRAGMA. Serving should
  reuse it, and still add input size limits for user-supplied schemas.

## Commands

```bash
pytest                                                    # unit tests (pythonpath configured in pyproject.toml)
python scripts/prepare_data.py --no-push --limit 2000     # quick data-prep dry run
python scripts/prepare_data.py --hub-repo <user>/<name>   # full run + private push
python scripts/evaluate.py --predictions preds.jsonl --report report.json
# GPU box only:
python scripts/train.py --config configs/qlora_llama32_3b.yaml [--max-train-samples 2000]
python scripts/generate.py --dataset <repo-or-dir> --split test --adapter <dir> --output preds.jsonl
```

Local venv: `.venv/` (Windows: `.venv\Scripts\python -m pytest`). Only `requirements.txt` is installed there.

## Status

- [x] Repo scaffold, data-prep module + script, evaluation module + script, unit tests
- [x] Read-only valid-SQL check; TRL train script + YAML; GPU generate script (written, not yet run)
- [x] 2,000-row data-prep dry run (`--no-push`)
- [x] Full data prep run (`--no-push`): 76,909 / 78,577 kept; drop reasons documented in README "Data quality"
- [x] `num_tables` per row; eval reports single- vs multi-table
- [x] `notebooks/colab_runner.ipynb` (clone → install → login → baseline → smoke train → adapter preds → download)
- [ ] Push private dataset to the Hub (needed before the Colab notebook can run)
- [ ] Baseline zero-shot eval of the base model
- [ ] First QLoRA run on Colab/EC2
- [ ] Merge → GGUF → Ollama, FastAPI service

## Data facts

- Duplicate-table schemas and other flawed rows are **dropped, never repaired** (user decision).
- Multi-table schemas are only ~2.2% of the data (1,660 rows), so splits are stratified:
  test/validation = 500 single + 100 multi each, train = 74,249 single + 1,460 multi.
  Eval splits over-represent multi-table; always report the two groups separately.

## TODO

- **Execution accuracy**: populate each schema with random rows (type-aware generation, several seeds),
  run gold and predicted SQL, and compare result sets (order-insensitive unless the gold has ORDER BY).
  Catches semantically equivalent queries that exact-match misses.
