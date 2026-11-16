# Text-to-SQL Fine-Tuning

Fine-tune a small open LLM to translate natural-language questions into SQLite queries,
given a `CREATE TABLE` schema. The project covers the full path: data cleaning, QLoRA
training, evaluation, and serving via GGUF + Ollama behind a FastAPI endpoint.

## Pipeline

| Stage | Where it runs | What it does |
|---|---|---|
| Data prep | CPU | Clean [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context), split, push to a private HF dataset |
| Training | GPU (Colab / EC2 g6.xlarge) | QLoRA on `unsloth/Llama-3.2-3B-Instruct` (later Llama-3.1-8B-Instruct), YAML config, logged to W&B |
| Evaluation | CPU | Valid-SQL % and normalized exact-match % on the 500-row test split |
| Serving | CPU | Merged model → GGUF → Ollama, behind FastAPI |

## Data cleaning

1. Strip whitespace from `question`, `context`, `answer`; drop rows where any are empty.
2. Deduplicate on `(question, context)`, keeping the first occurrence.
3. Load each `context` into an in-memory SQLite DB; drop rows whose schema fails to load
   or whose `answer` SQL fails to execute.
4. Shuffle with seed 42 → test (500), validation (500), train (rest).

## Metrics

- **Valid-SQL %**: the predicted query runs as a single statement against the schema in
  SQLite (tables are empty, so this checks syntax and schema binding, not results).
- **Exact-match %**: predicted and gold SQL are equal after normalization (lowercase outside
  string literals, unified quote style, whitespace and trailing `;` removed, `<>` → `!=`).

## Layout

```
src/text2sql/      library code (stdlib-only core: sql_utils, prompt, data_prep, evaluate)
scripts/           CLI entry points (prepare_data.py, evaluate.py)
configs/           training YAML configs
notebooks/         exploration and Colab training notebooks
tests/             pytest unit tests
```

## Quickstart (CPU)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest

# dry run on 2k rows without pushing
python scripts/prepare_data.py --no-push --limit 2000

# full run + push (needs `hf auth login` or HF_TOKEN)
python scripts/prepare_data.py --hub-repo <hf-user>/sql-create-context-clean

# score predictions (jsonl with context / answer / prediction)
python scripts/evaluate.py --predictions outputs/preds.jsonl --report outputs/report.json
```

## Results

_To be filled in after the first training run._

| Model | Valid-SQL % | Exact-match % |
|---|---|---|
| Llama-3.2-3B-Instruct (base, zero-shot) | – | – |
| Llama-3.2-3B-Instruct + QLoRA | – | – |
