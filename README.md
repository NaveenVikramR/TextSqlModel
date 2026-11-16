# Text-to-SQL Fine-Tuning

Fine-tune a small open LLM to translate natural-language questions into SQLite queries,
given a `CREATE TABLE` schema. The project covers the full path: data cleaning, QLoRA
training, evaluation, and serving via GGUF + Ollama behind a FastAPI endpoint.

## Notable findings

- **The source dataset corrupts numbers with thousands separators.** Executing every gold query
  in SQLite showed that 564 queries contain a stray `OFFSET` clause. A filter like
  `attendance < 46,735` was split on the comma during conversion and became
  `attendance < 46 OFFSET 735`. Those queries are syntax errors in SQLite. In engines that accept
  `OFFSET` without `LIMIT`, they would quietly mean something else. These rows are dropped, along
  with about 1,100 other rows whose schema or SQL doesn't execute (see [Data quality](#data-quality)).
- **Multi-table questions are rare (about 2%)**, so a plain random 500-row test split would hold
  only about 7 of them. The splits are stratified to reserve 100 multi-table examples each for test
  and validation.

## Pipeline

| Stage | Where it runs | What it does |
|---|---|---|
| Data prep | CPU | Clean [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context), split, push to a private HF dataset |
| Training | GPU (Colab / EC2 g6.xlarge) | QLoRA on `unsloth/Llama-3.2-3B-Instruct` (later Llama-3.1-8B-Instruct), YAML config, logged to W&B |
| Evaluation | CPU | Valid-SQL % and normalized exact-match % on the 600-row test split, overall and single- vs multi-table |
| Serving | CPU | Merged model → GGUF → Ollama, behind FastAPI |

## Data cleaning

1. Strip whitespace from `question`, `context`, `answer`; drop rows where any are empty.
2. Deduplicate on `(question, context)`, keeping the first occurrence.
3. Load each `context` into an in-memory SQLite DB; drop rows whose schema fails to load
   or whose `answer` SQL fails to execute.
4. Add `num_tables` (number of `CREATE TABLE` statements) to each row.
5. Split with seed 42, stratified by table count: test = 500 single-table + 100 multi-table,
   validation = 500 single-table + 100 multi-table, train = the rest.

## Data quality

The full cleaning run (`scripts/prepare_data.py`, numbers from `data/processed/stats.json`) keeps
**76,909 of 78,577 rows (97.9%)**. Flawed rows are dropped, not repaired, so every kept example is
guaranteed to execute against its own schema.

| Step | Rows dropped |
|---|---:|
| Empty field after strip | 0 |
| Duplicate (question, schema) | 1 |
| Schema fails to load in SQLite | 294 |
| Gold SQL fails on its schema | 1,373 |

**Schema flaws (294)**

| Error | Count | Cause / example |
|---|---:|---|
| `table already exists` | 137 | Same table defined twice with different column subsets: `CREATE TABLE station (name, id); CREATE TABLE station (name, id, installation_date)` |
| syntax error | 131 | Unquoted reserved words as column names: `group` (54), `order` (34), `table` (20), `index` (14), `returning` (6), e.g. `CREATE TABLE t (order VARCHAR, ...)` |
| `duplicate column name` | 26 | Same column in two casings, which SQLite treats as identical: `editor (editor_id, Name, Editor_ID)` |

**Gold SQL flaws (1,373)**

| Error | Count | Cause / example |
|---|---:|---|
| syntax error: stray `OFFSET` | 564 | Thousands separators split into an `OFFSET` clause: `attendance < 46,735` became `attendance < 46 OFFSET 735` |
| syntax error: unquoted odd tokens | 237 | Column aliases / literals like `@rpm`, `$__` left unquoted |
| syntax error: bad `AS` | 199 | Aliases inside function calls: `COUNT(1 AS st_prize__) AS $__` |
| syntax error: brackets | 85 | Footnote text in column names: `existential_question_[_not_in_citation_given_]` |
| syntax error: other | 5 | e.g. reserved word `cast` used as a column |
| `no such column` | 185 | Gold query uses a column missing from the schema: `WHERE catalog_level_number > 5` |
| `no such function` | 90 | Column headers parsed as functions: `rating / SHARE(18 - 49)` (from "Rating/Share (18-49)") |
| `ambiguous column name` | 4 | Unqualified column present in two joined tables |
| other | 4 | `RANK(_number)` (2), `ORDER BY` before `INTERSECT` (1), `no such table` (1) |

The duplicate-table/column flaws come from the Spider-derived rows (multi-table, named schemas); the
reserved-word and syntax flaws almost all come from the WikiSQL-derived rows (`table_<id>` names).

**Split composition.** Multi-table schemas are rare (1,660 rows, 2.2% of the cleaned data). A plain
random split put only 7 of them in a 500-row test set, so the split is stratified by table count
and reserves 100 multi-table rows each for test and validation. This makes the eval splits
over-represent multi-table questions (16.7% vs 1.9% in train), so compare the **single-table** and
**multi-table** metrics separately rather than reading the overall number as representative of
the training distribution.

| Split | Rows | Single-table | Multi-table |
|---|---:|---:|---:|
| train | 75,709 | 74,249 | 1,460 |
| validation | 600 | 500 | 100 |
| test | 600 | 500 | 100 |

## Metrics

- **Valid-SQL %**: the predicted query is a single read-only statement (SELECT / WITH ... SELECT)
  that runs against the schema in SQLite. Writes are rejected by a prefix check and an SQLite
  authorizer. Tables are empty, so this checks syntax and schema binding, not results.
- **Exact-match %**: predicted and gold SQL are equal after normalization (lowercase outside
  string literals, unified quote style, whitespace and trailing `;` removed, `<>` → `!=`).
- Both metrics are reported overall and split into **single-table** vs **multi-table** schemas
  (`num_tables` = number of `CREATE TABLE` statements, added to every row during prep).

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
cp .env.example .env      # fill in HF_TOKEN / WANDB_API_KEY; scripts load it automatically
pytest

# dry run on 2k rows without pushing
python scripts/prepare_data.py --no-push --limit 2000

# full run + push (uses HF_TOKEN from .env or the environment)
python scripts/prepare_data.py --hub-repo <hf-user>/sql-create-context-clean

# score predictions (jsonl with context / answer / prediction)
python scripts/evaluate.py --predictions outputs/preds.jsonl --report outputs/report.json
```

## Training and generation (GPU box)

```bash
pip install -r requirements-train.txt     # after installing the CUDA build of torch
wandb login && hf auth login
python scripts/train.py --config configs/qlora_llama32_3b.yaml
python scripts/generate.py --dataset <hf-user>/sql-create-context-clean --split test \
    --adapter outputs/llama32-3b-qlora-r16/final --output outputs/preds_qlora.jsonl
```

Copy the predictions file back and score it on CPU with `scripts/evaluate.py`.

## Results

_To be filled in after the first training run._

| Model | Valid-SQL % | Exact-match % |
|---|---|---|
| Llama-3.2-3B-Instruct (base, zero-shot) | – | – |
| Llama-3.2-3B-Instruct + QLoRA | – | – |
