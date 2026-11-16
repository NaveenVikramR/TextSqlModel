"""Load the cleaned splits either from the HF Hub or from a local data/processed dir.

`datasets` is imported lazily so the rest of the package stays stdlib-only.
"""

from __future__ import annotations

from pathlib import Path

_PLACEHOLDER_HINTS = ("<", ">", "your-hf-username", "your-")


def validate_dataset_ref(dataset: str | None) -> str:
    """Fail fast on unset or placeholder dataset ids (e.g. from an unedited config)."""
    if not dataset or any(h in dataset for h in _PLACEHOLDER_HINTS):
        raise ValueError(
            f"dataset is not set (got {dataset!r}). Pass a Hub id like "
            "'<your-hf-user>/sql-create-context-clean' or a local dir such as data/processed "
            "(train.py: --set data.dataset=...; generate.py: --dataset ...)."
        )
    return dataset


def validate_adapter_ref(adapter: str) -> str:
    """An adapter is either an existing local directory or a Hub id `namespace/name`.

    A path that doesn't exist locally but looks like one (e.g. outputs/run/final) would
    otherwise be sent to the Hub and fail with a confusing 'Repo id must be...' error.
    """
    if Path(adapter).is_dir():
        return adapter
    looks_local = adapter.startswith((".", "/", "~", "outputs")) or "\\" in adapter or adapter.count("/") != 1
    if looks_local:
        raise FileNotFoundError(
            f"adapter directory not found: {adapter!r}. Did training finish and save "
            "<output_dir>/final? Check the train.py output above."
        )
    return adapter


def load_split(dataset: str, split: str, limit: int | None = None):
    """Return a `datasets.Dataset` for `split`.

    `dataset` is either a local directory containing `<split>.jsonl` files
    (output of scripts/prepare_data.py) or a Hub repo id like `user/name`.
    """
    validate_dataset_ref(dataset)
    from datasets import load_dataset

    path = Path(dataset)
    if path.is_dir():
        ds = load_dataset("json", data_files={split: str(path / f"{split}.jsonl")}, split=split)
    else:
        ds = load_dataset(dataset, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
