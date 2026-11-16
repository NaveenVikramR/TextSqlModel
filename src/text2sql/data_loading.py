"""Load the cleaned splits either from the HF Hub or from a local data/processed dir.

`datasets` is imported lazily so the rest of the package stays stdlib-only.
"""

from __future__ import annotations

from pathlib import Path


def load_split(dataset: str, split: str, limit: int | None = None):
    """Return a `datasets.Dataset` for `split`.

    `dataset` is either a local directory containing `<split>.jsonl` files
    (output of scripts/prepare_data.py) or a Hub repo id like `user/name`.
    """
    from datasets import load_dataset

    path = Path(dataset)
    if path.is_dir():
        ds = load_dataset("json", data_files={split: str(path / f"{split}.jsonl")}, split=split)
    else:
        ds = load_dataset(dataset, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
