"""src.data
Data loading and preprocessing utilities."""

from pathlib import Path

__all__ = ["load_dataset", "Dataset"]


def load_dataset(path: Path):
    raise NotImplementedError


class Dataset:
    pass
