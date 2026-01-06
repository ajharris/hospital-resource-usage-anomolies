"""
Acquisition module for fetching and managing public data sources.
"""

from .registry import DatasetRegistry
from .fetch import fetch_dataset
from .storage import get_data_path, ensure_data_dirs
from .validate import validate_dataset

__all__ = [
    "DatasetRegistry",
    "fetch_dataset",
    "get_data_path",
    "ensure_data_dirs",
    "validate_dataset",
]
