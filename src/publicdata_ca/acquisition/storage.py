"""
Path conventions and storage management for raw and processed data.
"""

from pathlib import Path
from typing import Optional
import os


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assumes this file is in src/publicdata_ca/acquisition/
    return Path(__file__).parent.parent.parent.parent


def get_data_path(
    stage: str,
    source: str,
    filename: Optional[str] = None
) -> Path:
    """
    Get path for data files following the project convention.
    
    Args:
        stage: One of 'raw', 'interim', or 'processed'
        source: Data source name (e.g., 'cihi')
        filename: Optional filename to append
    
    Returns:
        Path to the data location
    """
    if stage not in ['raw', 'interim', 'processed']:
        raise ValueError(f"Invalid stage: {stage}. Must be 'raw', 'interim', or 'processed'")
    
    root = get_project_root()
    path = root / "data" / stage / source
    
    if filename:
        path = path / filename
    
    return path


def ensure_data_dirs(source: str = "cihi"):
    """
    Ensure data directories exist for a given source.
    
    Args:
        source: Data source name (e.g., 'cihi')
    """
    for stage in ['raw', 'interim', 'processed']:
        path = get_data_path(stage, source)
        path.mkdir(parents=True, exist_ok=True)


def get_versioned_path(
    stage: str,
    source: str,
    filename: str,
    version: str
) -> Path:
    """
    Get versioned path for data files.
    
    Args:
        stage: One of 'raw', 'interim', or 'processed'
        source: Data source name
        filename: Base filename
        version: Version string (e.g., '20240101', 'v1')
    
    Returns:
        Path with version directory
    """
    base_path = get_data_path(stage, source)
    return base_path / version / filename


def list_versions(stage: str, source: str) -> list:
    """
    List available versions for a data source.
    
    Args:
        stage: One of 'raw', 'interim', or 'processed'
        source: Data source name
    
    Returns:
        List of version strings
    """
    base_path = get_data_path(stage, source)
    if not base_path.exists():
        return []
    
    return [d.name for d in base_path.iterdir() if d.is_dir()]
