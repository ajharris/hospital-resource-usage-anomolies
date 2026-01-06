"""
I/O utilities for loading and saving data artifacts.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from publicdata_ca.utils.logging import get_logger

logger = get_logger(__name__)


def load_parquet(path: Path) -> pd.DataFrame:
    """
    Load a parquet file.
    
    Args:
        path: Path to parquet file
    
    Returns:
        DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading parquet from {path}")
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path):
    """
    Save a DataFrame to parquet format.
    
    Args:
        df: DataFrame to save
        path: Path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(path, index=False)
    logger.info(f"Saved parquet to {path} ({len(df)} rows, {len(df.columns)} columns)")


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file.
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading CSV from {path}")
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, **kwargs):
    """
    Save a DataFrame to CSV format.
    
    Args:
        df: DataFrame to save
        path: Path to save to
        **kwargs: Additional arguments for df.to_csv
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False, **kwargs)
    logger.info(f"Saved CSV to {path} ({len(df)} rows, {len(df.columns)} columns)")


def ensure_output_dirs(config: dict):
    """
    Ensure all output directories exist.
    
    Args:
        config: Configuration dictionary with paths
    """
    from publicdata_ca.acquisition.storage import get_project_root
    
    root = get_project_root()
    
    paths = config.get('paths', {})
    for key, rel_path in paths.items():
        full_path = root / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {full_path}")


def get_output_path(config: dict, path_key: str, filename: str) -> Path:
    """
    Get full output path from config and filename.
    
    Args:
        config: Configuration dictionary
        path_key: Key in config['paths']
        filename: Filename to append
    
    Returns:
        Full path
    """
    from publicdata_ca.acquisition.storage import get_project_root
    
    root = get_project_root()
    rel_path = config.get('paths', {}).get(path_key, f'data/{path_key}')
    
    return root / rel_path / filename


def save_results_summary(
    anomalies_df: pd.DataFrame,
    config: dict,
    filename: str = 'anomalies.csv'
):
    """
    Save anomaly results summary to CSV.
    
    Args:
        anomalies_df: DataFrame with anomaly results
        config: Configuration dictionary
        filename: Output filename
    """
    output_path = get_output_path(config, 'results', filename)
    save_csv(anomalies_df, output_path)


def save_features(
    features_df: pd.DataFrame,
    config: dict,
    filename: str = 'features.parquet'
):
    """
    Save engineered features to parquet.
    
    Args:
        features_df: DataFrame with features
        config: Configuration dictionary
        filename: Output filename
    """
    output_path = get_output_path(config, 'processed_data', filename)
    save_parquet(features_df, output_path)
