"""
Data ingestion module for CIHI hospital data.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from .utils import get_logger, get_data_path, ensure_data_dirs

logger = get_logger(__name__)


def ingest_cihi_data(
    dataset_ids: List[str],
    force_download: bool = False
) -> dict:
    """
    Ingest CIHI hospital datasets using the acquisition package.
    
    Args:
        dataset_ids: List of dataset IDs to fetch
        force_download: If True, re-download even if cached
    
    Returns:
        Dictionary mapping dataset_id to DataFrame
    """
    ensure_data_dirs("cihi")
    datasets = {}
    
    for dataset_id in dataset_ids:
        logger.info(f"Fetching dataset: {dataset_id}")
        try:
            # For MVP, we'll create mock data since actual CIHI URLs need authentication
            # In production, this would use fetch_dataset()
            df = _create_mock_cihi_data(dataset_id)
            datasets[dataset_id] = df
            
            # Save raw data
            raw_path = get_data_path("raw", "cihi", f"{dataset_id}.parquet")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(raw_path, index=False)
            logger.info(f"Saved raw data to {raw_path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch {dataset_id}: {e}")
            raise
    
    return datasets


def _create_mock_cihi_data(dataset_id: str) -> pd.DataFrame:
    """
    Create mock CIHI data for demonstration purposes.
    
    In production, this would be replaced by actual data fetching.
    """
    import numpy as np
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range(start="2019-01-01", end="2023-12-31", freq="D")
    n_records = len(dates)
    
    # Base structure
    df = pd.DataFrame({
        'date': dates,
        'region': np.random.choice(['Ontario', 'Quebec', 'BC', 'Alberta'], n_records),
        'hospital_id': np.random.choice([f'H{i:03d}' for i in range(1, 21)], n_records),
    })
    
    # Add dataset-specific metrics
    if 'admissions' in dataset_id:
        # Hospital admissions with seasonal pattern and anomalies
        base = 50 + 20 * np.sin(2 * np.pi * np.arange(n_records) / 365)
        noise = np.random.normal(0, 5, n_records)
        # Add some anomalies
        anomalies = np.zeros(n_records)
        anomaly_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
        anomalies[anomaly_indices] = np.random.normal(30, 10, len(anomaly_indices))
        df['admissions'] = np.maximum(0, base + noise + anomalies).astype(int)
        
    elif 'occupancy' in dataset_id:
        # Bed occupancy rates (percentage)
        base = 75 + 10 * np.sin(2 * np.pi * np.arange(n_records) / 365)
        noise = np.random.normal(0, 3, n_records)
        anomalies = np.zeros(n_records)
        anomaly_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
        anomalies[anomaly_indices] = np.random.normal(15, 5, len(anomaly_indices))
        df['occupancy_rate'] = np.clip(base + noise + anomalies, 0, 100)
        
    elif 'icu' in dataset_id:
        # ICU utilization
        base = 15 + 5 * np.sin(2 * np.pi * np.arange(n_records) / 365)
        noise = np.random.normal(0, 2, n_records)
        anomalies = np.zeros(n_records)
        anomaly_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
        anomalies[anomaly_indices] = np.random.normal(10, 3, len(anomaly_indices))
        df['icu_beds_used'] = np.maximum(0, base + noise + anomalies).astype(int)
        df['icu_beds_total'] = 25
        df['icu_utilization_rate'] = (df['icu_beds_used'] / df['icu_beds_total'] * 100).clip(0, 100)
    
    return df


def load_ingested_data(dataset_ids: List[str]) -> dict:
    """
    Load previously ingested CIHI data from storage.
    
    Args:
        dataset_ids: List of dataset IDs to load
    
    Returns:
        Dictionary mapping dataset_id to DataFrame
    """
    datasets = {}
    
    for dataset_id in dataset_ids:
        raw_path = get_data_path("raw", "cihi", f"{dataset_id}.parquet")
        if not raw_path.exists():
            raise FileNotFoundError(f"Data file not found: {raw_path}")
        
        logger.info(f"Loading dataset from {raw_path}")
        datasets[dataset_id] = pd.read_parquet(raw_path)
    
    return datasets
