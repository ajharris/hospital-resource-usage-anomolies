"""
Smoke tests for data ingestion.
"""

import pytest
import pandas as pd
import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from case_studies.hospital_anomalies.src.ingest import ingest_cihi_data, _create_mock_cihi_data
from case_studies.hospital_anomalies.src.utils import get_data_path


def test_create_mock_cihi_admissions():
    """Test mock CIHI admissions data creation."""
    df = _create_mock_cihi_data('cihi_hospital_admissions')
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'date' in df.columns
    assert 'admissions' in df.columns
    assert 'region' in df.columns
    assert 'hospital_id' in df.columns


def test_create_mock_cihi_occupancy():
    """Test mock CIHI occupancy data creation."""
    df = _create_mock_cihi_data('cihi_bed_occupancy')
    
    assert isinstance(df, pd.DataFrame)
    assert 'occupancy_rate' in df.columns
    assert df['occupancy_rate'].min() >= 0
    assert df['occupancy_rate'].max() <= 100


def test_create_mock_cihi_icu():
    """Test mock CIHI ICU data creation."""
    df = _create_mock_cihi_data('cihi_icu_utilization')
    
    assert isinstance(df, pd.DataFrame)
    assert 'icu_beds_used' in df.columns
    assert 'icu_utilization_rate' in df.columns


def test_ingest_cihi_data():
    """Test data ingestion for multiple datasets."""
    dataset_ids = ['cihi_hospital_admissions', 'cihi_bed_occupancy']
    datasets = ingest_cihi_data(dataset_ids, force_download=True)
    
    assert len(datasets) == 2
    assert 'cihi_hospital_admissions' in datasets
    assert 'cihi_bed_occupancy' in datasets
    
    for dataset_id, df in datasets.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_ingest_creates_metadata():
    """Test that ingestion creates metadata sidecar files."""
    dataset_ids = ['cihi_hospital_admissions']
    datasets = ingest_cihi_data(dataset_ids, force_download=True)
    
    # Check that metadata file exists
    metadata_path = get_data_path("raw", "cihi", f"{dataset_ids[0]}_metadata.json")
    assert metadata_path.exists(), f"Metadata file not found: {metadata_path}"
    
    # Check metadata contents
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    assert 'source' in metadata
    assert 'dataset_id' in metadata
    assert 'retrieval_date' in metadata
    assert metadata['dataset_id'] == dataset_ids[0]
    assert 'CIHI' in metadata['source']


def test_ingest_parsed_dates():
    """Test that ingested data has properly parsed date columns."""
    dataset_ids = ['cihi_hospital_admissions']
    datasets = ingest_cihi_data(dataset_ids, force_download=True)
    
    # Check raw data
    raw_path = get_data_path("raw", "cihi", f"{dataset_ids[0]}.parquet")
    raw_df = pd.read_parquet(raw_path)
    assert 'date' in raw_df.columns
    assert pd.api.types.is_datetime64_any_dtype(raw_df['date'])
    
    # Check processed data
    processed_path = get_data_path("processed", "cihi", f"{dataset_ids[0]}.parquet")
    processed_df = pd.read_parquet(processed_path)
    assert 'date' in processed_df.columns
    assert pd.api.types.is_datetime64_any_dtype(processed_df['date'])
