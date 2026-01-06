"""
Smoke tests for data ingestion.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from case_studies.hospital_anomalies.src.ingest import ingest_cihi_data, _create_mock_cihi_data


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
