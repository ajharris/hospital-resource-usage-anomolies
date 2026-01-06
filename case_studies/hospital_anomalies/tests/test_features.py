"""
Tests for feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from case_studies.hospital_anomalies.src.features import (
    create_rolling_features,
    create_lag_features,
    create_seasonal_features,
    create_difference_features,
    engineer_features
)


@pytest.fixture
def sample_timeseries():
    """Create sample time series data."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(len(dates)) * 10 + 50
    })
    return df


def test_create_rolling_features(sample_timeseries):
    """Test rolling feature creation."""
    df = create_rolling_features(sample_timeseries, ['value'], windows=[7, 14])
    
    assert 'value_rolling_mean_7' in df.columns
    assert 'value_rolling_std_7' in df.columns
    assert 'value_rolling_mean_14' in df.columns
    assert 'value_rolling_std_14' in df.columns
    
    # Check that rolling mean is calculated correctly
    assert df['value_rolling_mean_7'].notna().sum() > 0


def test_create_lag_features(sample_timeseries):
    """Test lag feature creation."""
    df = create_lag_features(sample_timeseries, ['value'], lags=[1, 7])
    
    assert 'value_lag_1' in df.columns
    assert 'value_lag_7' in df.columns
    
    # Check lag is correct (first lag should be NaN)
    assert pd.isna(df['value_lag_1'].iloc[0])
    assert df['value_lag_1'].iloc[1] == df['value'].iloc[0]


def test_create_seasonal_features(sample_timeseries):
    """Test seasonal feature creation."""
    df = create_seasonal_features(sample_timeseries, 'date')
    
    assert 'month' in df.columns
    assert 'day_of_week' in df.columns
    assert 'quarter' in df.columns
    assert 'month_sin' in df.columns
    assert 'month_cos' in df.columns
    
    # Check month values are valid
    assert df['month'].min() >= 1
    assert df['month'].max() <= 12


def test_create_difference_features(sample_timeseries):
    """Test difference feature creation."""
    df = create_difference_features(sample_timeseries, ['value'], periods=[1])
    
    assert 'value_diff_1' in df.columns
    assert 'value_pct_change_1' in df.columns
    
    # First difference should be NaN
    assert pd.isna(df['value_diff_1'].iloc[0])


def test_engineer_features_complete(sample_timeseries):
    """Test complete feature engineering pipeline."""
    config = {
        'features': {
            'seasonal_features': True,
            'rolling_windows': [7],
            'lag_features': [1, 7]
        }
    }
    
    df = engineer_features(sample_timeseries, config, value_cols=['value'])
    
    # Check that multiple feature types are created
    assert len(df.columns) > len(sample_timeseries.columns)
    
    # Check specific features exist
    assert 'value_rolling_mean_7' in df.columns
    assert 'value_lag_1' in df.columns
    assert 'month' in df.columns
