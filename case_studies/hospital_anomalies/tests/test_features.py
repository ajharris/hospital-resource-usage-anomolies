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
    engineer_features,
    build_features,
    handle_missing_values
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


def test_handle_missing_values(sample_timeseries):
    """Test missing value handling."""
    # Create a copy with some NaN values
    df = sample_timeseries.copy()
    df.loc[10:15, 'value'] = np.nan
    
    # Test ffill method
    df_filled = handle_missing_values(df, method='ffill')
    assert df_filled['value'].isna().sum() == 0
    
    # Test zero method
    df_zero = handle_missing_values(df, method='zero', fill_value=0)
    assert df_zero['value'].isna().sum() == 0
    assert df_zero.loc[10, 'value'] == 0


def test_build_features(sample_timeseries):
    """Test build_features function with NaN handling."""
    # Test with default config
    df_features = build_features(sample_timeseries)
    
    # Check that there are no NaN values in the output
    assert df_features.isna().sum().sum() == 0
    
    # Check that features were created
    assert len(df_features.columns) > len(sample_timeseries.columns)
    
    # Check that it has rolling and lag features (from default config)
    rolling_cols = [col for col in df_features.columns if 'rolling' in col]
    lag_cols = [col for col in df_features.columns if 'lag' in col]
    assert len(rolling_cols) > 0
    assert len(lag_cols) > 0


def test_build_features_custom_config(sample_timeseries):
    """Test build_features with custom configuration."""
    config = {
        'features': {
            'seasonal_features': True,
            'rolling_windows': [3, 6, 12],
            'lag_features': [1, 2, 3]
        }
    }
    
    df_features = build_features(
        sample_timeseries,
        config=config,
        nan_method='zero'
    )
    
    # Verify no NaNs
    assert df_features.isna().sum().sum() == 0
    
    # Verify specific rolling features exist
    assert 'value_rolling_mean_3' in df_features.columns
    assert 'value_rolling_mean_6' in df_features.columns
    assert 'value_rolling_mean_12' in df_features.columns
    
    # Verify specific lag features exist
    assert 'value_lag_1' in df_features.columns
    assert 'value_lag_2' in df_features.columns
    assert 'value_lag_3' in df_features.columns
