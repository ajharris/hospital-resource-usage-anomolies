"""
Feature engineering for time-series anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .utils import get_logger

logger = get_logger(__name__)


def create_rolling_features(
    df: pd.DataFrame,
    value_cols: List[str],
    windows: List[int],
    min_periods: Optional[int] = 1
) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Args:
        df: DataFrame with time-series data (must be sorted by date)
        value_cols: Columns to compute rolling features for
        windows: List of window sizes (in rows)
        min_periods: Minimum observations in window
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for col in value_cols:
        if col not in df.columns:
            continue
        
        for window in windows:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).mean()
            
            # Rolling std
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).std()
            
            # Rolling min/max
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).min()
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).max()
    
    logger.info(f"Created rolling features for windows: {windows}")
    return df


def create_lag_features(
    df: pd.DataFrame,
    value_cols: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lagged features for time-series data.
    
    Args:
        df: DataFrame with time-series data (must be sorted by date)
        value_cols: Columns to create lags for
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for col in value_cols:
        if col not in df.columns:
            continue
        
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    logger.info(f"Created lag features for lags: {lags}")
    return df


def create_seasonal_features(
    df: pd.DataFrame,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Create seasonal/temporal features from date column.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
    
    Returns:
        DataFrame with seasonal features added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract temporal features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['quarter'] = df[date_col].dt.quarter
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # Cyclical encoding for month and day_of_week
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    logger.info("Created seasonal features")
    return df


def create_difference_features(
    df: pd.DataFrame,
    value_cols: List[str],
    periods: List[int] = [1]
) -> pd.DataFrame:
    """
    Create difference features (current - previous).
    
    Args:
        df: DataFrame with time-series data
        value_cols: Columns to compute differences for
        periods: Periods for differencing
    
    Returns:
        DataFrame with difference features added
    """
    df = df.copy()
    
    for col in value_cols:
        if col not in df.columns:
            continue
        
        for period in periods:
            df[f'{col}_diff_{period}'] = df[col].diff(period)
            df[f'{col}_pct_change_{period}'] = df[col].pct_change(period)
    
    logger.info(f"Created difference features for periods: {periods}")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'ffill',
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Handle missing values in feature matrix deterministically.
    
    Args:
        df: DataFrame with potential missing values
        method: Method for handling NaNs ('ffill', 'bfill', 'zero', 'mean', 'drop')
        fill_value: Value to use for 'zero' method
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if method == 'ffill':
        # Forward fill, then fill remaining NaNs with zeros
        df = df.ffill().fillna(fill_value)
    elif method == 'bfill':
        # Backward fill, then fill remaining NaNs with zeros
        df = df.bfill().fillna(fill_value)
    elif method == 'zero':
        # Fill with specified value (default 0)
        df = df.fillna(fill_value)
    elif method == 'mean':
        # Fill with column means
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'drop':
        # Drop rows with any NaN values
        df = df.dropna()
    else:
        logger.warning(f"Unknown method '{method}', using 'ffill'")
        df = df.ffill().fillna(fill_value)
    
    logger.info(f"Handled missing values using method: {method}")
    logger.info(f"Remaining NaNs: {df.isna().sum().sum()}")
    
    return df


def build_features(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    value_cols: Optional[List[str]] = None,
    handle_nans: bool = True,
    nan_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Build feature matrix from raw data with deterministic NaN handling.
    
    This is the main entry point for feature engineering. It applies all
    feature transformations and ensures the output has no NaN values.
    
    Args:
        df: DataFrame with raw data (must include 'date' column)
        config: Configuration dictionary (if None, uses sensible defaults)
        value_cols: Columns to create features for (auto-detect if None)
        handle_nans: Whether to handle NaN values
        nan_method: Method for handling NaNs ('ffill', 'bfill', 'zero', 'mean', 'drop')
    
    Returns:
        DataFrame with engineered features and no NaN values
    """
    if config is None:
        # Use sensible defaults
        config = {
            'features': {
                'seasonal_features': True,
                'rolling_windows': [3, 7],
                'lag_features': [1, 7]
            }
        }
    
    # Apply feature engineering
    df_features = engineer_features(df, config, value_cols)
    
    # Handle missing values
    if handle_nans:
        df_features = handle_missing_values(
            df_features,
            method=nan_method,
            fill_value=0.0
        )
    
    return df_features


def engineer_features(
    df: pd.DataFrame,
    config: dict,
    value_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply all feature engineering steps based on configuration.
    
    Args:
        df: DataFrame with raw data
        config: Configuration dictionary
        value_cols: Columns to create features for (auto-detect if None)
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Ensure sorted by date
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    # Auto-detect numeric columns if not specified
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID columns
        value_cols = [c for c in value_cols if not any(
            x in c.lower() for x in ['id', 'year', 'month', 'day', 'week', 'quarter']
        )]
    
    logger.info(f"Engineering features for columns: {value_cols}")
    
    # Seasonal features
    if config.get('features', {}).get('seasonal_features', False) and 'date' in df.columns:
        df = create_seasonal_features(df, 'date')
    
    # Rolling features
    rolling_windows = config.get('features', {}).get('rolling_windows', [])
    if rolling_windows:
        df = create_rolling_features(df, value_cols, rolling_windows)
    
    # Lag features
    lag_features = config.get('features', {}).get('lag_features', [])
    if lag_features:
        df = create_lag_features(df, value_cols, lag_features)
    
    # Difference features
    df = create_difference_features(df, value_cols, periods=[1])
    
    logger.info(f"Feature engineering complete. Total columns: {len(df.columns)}")
    
    return df
