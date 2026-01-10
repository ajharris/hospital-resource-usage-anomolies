"""
Quality control and validation for hospital data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .utils import get_logger

logger = get_logger(__name__)


def check_missingness(df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, float]:
    """
    Check for missing data in the dataframe.
    
    Args:
        df: DataFrame to check
        threshold: Maximum allowed missing ratio
    
    Returns:
        Dictionary of column names to missing ratios
    """
    missing_ratios = df.isnull().sum() / len(df)
    
    logger.info("Missing data summary:")
    for col, ratio in missing_ratios.items():
        if ratio > 0:
            logger.info(f"  {col}: {ratio:.2%}")
            if ratio > threshold:
                logger.warning(f"  {col} exceeds threshold ({ratio:.2%} > {threshold:.2%})")
    
    return missing_ratios.to_dict()


def detect_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers in numeric columns.
    
    Args:
        df: DataFrame to check
        columns: List of columns to check for outliers
        method: Detection method ('zscore' or 'iqr')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outlier flags for each column
    """
    outliers = pd.DataFrame(index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[f'{col}_outlier'] = z_scores > threshold
            outlier_count = outliers[f'{col}_outlier'].sum()
            
        elif method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers[f'{col}_outlier'].sum()
        
        logger.info(f"Outliers in {col}: {outlier_count} ({outlier_count/len(df):.2%})")
    
    return outliers


def check_seasonality(
    df: pd.DataFrame,
    date_col: str,
    value_col: str
) -> Dict[str, float]:
    """
    Perform basic seasonality checks on time-series data.
    
    Args:
        df: DataFrame with time-series data
        date_col: Name of date column
        value_col: Name of value column
    
    Returns:
        Dictionary with seasonality metrics
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Extract month
    df['month'] = df[date_col].dt.month
    
    # Calculate monthly averages
    monthly_avg = df.groupby('month')[value_col].mean()
    
    # Calculate coefficient of variation
    cv = monthly_avg.std() / monthly_avg.mean()
    
    # Find peak and trough months
    peak_month = monthly_avg.idxmax()
    trough_month = monthly_avg.idxmin()
    
    seasonality_info = {
        'coefficient_of_variation': cv,
        'peak_month': int(peak_month),
        'trough_month': int(trough_month),
        'peak_value': float(monthly_avg.max()),
        'trough_value': float(monthly_avg.min()),
    }
    
    logger.info(f"Seasonality analysis for {value_col}:")
    logger.info(f"  CV: {cv:.3f}")
    logger.info(f"  Peak month: {peak_month} (avg: {monthly_avg.max():.2f})")
    logger.info(f"  Trough month: {trough_month} (avg: {monthly_avg.min():.2f})")
    
    return seasonality_info


def validate_date_range(
    df: pd.DataFrame,
    date_col: str,
    expected_start: str,
    expected_end: str
) -> bool:
    """
    Validate that data covers the expected date range.
    
    Args:
        df: DataFrame to validate
        date_col: Name of date column
        expected_start: Expected start date (YYYY-MM-DD)
        expected_end: Expected end date (YYYY-MM-DD)
    
    Returns:
        True if date range is valid
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    actual_start = df[date_col].min()
    actual_end = df[date_col].max()
    expected_start = pd.to_datetime(expected_start)
    expected_end = pd.to_datetime(expected_end)
    
    logger.info(f"Date range validation:")
    logger.info(f"  Expected: {expected_start.date()} to {expected_end.date()}")
    logger.info(f"  Actual:   {actual_start.date()} to {actual_end.date()}")
    
    if actual_start > expected_start:
        logger.warning(f"Data starts later than expected")
    if actual_end < expected_end:
        logger.warning(f"Data ends earlier than expected")
    
    return actual_start <= expected_start and actual_end >= expected_end


def run_qc_checks(
    datasets: Dict[str, pd.DataFrame],
    config: dict
) -> Dict[str, dict]:
    """
    Run all quality control checks on datasets.
    
    Args:
        datasets: Dictionary of dataset_id to DataFrame
        config: Configuration dictionary with QC parameters
    
    Returns:
        Dictionary of QC results for each dataset
    """
    qc_results = {}
    
    for dataset_id, df in datasets.items():
        logger.info(f"\n=== QC Checks for {dataset_id} ===")
        
        results = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_data': {},
            'outliers': None,
        }
        
        # Check missingness
        missing_threshold = config.get('processing', {}).get('missing_threshold', 0.3)
        results['missing_data'] = check_missingness(df, threshold=missing_threshold)
        
        # Detect outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            outlier_threshold = config.get('processing', {}).get('outlier_std_threshold', 3.0)
            results['outliers'] = detect_outliers(df, numeric_cols, threshold=outlier_threshold)
        
        # Check date range if configured
        if 'date_range' in config and 'date' in df.columns:
            results['date_range_valid'] = validate_date_range(
                df, 'date',
                config['date_range']['start'],
                config['date_range']['end']
            )
        
        qc_results[dataset_id] = results
    
    return qc_results
