"""
Quality control and validation for hospital data.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from .utils import get_logger
from .io import get_output_path

logger = get_logger(__name__)


class QCValidationError(Exception):
    """Exception raised when QC validation fails."""
    pass


def check_required_columns(
    df: pd.DataFrame, 
    required_columns: List[str]
) -> Dict[str, Any]:
    """
    Check that all required columns exist in the DataFrame.
    
    Args:
        df: DataFrame to check
        required_columns: List of required column names
    
    Returns:
        Dictionary with check results
        
    Raises:
        QCValidationError: If any required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    result = {
        'required_columns': required_columns,
        'present_columns': [col for col in required_columns if col in df.columns],
        'missing_columns': missing_columns,
        'passed': len(missing_columns) == 0
    }
    
    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        logger.error(error_msg)
        raise QCValidationError(error_msg)
    
    logger.info(f"✓ All {len(required_columns)} required columns present")
    return result


def check_date_monotonic(
    df: pd.DataFrame, 
    date_col: str = 'date',
    group_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Check that the date column is monotonic (increasing) after sorting.
    
    Args:
        df: DataFrame to check
        date_col: Name of the date column
        group_cols: Optional columns to group by before checking monotonicity
    
    Returns:
        Dictionary with check results
    """
    if date_col not in df.columns:
        error_msg = f"Date column '{date_col}' not found in DataFrame"
        logger.error(error_msg)
        raise QCValidationError(error_msg)
    
    df_check = df.copy()
    df_check[date_col] = pd.to_datetime(df_check[date_col])
    
    if group_cols:
        # Check monotonicity within each group
        non_monotonic_groups = []
        for group_name, group_df in df_check.groupby(group_cols):
            sorted_dates = group_df.sort_values(date_col)[date_col]
            if not sorted_dates.is_monotonic_increasing:
                non_monotonic_groups.append(group_name)
        
        result = {
            'date_column': date_col,
            'group_columns': group_cols,
            'non_monotonic_groups': non_monotonic_groups,
            'passed': len(non_monotonic_groups) == 0
        }
        
        if non_monotonic_groups:
            logger.warning(f"Date column not monotonic in {len(non_monotonic_groups)} groups")
        else:
            logger.info(f"✓ Date column is monotonic within all groups")
    else:
        # Check global monotonicity after sorting
        sorted_dates = df_check.sort_values(date_col)[date_col]
        is_monotonic = sorted_dates.is_monotonic_increasing
        
        result = {
            'date_column': date_col,
            'is_monotonic': is_monotonic,
            'passed': is_monotonic
        }
        
        if is_monotonic:
            logger.info(f"✓ Date column '{date_col}' is monotonic after sorting")
        else:
            logger.warning(f"Date column '{date_col}' is not monotonic after sorting")
    
    return result


def check_numeric_bounds(
    df: pd.DataFrame,
    bounds_config: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Check that numeric columns satisfy specified bounds.
    
    Args:
        df: DataFrame to check
        bounds_config: Dictionary mapping column names to bounds specs
            e.g., {'admissions': {'min': 0}, 'occupancy_rate': {'min': 0, 'max': 100}}
    
    Returns:
        Dictionary with check results
    """
    violations = {}
    
    for col, bounds in bounds_config.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping bounds check")
            continue
        
        col_violations = []
        
        if 'min' in bounds:
            min_val = bounds['min']
            below_min = df[col] < min_val
            n_below = below_min.sum()
            if n_below > 0:
                col_violations.append({
                    'type': 'below_minimum',
                    'bound': min_val,
                    'count': int(n_below),
                    'percentage': float(n_below / len(df) * 100)
                })
                logger.warning(f"  {col}: {n_below} values ({n_below/len(df):.2%}) below minimum {min_val}")
        
        if 'max' in bounds:
            max_val = bounds['max']
            above_max = df[col] > max_val
            n_above = above_max.sum()
            if n_above > 0:
                col_violations.append({
                    'type': 'above_maximum',
                    'bound': max_val,
                    'count': int(n_above),
                    'percentage': float(n_above / len(df) * 100)
                })
                logger.warning(f"  {col}: {n_above} values ({n_above/len(df):.2%}) above maximum {max_val}")
        
        if col_violations:
            violations[col] = col_violations
        else:
            logger.info(f"✓ {col}: all values within bounds")
    
    result = {
        'bounds_checked': bounds_config,
        'violations': violations,
        'passed': len(violations) == 0
    }
    
    return result


def check_missingness(df: pd.DataFrame, threshold: float = 0.3, fail_on_threshold: bool = False) -> Dict[str, float]:
    """
    Check for missing data in the dataframe.
    
    Args:
        df: DataFrame to check
        threshold: Maximum allowed missing ratio
        fail_on_threshold: If True, raise exception when threshold is exceeded
    
    Returns:
        Dictionary of column names to missing ratios
        
    Raises:
        QCValidationError: If fail_on_threshold=True and any column exceeds threshold
    """
    missing_ratios = df.isnull().sum() / len(df)
    
    logger.info("Missing data summary:")
    exceeds_threshold = []
    for col, ratio in missing_ratios.items():
        if ratio > 0:
            logger.info(f"  {col}: {ratio:.2%}")
            if ratio > threshold:
                logger.warning(f"  {col} exceeds threshold ({ratio:.2%} > {threshold:.2%})")
                exceeds_threshold.append(col)
    
    if fail_on_threshold and exceeds_threshold:
        error_msg = f"Columns exceed missing data threshold: {exceeds_threshold}"
        logger.error(error_msg)
        raise QCValidationError(error_msg)
    
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


def persist_qc_report(
    qc_results: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save QC report to a JSON file.
    
    Args:
        qc_results: Dictionary with QC results
        output_path: Path to save the JSON report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(qc_results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"QC report saved to {output_path}")


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
        
    Raises:
        QCValidationError: If critical validation checks fail
    """
    qc_results = {
        'timestamp': datetime.now().isoformat(),
        'datasets': {}
    }
    
    # Get QC configuration
    qc_config = config.get('qc', {})
    fail_fast = qc_config.get('fail_fast', True)
    
    for dataset_id, df in datasets.items():
        logger.info(f"\n=== QC Checks for {dataset_id} ===")
        
        results = {
            'dataset_id': dataset_id,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'checks': {}
        }
        
        try:
            # 1. Check required columns
            required_columns = qc_config.get('required_columns', ['date'])
            results['checks']['required_columns'] = check_required_columns(df, required_columns)
            
            # 2. Check date monotonicity
            date_col = qc_config.get('date_column', 'date')
            if date_col in df.columns:
                group_cols = qc_config.get('date_monotonic_group_by', ['region', 'hospital_id'])
                # Only use group columns that exist in the dataframe
                existing_group_cols = [col for col in group_cols if col in df.columns]
                results['checks']['date_monotonic'] = check_date_monotonic(
                    df, 
                    date_col=date_col,
                    group_cols=existing_group_cols if existing_group_cols else None
                )
            
            # 3. Check missingness
            missing_threshold = qc_config.get('missing_threshold', 
                                             config.get('processing', {}).get('missing_threshold', 0.3))
            fail_on_missing = qc_config.get('fail_on_missing_threshold', False)
            results['checks']['missingness'] = {
                'threshold': missing_threshold,
                'ratios': check_missingness(df, threshold=missing_threshold, fail_on_threshold=fail_on_missing)
            }
            
            # 4. Check numeric bounds
            # Auto-detect bounds based on column names
            bounds_config = qc_config.get('numeric_bounds')
            if bounds_config is None:
                bounds_config = {}
            
            # Add default bounds for common column patterns
            for col in df.columns:
                if col not in bounds_config:
                    # Admissions should be non-negative
                    if 'admission' in col.lower() or 'beds_used' in col.lower() or 'icu_beds' in col.lower():
                        bounds_config[col] = {'min': 0}
                    # Rates and percentages should be in [0, 100]
                    elif col.lower().endswith('_rate'):
                        bounds_config[col] = {'min': 0, 'max': 100}
                    # Occupancy as decimal should be in [0, 1]
                    # Use threshold of 1.5 to distinguish decimal (<=1) from percentage (>1)
                    elif 'occupancy' in col.lower() and df[col].max() <= 1.5:
                        bounds_config[col] = {'min': 0, 'max': 1}
            
            if bounds_config:
                results['checks']['numeric_bounds'] = check_numeric_bounds(df, bounds_config)
            
            # 5. Detect outliers (existing functionality)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols and qc_config.get('check_outliers', False):
                outlier_threshold = config.get('processing', {}).get('outlier_std_threshold', 3.0)
                outlier_results = detect_outliers(df, numeric_cols, threshold=outlier_threshold)
                results['checks']['outliers'] = {
                    'method': 'zscore',
                    'threshold': outlier_threshold,
                    'summary': {col: int(outlier_results[col].sum()) for col in outlier_results.columns}
                }
            
            # 6. Check date range if configured
            if 'date_range' in config and date_col in df.columns:
                date_range_valid = validate_date_range(
                    df, date_col,
                    config['date_range']['start'],
                    config['date_range']['end']
                )
                results['checks']['date_range'] = {
                    'expected_start': config['date_range']['start'],
                    'expected_end': config['date_range']['end'],
                    'valid': date_range_valid
                }
            
            # Overall status
            results['status'] = 'passed'
            
        except QCValidationError as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"QC validation failed for {dataset_id}: {e}")
            
            if fail_fast:
                # Save partial results before failing
                qc_results['datasets'][dataset_id] = results
                qc_report_path = get_output_path(config, 'results', 'qc_report.json')
                persist_qc_report(qc_results, qc_report_path)
                raise
        
        qc_results['datasets'][dataset_id] = results
    
    # Persist QC report
    qc_report_path = get_output_path(config, 'results', 'qc_report.json')
    persist_qc_report(qc_results, qc_report_path)
    
    logger.info(f"\n✓ QC checks complete. Report saved to {qc_report_path}")
    
    return qc_results
