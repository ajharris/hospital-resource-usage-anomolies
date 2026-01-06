"""
Minimal normalization utilities for data transformation.
"""

import pandas as pd
from typing import Optional, List


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase with underscores.
    
    Args:
        df: DataFrame to transform
    
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    return df


def parse_dates(
    df: pd.DataFrame,
    date_columns: List[str],
    date_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse date columns to datetime format.
    
    Args:
        df: DataFrame to transform
        date_columns: List of column names to parse
        date_format: Optional date format string
    
    Returns:
        DataFrame with parsed date columns
    """
    df = df.copy()
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
    return df


def fill_missing_numeric(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'zero'
) -> pd.DataFrame:
    """
    Fill missing values in numeric columns.
    
    Args:
        df: DataFrame to transform
        columns: List of columns to fill (None = all numeric columns)
        method: Fill method ('zero', 'mean', 'median', 'forward', 'backward')
    
    Returns:
        DataFrame with filled values
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'zero':
            df[col] = df[col].fillna(0)
        elif method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'forward':
            df[col] = df[col].fillna(method='ffill')
        elif method == 'backward':
            df[col] = df[col].fillna(method='bfill')
    
    return df


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate rows.
    
    Args:
        df: DataFrame to transform
        subset: Column names to consider for duplicates (None = all columns)
        keep: Which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()
    return df.drop_duplicates(subset=subset, keep=keep)


def normalize_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'minmax'
) -> pd.DataFrame:
    """
    Normalize numeric columns.
    
    Args:
        df: DataFrame to transform
        columns: List of column names to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized columns
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
    
    return df
