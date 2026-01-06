"""
Schema validation utilities for datasets.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path


class ValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


def validate_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate that required columns are present in the dataframe.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        optional_columns: List of optional column names
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")
    return True


def validate_types(
    df: pd.DataFrame,
    column_types: Dict[str, str]
) -> bool:
    """
    Validate column data types.
    
    Args:
        df: DataFrame to validate
        column_types: Dict mapping column names to expected types
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If column types don't match
    """
    for col, expected_type in column_types.items():
        if col not in df.columns:
            continue
        
        actual_type = df[col].dtype
        if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(actual_type):
            raise ValidationError(f"Column {col} expected numeric type, got {actual_type}")
        elif expected_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(actual_type):
            raise ValidationError(f"Column {col} expected datetime type, got {actual_type}")
        elif expected_type == 'string' and not pd.api.types.is_string_dtype(actual_type) and actual_type != 'object':
            raise ValidationError(f"Column {col} expected string type, got {actual_type}")
    
    return True


def validate_row_count(
    df: pd.DataFrame,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None
) -> bool:
    """
    Validate row count is within expected range.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum expected rows
        max_rows: Maximum expected rows
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If row count is out of range
    """
    row_count = len(df)
    
    if min_rows is not None and row_count < min_rows:
        raise ValidationError(f"Row count {row_count} below minimum {min_rows}")
    
    if max_rows is not None and row_count > max_rows:
        raise ValidationError(f"Row count {row_count} above maximum {max_rows}")
    
    return True


def validate_no_nulls(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> bool:
    """
    Validate that specified columns have no null values.
    
    Args:
        df: DataFrame to validate
        columns: List of column names to check (None = all columns)
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If null values are found
    """
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValidationError(f"Column {col} has {null_count} null values")
    
    return True


def validate_dataset(
    df: pd.DataFrame,
    schema: Dict[str, Any]
) -> bool:
    """
    Run complete validation based on schema definition.
    
    Args:
        df: DataFrame to validate
        schema: Schema definition with validation rules
    
    Returns:
        True if all validations pass
    
    Raises:
        ValidationError: If any validation fails
    """
    if 'required_columns' in schema:
        validate_columns(df, schema['required_columns'], schema.get('optional_columns'))
    
    if 'column_types' in schema:
        validate_types(df, schema['column_types'])
    
    if 'min_rows' in schema or 'max_rows' in schema:
        validate_row_count(df, schema.get('min_rows'), schema.get('max_rows'))
    
    if 'no_null_columns' in schema:
        validate_no_nulls(df, schema['no_null_columns'])
    
    return True
