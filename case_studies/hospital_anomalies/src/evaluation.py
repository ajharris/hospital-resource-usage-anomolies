"""
Evaluation utilities for unsupervised anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from publicdata_ca.utils.logging import get_logger

logger = get_logger(__name__)


def check_anomaly_persistence(
    df: pd.DataFrame,
    date_col: str,
    anomaly_col: str,
    window: int = 7
) -> pd.DataFrame:
    """
    Check if anomalies persist over a time window.
    
    Args:
        df: DataFrame with anomalies
        date_col: Name of date column
        anomaly_col: Name of anomaly indicator column
        window: Number of days to check for persistence
    
    Returns:
        DataFrame with persistence metrics
    """
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Calculate rolling sum of anomalies
    df['anomaly_count_window'] = df[anomaly_col].rolling(
        window=window, min_periods=1
    ).sum()
    
    # Mark persistent anomalies (multiple anomalies in window)
    df['persistent_anomaly'] = df['anomaly_count_window'] > 1
    
    persistent_count = df['persistent_anomaly'].sum()
    total_anomalies = df[anomaly_col].sum()
    
    logger.info(f"Persistence check (window={window} days):")
    logger.info(f"  Total anomalies: {total_anomalies}")
    logger.info(f"  Persistent anomalies: {persistent_count}")
    if total_anomalies > 0:
        logger.info(f"  Persistence rate: {persistent_count/total_anomalies:.2%}")
    
    return df


def get_top_anomalies(
    df: pd.DataFrame,
    score_col: str,
    k: int = 20,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Get top-k most anomalous periods.
    
    Args:
        df: DataFrame with anomaly scores
        score_col: Name of anomaly score column
        k: Number of top anomalies to return
        ascending: If True, lower scores are more anomalous
    
    Returns:
        DataFrame with top-k anomalies
    """
    top_k = df.nsmallest(k, score_col) if ascending else df.nlargest(k, score_col)
    
    logger.info(f"Top {k} anomalies:")
    for idx, row in top_k.iterrows():
        logger.info(f"  {row.get('date', idx)}: score={row[score_col]:.4f}")
    
    return top_k


def seasonality_sanity_check(
    df: pd.DataFrame,
    date_col: str,
    anomaly_col: str
) -> Dict[str, float]:
    """
    Check if anomalies align with expected seasonal patterns.
    
    Args:
        df: DataFrame with dates and anomalies
        date_col: Name of date column
        anomaly_col: Name of anomaly indicator column
    
    Returns:
        Dictionary with seasonal distribution of anomalies
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    
    # Calculate anomaly rate by month
    monthly_anomalies = df.groupby('month')[anomaly_col].agg(['sum', 'count', 'mean'])
    
    # Calculate anomaly rate by quarter
    quarterly_anomalies = df.groupby('quarter')[anomaly_col].agg(['sum', 'count', 'mean'])
    
    logger.info("Seasonal anomaly distribution:")
    logger.info("\nBy month:")
    for month, row in monthly_anomalies.iterrows():
        logger.info(f"  Month {month}: {row['sum']} anomalies ({row['mean']:.2%} rate)")
    
    logger.info("\nBy quarter:")
    for quarter, row in quarterly_anomalies.iterrows():
        logger.info(f"  Q{quarter}: {row['sum']} anomalies ({row['mean']:.2%} rate)")
    
    return {
        'monthly': monthly_anomalies.to_dict(),
        'quarterly': quarterly_anomalies.to_dict()
    }


def calculate_stability_metrics(
    df: pd.DataFrame,
    score_col: str,
    window: int = 30
) -> Dict[str, float]:
    """
    Calculate stability metrics for anomaly scores over time.
    
    Args:
        df: DataFrame with anomaly scores
        score_col: Name of score column
        window: Window size for rolling statistics
    
    Returns:
        Dictionary of stability metrics
    """
    # Calculate rolling statistics
    rolling_mean = df[score_col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[score_col].rolling(window=window, min_periods=1).std()
    
    # Calculate coefficient of variation
    cv = rolling_std / rolling_mean
    
    metrics = {
        'mean_score': float(df[score_col].mean()),
        'std_score': float(df[score_col].std()),
        'cv_mean': float(cv.mean()),
        'cv_std': float(cv.std()),
        'score_range': float(df[score_col].max() - df[score_col].min())
    }
    
    logger.info("Stability metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return metrics


def evaluate_anomalies(
    df: pd.DataFrame,
    config: dict,
    date_col: str = 'date',
    anomaly_col: str = 'is_anomaly',
    score_col: str = 'anomaly_score'
) -> Dict[str, any]:
    """
    Run all evaluation checks for anomaly detection results.
    
    Args:
        df: DataFrame with predictions and scores
        config: Configuration dictionary
        date_col: Name of date column
        anomaly_col: Name of anomaly indicator column
        score_col: Name of anomaly score column
    
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    # Persistence check
    persistence_window = config.get('evaluation', {}).get('persistence_window', 7)
    df_with_persistence = check_anomaly_persistence(
        df, date_col, anomaly_col, window=persistence_window
    )
    results['persistence'] = df_with_persistence
    
    # Top-k anomalies
    top_k = config.get('evaluation', {}).get('top_k_anomalies', 20)
    results['top_anomalies'] = get_top_anomalies(df, score_col, k=top_k)
    
    # Seasonality check
    if config.get('evaluation', {}).get('seasonality_check', True):
        results['seasonality'] = seasonality_sanity_check(df, date_col, anomaly_col)
    
    # Stability metrics
    results['stability'] = calculate_stability_metrics(df, score_col)
    
    return results
