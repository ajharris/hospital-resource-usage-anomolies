"""
Evaluation utilities for unsupervised anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .utils import get_logger

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


def get_anomaly_dates(
    df: pd.DataFrame,
    date_col: str = 'date',
    anomaly_col: str = 'is_anomaly',
    score_col: str = 'anomaly_score',
    include_scores: bool = True,
    value_cols: Optional[List[str]] = None,
    include_search_links: bool = True,
    include_news_placeholder: bool = True
) -> pd.DataFrame:
    """
    Extract all anomaly dates with associated information.
    
    This function creates a summary table of all detected anomalies,
    including dates, scores, Google search links for Canadian events,
    and placeholder for news headlines.
    
    Args:
        df: DataFrame with anomaly predictions
        date_col: Name of date column
        anomaly_col: Name of anomaly indicator column
        score_col: Name of anomaly score column
        include_scores: Whether to include anomaly scores
        value_cols: Optional list of value columns to include
        include_search_links: Whether to include Google search links
        include_news_placeholder: Whether to include news headline placeholder
    
    Returns:
        DataFrame with anomaly dates and related information
    """
    # Filter to only anomalies
    anomalies_df = df[df[anomaly_col]].copy()
    
    if len(anomalies_df) == 0:
        logger.warning("No anomalies detected in the dataset")
        return pd.DataFrame()
    
    # Select columns to include
    cols_to_include = [date_col]
    
    if include_scores and score_col in anomalies_df.columns:
        cols_to_include.append(score_col)
    
    if value_cols:
        for col in value_cols:
            if col in anomalies_df.columns:
                cols_to_include.append(col)
    
    # Create summary dataframe
    anomaly_summary = anomalies_df[cols_to_include].copy()
    
    # Add Google search links for Canadian events
    if include_search_links:
        anomaly_summary['google_search_link'] = anomaly_summary[date_col].apply(
            lambda d: _create_google_search_link(d)
        )
    
    # Add news headline placeholder
    if include_news_placeholder:
        anomaly_summary['news_headline'] = anomaly_summary[date_col].apply(
            lambda d: _get_news_headline_placeholder(d)
        )
    
    # Sort by date (or by score if available)
    if include_scores and score_col in anomaly_summary.columns:
        anomaly_summary = anomaly_summary.sort_values(score_col)
    else:
        anomaly_summary = anomaly_summary.sort_values(date_col)
    
    # Reset index for clean output
    anomaly_summary = anomaly_summary.reset_index(drop=True)
    
    logger.info(f"Extracted {len(anomaly_summary)} anomaly dates")
    
    return anomaly_summary


def _create_google_search_link(date_value) -> str:
    """
    Create a Google search link for Canadian news/events on a specific date.
    
    Args:
        date_value: Date value (can be string, datetime, or Timestamp)
    
    Returns:
        Google search URL string
    """
    # Convert to pandas Timestamp if needed
    date_ts = pd.to_datetime(date_value)
    date_str = date_ts.strftime('%Y-%m-%d')
    
    # Create Google search query for Canadian news/events on that date
    # Format: "Canada news [date]" limited to Canadian sites
    query = f"Canada news {date_str}"
    # URL encode the query
    import urllib.parse
    encoded_query = urllib.parse.quote(query)
    
    # Create Google search URL with Canadian region preference
    search_url = f"https://www.google.com/search?q={encoded_query}&gl=ca&hl=en"
    
    return search_url


def _get_news_headline_placeholder(date_value) -> str:
    """
    Get a placeholder for news headline on a specific date.
    
    In production, this could call a news API. For now, returns a placeholder
    indicating manual lookup is needed.
    
    Args:
        date_value: Date value (can be string, datetime, or Timestamp)
    
    Returns:
        Placeholder text or actual headline if available
    """
    # Convert to pandas Timestamp if needed
    date_ts = pd.to_datetime(date_value)
    date_str = date_ts.strftime('%B %d, %Y')
    
    # Return placeholder - in production this would call a news API
    return f"[Check news for {date_str}]"


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


def print_anomaly_dates_table(
    anomaly_dates: pd.DataFrame,
    date_col: str = 'date',
    max_rows: Optional[int] = None
):
    """
    Print anomaly dates as a formatted table to console.
    
    Args:
        anomaly_dates: DataFrame with anomaly dates and information
        date_col: Name of date column
        max_rows: Maximum number of rows to print (None for all)
    """
    if len(anomaly_dates) == 0:
        logger.info("No anomalies to display")
        return
    
    # Limit rows if specified
    display_df = anomaly_dates.head(max_rows) if max_rows else anomaly_dates
    
    # Print header
    print("\n" + "=" * 120)
    print("ANOMALY DATES SUMMARY")
    print("=" * 120)
    print(f"Total anomalies detected: {len(anomaly_dates)}")
    if max_rows and len(anomaly_dates) > max_rows:
        print(f"Showing top {max_rows} anomalies")
    print("=" * 120)
    
    # Print table
    # Use pandas to_string with formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 80)
    
    print(display_df.to_string(index=True))
    
    print("=" * 120)
    
    # Print instructions for using Google search links
    if 'google_search_link' in display_df.columns:
        print("\nTo investigate events on these dates:")
        print("- Copy the google_search_link URL from the table above")
        print("- Paste it into your browser to search for Canadian news/events on that date")
    
    print("\n")


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
    
    # All anomaly dates
    value_cols = [col for col in df.columns if col not in [date_col, anomaly_col, score_col, 'prediction']
                  and not any(x in col.lower() for x in ['id', 'rolling', 'lag', '_diff_', '_pct_', 
                                                          'year', 'month', 'day', 'week', 'quarter',
                                                          'sin', 'cos', '_std_', '_min_', '_max_'])]
    results['anomaly_dates'] = get_anomaly_dates(
        df, date_col, anomaly_col, score_col, 
        include_scores=True, 
        value_cols=value_cols[:5]  # Limit to first 5 value columns to keep table manageable
    )
    
    # Seasonality check
    if config.get('evaluation', {}).get('seasonality_check', True):
        results['seasonality'] = seasonality_sanity_check(df, date_col, anomaly_col)
    
    # Stability metrics
    results['stability'] = calculate_stability_metrics(df, score_col)
    
    return results
