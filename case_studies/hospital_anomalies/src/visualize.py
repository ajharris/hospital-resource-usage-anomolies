"""
Visualization utilities for anomaly detection results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
from .utils import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def plot_time_series_with_anomalies(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    anomaly_col: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot time series with anomaly overlays.
    
    Args:
        df: DataFrame with time series and anomalies
        date_col: Name of date column
        value_col: Name of value column to plot
        anomaly_col: Name of anomaly indicator column
        title: Optional plot title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the time series
    ax.plot(df[date_col], df[value_col], label='Values', linewidth=1.5, alpha=0.7)
    
    # Highlight anomalies
    anomalies = df[df[anomaly_col]]
    if len(anomalies) > 0:
        ax.scatter(
            anomalies[date_col],
            anomalies[value_col],
            color='red',
            s=50,
            label='Anomalies',
            zorder=5,
            alpha=0.6
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col)
    ax.set_title(title or f'{value_col} with Anomalies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_anomaly_scores(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot anomaly scores over time.
    
    Args:
        df: DataFrame with anomaly scores
        date_col: Name of date column
        score_col: Name of score column
        threshold: Optional threshold line to show
        title: Optional plot title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df[date_col], df[score_col], linewidth=1, alpha=0.7)
    
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
        ax.legend()
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(title or 'Anomaly Scores Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_seasonal_anomaly_distribution(
    df: pd.DataFrame,
    date_col: str,
    anomaly_col: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot distribution of anomalies by month and quarter.
    
    Args:
        df: DataFrame with dates and anomalies
        date_col: Name of date column
        anomaly_col: Name of anomaly indicator column
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.month
    
    # Calculate anomaly rate by month
    monthly_rates = df.groupby('month')[anomaly_col].mean()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.bar(monthly_rates.index, monthly_rates.values, alpha=0.7)
    ax.set_xlabel('Month')
    ax.set_ylabel('Anomaly Rate')
    ax.set_title('Anomaly Rate by Month')
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def create_anomaly_report_figures(
    df: pd.DataFrame,
    value_cols: List[str],
    config: dict,
    output_dir: Path
) -> List[Path]:
    """
    Generate all visualization figures for the anomaly report.
    
    Args:
        df: DataFrame with predictions and scores
        value_cols: List of value columns to visualize
        config: Configuration dictionary
        output_dir: Directory to save figures
    
    Returns:
        List of paths to saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []
    
    date_col = 'date'
    anomaly_col = 'is_anomaly'
    score_col = 'anomaly_score'
    
    # Plot time series with anomalies for each value column
    for col in value_cols:
        if col in df.columns:
            fig_path = output_dir / f'{col}_with_anomalies.png'
            plot_time_series_with_anomalies(
                df, date_col, col, anomaly_col,
                title=f'{col.replace("_", " ").title()} with Detected Anomalies',
                save_path=fig_path
            )
            plt.close()
            saved_figures.append(fig_path)
    
    # Plot anomaly scores
    score_fig_path = output_dir / 'anomaly_scores.png'
    plot_anomaly_scores(
        df, date_col, score_col,
        title='Anomaly Scores Over Time',
        save_path=score_fig_path
    )
    plt.close()
    saved_figures.append(score_fig_path)
    
    # Plot seasonal distribution
    seasonal_fig_path = output_dir / 'seasonal_distribution.png'
    plot_seasonal_anomaly_distribution(
        df, date_col, anomaly_col,
        save_path=seasonal_fig_path
    )
    plt.close()
    saved_figures.append(seasonal_fig_path)
    
    logger.info(f"Created {len(saved_figures)} figures in {output_dir}")
    
    return saved_figures
