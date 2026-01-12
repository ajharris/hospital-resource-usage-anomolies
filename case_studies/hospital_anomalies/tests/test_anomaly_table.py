"""
Test for the enhanced anomaly dates functionality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from case_studies.hospital_anomalies.src.evaluation import (
    get_anomaly_dates,
    print_anomaly_dates_table,
    _create_google_search_link,
    _get_news_headline_placeholder
)


@pytest.fixture
def sample_anomaly_data():
    """Create sample data with anomalies."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(len(dates)) * 10 + 50
    })
    
    # Add some anomalies
    df['is_anomaly'] = False
    df.loc[10:12, 'is_anomaly'] = True
    df.loc[50:51, 'is_anomaly'] = True
    
    df['anomaly_score'] = np.random.uniform(-0.1, 0.1, len(df))
    df.loc[df['is_anomaly'], 'anomaly_score'] = np.random.uniform(-0.8, -0.5, df['is_anomaly'].sum())
    
    return df


def test_get_anomaly_dates_with_links(sample_anomaly_data):
    """Test that anomaly dates include Google search links."""
    anomaly_dates = get_anomaly_dates(
        sample_anomaly_data,
        include_search_links=True,
        include_news_placeholder=True,
        value_cols=['value']
    )
    
    # Check that we got anomalies
    assert len(anomaly_dates) > 0
    
    # Check that google_search_link column exists
    assert 'google_search_link' in anomaly_dates.columns
    
    # Check that news_headline column exists
    assert 'news_headline' in anomaly_dates.columns
    
    # Check that links are properly formatted
    first_link = anomaly_dates['google_search_link'].iloc[0]
    assert 'google.com/search' in first_link
    assert 'Canada' in first_link
    assert 'gl=ca' in first_link  # Canadian region


def test_google_search_link_creation():
    """Test Google search link creation."""
    test_date = pd.Timestamp('2023-02-15')
    link = _create_google_search_link(test_date)
    
    assert 'google.com/search' in link
    assert '2023-02-15' in link
    assert 'Canada' in link
    assert 'gl=ca' in link


def test_news_headline_placeholder():
    """Test news headline placeholder creation."""
    test_date = pd.Timestamp('2023-02-15')
    placeholder = _get_news_headline_placeholder(test_date)
    
    assert 'February 15, 2023' in placeholder
    assert 'Check news' in placeholder


def test_print_anomaly_dates_table(sample_anomaly_data, capsys):
    """Test that table printing works."""
    anomaly_dates = get_anomaly_dates(
        sample_anomaly_data,
        include_search_links=True,
        value_cols=['value']
    )
    
    # Print the table
    print_anomaly_dates_table(anomaly_dates, max_rows=5)
    
    # Capture output
    captured = capsys.readouterr()
    
    # Check that output contains expected elements
    # The function displays differently in notebooks vs console,
    # so we just verify it ran and produced some output
    assert len(captured.out) > 0
    assert ('ANOMALY DATES SUMMARY' in captured.out or 'HTML object' in captured.out)
