"""
Tests for quality control and validation functions.
"""

import pytest
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from case_studies.hospital_anomalies.src.qc import (
    check_required_columns,
    check_date_monotonic,
    check_numeric_bounds,
    check_missingness,
    persist_qc_report,
    run_qc_checks,
    QCValidationError
)


@pytest.fixture
def sample_hospital_data():
    """Create sample hospital data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'region': np.random.choice(['Ontario', 'Quebec', 'BC'], len(dates)),
        'hospital_id': np.random.choice(['H001', 'H002', 'H003'], len(dates)),
        'admissions': np.random.randint(20, 100, len(dates)),
        'occupancy_rate': np.random.uniform(60, 95, len(dates)),
        'icu_beds_used': np.random.randint(5, 20, len(dates))
    })
    
    return df


@pytest.fixture
def sample_data_with_issues():
    """Create sample data with various QC issues."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'region': np.random.choice(['Ontario', 'Quebec'], len(dates)),
        'admissions': np.random.randint(20, 100, len(dates)),
        'occupancy_rate': np.random.uniform(60, 95, len(dates)),
    })
    
    # Introduce negative values
    df.loc[5, 'admissions'] = -10
    
    # Introduce out-of-bounds occupancy rate
    df.loc[10, 'occupancy_rate'] = 120
    
    # Introduce missing values
    df.loc[15:20, 'admissions'] = np.nan
    
    return df


def test_check_required_columns_success(sample_hospital_data):
    """Test that required columns check passes when all columns present."""
    required = ['date', 'region', 'hospital_id']
    result = check_required_columns(sample_hospital_data, required)
    
    assert result['passed'] is True
    assert result['missing_columns'] == []
    assert len(result['present_columns']) == len(required)


def test_check_required_columns_failure():
    """Test that required columns check fails when columns are missing."""
    df = pd.DataFrame({'date': [1, 2, 3], 'value': [10, 20, 30]})
    required = ['date', 'region', 'hospital_id']
    
    with pytest.raises(QCValidationError) as exc_info:
        check_required_columns(df, required)
    
    assert 'Missing required columns' in str(exc_info.value)
    assert 'region' in str(exc_info.value)
    assert 'hospital_id' in str(exc_info.value)


def test_check_date_monotonic_success(sample_hospital_data):
    """Test that date monotonic check passes for sorted data."""
    result = check_date_monotonic(sample_hospital_data, date_col='date')
    
    assert result['passed'] is True
    assert result['is_monotonic'] is True


def test_check_date_monotonic_with_grouping(sample_hospital_data):
    """Test date monotonic check with grouping columns."""
    result = check_date_monotonic(
        sample_hospital_data, 
        date_col='date',
        group_cols=['region', 'hospital_id']
    )
    
    assert result['passed'] is True
    assert result['non_monotonic_groups'] == []


def test_check_date_monotonic_unsorted():
    """Test date monotonic check with unsorted data."""
    df = pd.DataFrame({
        'date': ['2023-01-05', '2023-01-03', '2023-01-04', '2023-01-01'],
        'value': [1, 2, 3, 4]
    })
    
    result = check_date_monotonic(df, date_col='date')
    
    # After sorting, dates should be monotonic
    assert result['passed'] is True


def test_check_date_monotonic_missing_column():
    """Test that date monotonic check fails when date column is missing."""
    df = pd.DataFrame({'value': [1, 2, 3]})
    
    with pytest.raises(QCValidationError) as exc_info:
        check_date_monotonic(df, date_col='date')
    
    assert 'Date column' in str(exc_info.value)
    assert 'not found' in str(exc_info.value)


def test_check_numeric_bounds_success(sample_hospital_data):
    """Test numeric bounds check with valid data."""
    bounds_config = {
        'admissions': {'min': 0},
        'occupancy_rate': {'min': 0, 'max': 100},
        'icu_beds_used': {'min': 0}
    }
    
    result = check_numeric_bounds(sample_hospital_data, bounds_config)
    
    assert result['passed'] is True
    assert result['violations'] == {}


def test_check_numeric_bounds_violations(sample_data_with_issues):
    """Test numeric bounds check detects violations."""
    bounds_config = {
        'admissions': {'min': 0},
        'occupancy_rate': {'min': 0, 'max': 100}
    }
    
    result = check_numeric_bounds(sample_data_with_issues, bounds_config)
    
    assert result['passed'] is False
    assert 'admissions' in result['violations']
    assert 'occupancy_rate' in result['violations']
    
    # Check admissions has below minimum violation
    admissions_violations = result['violations']['admissions']
    assert any(v['type'] == 'below_minimum' for v in admissions_violations)
    
    # Check occupancy_rate has above maximum violation
    occupancy_violations = result['violations']['occupancy_rate']
    assert any(v['type'] == 'above_maximum' for v in occupancy_violations)


def test_check_numeric_bounds_missing_column(sample_hospital_data):
    """Test that bounds check handles missing columns gracefully."""
    bounds_config = {
        'admissions': {'min': 0},
        'nonexistent_column': {'min': 0, 'max': 100}
    }
    
    # Should not raise an error, just skip the missing column
    result = check_numeric_bounds(sample_hospital_data, bounds_config)
    
    assert 'nonexistent_column' not in result['violations']


def test_check_missingness_no_missing(sample_hospital_data):
    """Test missingness check with no missing data."""
    result = check_missingness(sample_hospital_data, threshold=0.3)
    
    # All ratios should be 0
    assert all(ratio == 0 for ratio in result.values())


def test_check_missingness_with_missing(sample_data_with_issues):
    """Test missingness check detects missing data."""
    result = check_missingness(sample_data_with_issues, threshold=0.3)
    
    # admissions column should have missing data
    assert result['admissions'] > 0
    # Other columns should have no missing data
    assert result['date'] == 0


def test_check_missingness_exceeds_threshold(sample_data_with_issues):
    """Test that missingness check can fail when threshold is exceeded."""
    # Set a very low threshold
    with pytest.raises(QCValidationError) as exc_info:
        check_missingness(sample_data_with_issues, threshold=0.1, fail_on_threshold=True)
    
    assert 'exceed missing data threshold' in str(exc_info.value)


def test_persist_qc_report(tmp_path):
    """Test that QC report can be persisted as JSON."""
    qc_results = {
        'timestamp': datetime.now().isoformat(),
        'datasets': {
            'test_dataset': {
                'row_count': 100,
                'column_count': 5,
                'checks': {
                    'required_columns': {'passed': True},
                    'date_monotonic': {'passed': True}
                }
            }
        }
    }
    
    output_path = tmp_path / "qc_report.json"
    persist_qc_report(qc_results, output_path)
    
    # Verify file was created
    assert output_path.exists()
    
    # Verify content is valid JSON
    with open(output_path, 'r') as f:
        loaded_results = json.load(f)
    
    assert loaded_results['datasets']['test_dataset']['row_count'] == 100


def test_persist_qc_report_with_numpy_types(tmp_path):
    """Test that QC report handles numpy types correctly."""
    qc_results = {
        'timestamp': datetime.now().isoformat(),
        'datasets': {
            'test_dataset': {
                'row_count': np.int64(100),
                'missing_ratio': np.float64(0.05),
                'violations': {
                    'count': np.int32(5)
                }
            }
        }
    }
    
    output_path = tmp_path / "qc_report.json"
    persist_qc_report(qc_results, output_path)
    
    # Verify file was created and is valid JSON
    assert output_path.exists()
    with open(output_path, 'r') as f:
        loaded_results = json.load(f)
    
    assert loaded_results['datasets']['test_dataset']['row_count'] == 100


def test_run_qc_checks_success(sample_hospital_data, tmp_path):
    """Test full QC checks run successfully on valid data."""
    datasets = {'test_dataset': sample_hospital_data}
    
    config = {
        'qc': {
            'fail_fast': True,
            'required_columns': ['date', 'region'],
            'date_column': 'date',
            'date_monotonic_group_by': ['region', 'hospital_id'],
            'missing_threshold': 0.3,
            'fail_on_missing_threshold': False,
            'numeric_bounds': {
                'admissions': {'min': 0}
            }
        },
        'paths': {
            'results': str(tmp_path)
        }
    }
    
    results = run_qc_checks(datasets, config)
    
    assert 'datasets' in results
    assert 'test_dataset' in results['datasets']
    assert results['datasets']['test_dataset']['status'] == 'passed'
    
    # Check that report was saved
    qc_report_path = tmp_path / 'qc_report.json'
    assert qc_report_path.exists()


def test_run_qc_checks_missing_required_columns(sample_hospital_data, tmp_path):
    """Test that QC checks fail when required columns are missing."""
    # Remove a required column
    df = sample_hospital_data.drop(columns=['region'])
    datasets = {'test_dataset': df}
    
    config = {
        'qc': {
            'fail_fast': True,
            'required_columns': ['date', 'region', 'hospital_id'],
            'date_column': 'date'
        },
        'paths': {
            'results': str(tmp_path)
        }
    }
    
    with pytest.raises(QCValidationError) as exc_info:
        run_qc_checks(datasets, config)
    
    assert 'Missing required columns' in str(exc_info.value)
    
    # Check that partial report was saved
    qc_report_path = tmp_path / 'qc_report.json'
    assert qc_report_path.exists()


def test_run_qc_checks_auto_detect_bounds(sample_hospital_data, tmp_path):
    """Test that QC checks auto-detect numeric bounds."""
    datasets = {'test_dataset': sample_hospital_data}
    
    config = {
        'qc': {
            'fail_fast': False,
            'required_columns': ['date'],
            'date_column': 'date'
        },
        'paths': {
            'results': str(tmp_path)
        }
    }
    
    results = run_qc_checks(datasets, config)
    
    # Check that bounds were checked
    assert 'numeric_bounds' in results['datasets']['test_dataset']['checks']
    
    # Verify auto-detected bounds
    bounds_result = results['datasets']['test_dataset']['checks']['numeric_bounds']
    assert 'admissions' in bounds_result['bounds_checked']
    assert 'occupancy_rate' in bounds_result['bounds_checked']
