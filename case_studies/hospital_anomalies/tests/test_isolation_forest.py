"""
Tests for Isolation Forest model.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from case_studies.hospital_anomalies.src.models.isolation_forest import IsolationForestDetector


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Create mostly normal data
    normal_data = np.random.randn(n_samples - 50, n_features)
    
    # Add some anomalies
    anomalies = np.random.randn(50, n_features) * 3 + 5
    
    data = np.vstack([normal_data, anomalies])
    np.random.shuffle(data)
    
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    return df


def test_isolation_forest_init():
    """Test Isolation Forest initialization."""
    detector = IsolationForestDetector(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    
    assert detector.n_estimators == 100
    assert detector.contamination == 0.05
    assert detector.random_state == 42
    assert not detector.is_fitted_


def test_isolation_forest_fit(sample_data):
    """Test model fitting."""
    detector = IsolationForestDetector(random_state=42)
    detector.fit(sample_data)
    
    assert detector.is_fitted_
    assert detector.feature_columns_ == sample_data.columns.tolist()


def test_isolation_forest_predict(sample_data):
    """Test prediction."""
    detector = IsolationForestDetector(contamination=0.1, random_state=42)
    detector.fit(sample_data)
    
    predictions = detector.predict(sample_data)
    
    assert len(predictions) == len(sample_data)
    assert set(predictions).issubset({-1, 1})
    
    # Should detect approximately 10% as anomalies (contamination=0.1)
    anomaly_rate = (predictions == -1).sum() / len(predictions)
    assert 0.05 <= anomaly_rate <= 0.15


def test_isolation_forest_score_samples(sample_data):
    """Test anomaly scoring."""
    detector = IsolationForestDetector(random_state=42)
    detector.fit(sample_data)
    
    scores = detector.score_samples(sample_data)
    
    assert len(scores) == len(sample_data)
    assert scores.dtype == np.float64


def test_isolation_forest_get_anomalies(sample_data):
    """Test getting anomaly results."""
    detector = IsolationForestDetector(random_state=42)
    detector.fit(sample_data)
    
    results = detector.get_anomalies(sample_data)
    
    assert isinstance(results, pd.DataFrame)
    assert 'prediction' in results.columns
    assert 'anomaly_score' in results.columns
    assert 'is_anomaly' in results.columns


def test_isolation_forest_save_load(sample_data):
    """Test model save and load."""
    detector = IsolationForestDetector(random_state=42)
    detector.fit(sample_data)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / 'test_model.joblib'
        detector.save(model_path)
        
        assert model_path.exists()
        
        # Load model
        loaded_detector = IsolationForestDetector.load(model_path)
        
        assert loaded_detector.is_fitted_
        assert loaded_detector.feature_columns_ == detector.feature_columns_
        
        # Check predictions are the same
        orig_preds = detector.predict(sample_data)
        loaded_preds = loaded_detector.predict(sample_data)
        
        np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_deterministic_behavior(sample_data):
    """Test that results are deterministic with fixed random seed."""
    detector1 = IsolationForestDetector(random_state=42)
    detector1.fit(sample_data)
    preds1 = detector1.predict(sample_data)
    
    detector2 = IsolationForestDetector(random_state=42)
    detector2.fit(sample_data)
    preds2 = detector2.predict(sample_data)
    
    np.testing.assert_array_equal(preds1, preds2)
