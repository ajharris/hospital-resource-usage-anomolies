"""
Isolation Forest anomaly detection model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, Dict, Any
import joblib
from pathlib import Path
from ..utils import get_logger

logger = get_logger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest wrapper for anomaly detection with deterministic behavior.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of isolation trees
            max_samples: Number of samples to draw for each tree
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_columns_ = None
        self.feature_means_ = None  # Store means for consistent imputation
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest model.
        
        Args:
            X: Training data (features only)
        
        Returns:
            Self
        """
        # Store feature columns and statistics for imputation
        self.feature_columns_ = X.columns.tolist()
        self.feature_means_ = X.mean().to_dict()
        
        # Drop rows with NaN values
        X_clean = X.dropna()
        logger.info(f"Fitting Isolation Forest on {len(X_clean)} samples with {len(self.feature_columns_)} features")
        
        # Fit the model
        self.model.fit(X_clean)
        self.is_fitted_ = True
        
        logger.info("Model fitting complete")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies (1 = normal, -1 = anomaly).
        
        Args:
            X: Data to predict on
        
        Returns:
            Array of predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure same feature columns
        X = X[self.feature_columns_]
        
        # Handle NaN values using training statistics
        X_clean = X.fillna(pd.Series(self.feature_means_))
        
        predictions = self.model.predict(X_clean)
        return predictions
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores for samples (lower = more anomalous).
        
        Args:
            X: Data to score
        
        Returns:
            Array of anomaly scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before scoring")
        
        # Ensure same feature columns
        X = X[self.feature_columns_]
        
        # Handle NaN values using training statistics
        X_clean = X.fillna(pd.Series(self.feature_means_))
        
        scores = self.model.score_samples(X_clean)
        return scores
    
    def get_anomalies(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get anomaly predictions and scores.
        
        Args:
            X: Data to evaluate
            threshold: Optional custom threshold for anomaly scores
        
        Returns:
            DataFrame with predictions and scores
        """
        predictions = self.predict(X)
        scores = self.score_samples(X)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'anomaly_score': scores,
            'is_anomaly': predictions == -1
        }, index=X.index)
        
        if threshold is not None:
            results['is_anomaly'] = scores < threshold
        
        return results
    
    def save(self, path: Path):
        """
        Save the fitted model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns_,
            'feature_means': self.feature_means_,
            'params': {
                'n_estimators': self.n_estimators,
                'max_samples': self.max_samples,
                'contamination': self.contamination,
                'random_state': self.random_state,
            }
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'IsolationForestDetector':
        """
        Load a fitted model from disk.
        
        Args:
            path: Path to load the model from
        
        Returns:
            Loaded detector instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        # Create instance with saved parameters
        detector = cls(**model_data['params'])
        detector.model = model_data['model']
        detector.feature_columns_ = model_data['feature_columns']
        detector.feature_means_ = model_data.get('feature_means', {})
        detector.is_fitted_ = True
        
        logger.info(f"Model loaded from {path}")
        return detector
