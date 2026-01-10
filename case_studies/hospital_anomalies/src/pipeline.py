"""
Pipeline orchestration for the hospital anomalies case study.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .utils import setup_logging, get_logger, load_config

from .ingest import ingest_cihi_data
from .qc import run_qc_checks
from .features import engineer_features
from .models.isolation_forest import IsolationForestDetector
from .evaluation import evaluate_anomalies
from .visualize import create_anomaly_report_figures
from .io import save_results_summary, save_features, ensure_output_dirs, get_output_path

logger = get_logger(__name__)


def run_pipeline(config_path: Path) -> Dict[str, Any]:
    """
    Run the complete anomaly detection pipeline.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dictionary with pipeline results
    """
    # Load configuration
    config = load_config(config_path)
    config_dict = config.to_dict()
    
    # Setup logging
    setup_logging(level=config.get('log_level', 'INFO'))
    logger.info("=" * 80)
    logger.info("Hospital Anomalies Detection Pipeline")
    logger.info("=" * 80)
    
    # Ensure output directories exist
    ensure_output_dirs(config_dict)
    
    # Step 1: Ingest data
    logger.info("\n[Step 1/6] Ingesting CIHI data...")
    dataset_ids = config_dict.get('datasets', [])
    datasets = ingest_cihi_data(dataset_ids)
    
    # Step 2: Quality control
    logger.info("\n[Step 2/6] Running quality control checks...")
    qc_results = run_qc_checks(datasets, config_dict)
    
    # Step 3: Combine and prepare data
    logger.info("\n[Step 3/6] Preparing combined dataset...")
    # For simplicity, use the first dataset with the most features
    # In production, you might merge multiple datasets
    combined_df = None
    for dataset_id, df in datasets.items():
        if combined_df is None or len(df.columns) > len(combined_df.columns):
            combined_df = df.copy()
    
    # Step 4: Feature engineering
    logger.info("\n[Step 4/6] Engineering features...")
    features_df = engineer_features(combined_df, config_dict)
    
    # Save features if configured
    if config_dict.get('output', {}).get('save_features', True):
        save_features(features_df, config_dict)
    
    # Step 5: Train model and detect anomalies
    logger.info("\n[Step 5/6] Training model and detecting anomalies...")
    
    # Prepare feature matrix
    feature_cols = [
        col for col in features_df.columns
        if col not in ['date', 'region', 'hospital_id', 'year', 'month', 'day', 'quarter', 'week_of_year', 'day_of_week', 'day_of_year']
        and not col.endswith('_outlier')
    ]
    X = features_df[feature_cols]
    
    # Train Isolation Forest
    if_config = config_dict.get('isolation_forest', {})
    detector = IsolationForestDetector(
        n_estimators=if_config.get('n_estimators', 100),
        max_samples=if_config.get('max_samples', 256),
        contamination=if_config.get('contamination', 0.05),
        random_state=if_config.get('random_state', 42),
        n_jobs=if_config.get('n_jobs', -1)
    )
    
    detector.fit(X)
    
    # Get predictions
    predictions = detector.get_anomalies(X)
    
    # Combine with original data
    results_df = features_df.copy()
    results_df['is_anomaly'] = predictions['is_anomaly']
    results_df['anomaly_score'] = predictions['anomaly_score']
    results_df['prediction'] = predictions['prediction']
    
    # Save model
    model_path = get_output_path(config_dict, 'models', 'isolation_forest.joblib')
    detector.save(model_path)
    
    # Step 6: Evaluate and visualize
    logger.info("\n[Step 6/6] Evaluating results and creating visualizations...")
    
    # Run evaluation
    eval_results = evaluate_anomalies(results_df, config_dict)
    
    # Save top anomalies
    if config_dict.get('output', {}).get('save_predictions', True):
        save_results_summary(eval_results['top_anomalies'], config_dict)
    
    # Create visualizations
    if config_dict.get('output', {}).get('save_figures', True):
        # Identify value columns
        value_cols = [col for col in combined_df.columns if col not in ['date', 'region', 'hospital_id']]
        figures_dir = get_output_path(config_dict, 'figures', '')
        created_figures = create_anomaly_report_figures(results_df, value_cols, config_dict, figures_dir)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Complete!")
    logger.info(f"Total records processed: {len(results_df)}")
    logger.info(f"Anomalies detected: {predictions['is_anomaly'].sum()} ({predictions['is_anomaly'].mean():.2%})")
    logger.info("=" * 80)
    
    return {
        'datasets': datasets,
        'qc_results': qc_results,
        'features': features_df,
        'predictions': results_df,
        'evaluation': eval_results,
        'model': detector
    }


if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    
    run_pipeline(config_path)
