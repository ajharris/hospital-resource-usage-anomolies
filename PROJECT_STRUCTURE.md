# Project Structure Overview

This document provides a comprehensive overview of the hospital resource usage anomalies project structure.

## Directory Structure

```
hospital-resource-usage-anomolies/
├── README.md                          # Project overview
├── LICENSE                            # Project license
├── pyproject.toml                     # Python project configuration & dependencies
│
├── src/publicdata_ca/                 # Main data acquisition package
│   ├── __init__.py
│   ├── acquisition/                   # Data acquisition modules
│   │   ├── __init__.py
│   │   ├── registry.py                # Dataset registry with metadata
│   │   ├── fetch.py                   # Download helpers (HTTP, retry, caching)
│   │   ├── storage.py                 # Path conventions, versioned folders
│   │   ├── validate.py                # Schema checks, types, row counts
│   │   └── transforms.py              # Minimal normalization utilities
│   └── utils/                         # Utility modules
│       ├── __init__.py
│       ├── logging.py                 # Logging configuration
│       ├── dates.py                   # Date/time utilities
│       └── config.py                  # Environment & defaults
│
├── data/                              # Data storage (gitignored)
│   ├── raw/cihi/                      # Raw downloaded data
│   ├── interim/cihi/                  # Intermediate processing steps
│   └── processed/cihi/                # Final processed datasets
│
├── case_studies/hospital_anomalies/   # Hospital anomalies case study
│   ├── README.md                      # Case study documentation
│   ├── cli.py                         # Command-line interface entrypoint
│   │
│   ├── config/
│   │   └── default.yaml               # Dataset IDs, date ranges, model params
│   │
│   ├── notebooks/                     # Jupyter notebooks for exploration
│   │   ├── 01_ingest_and_qc.ipynb     # Data ingestion & quality checks
│   │   ├── 02_baseline_model.ipynb    # Isolation Forest baseline
│   │   └── 03_autoencoder.ipynb       # Optional extension (stub)
│   │
│   ├── src/                           # Importable source code
│   │   ├── __init__.py
│   │   ├── pipeline.py                # Orchestrates all steps
│   │   ├── ingest.py                  # Calls publicdata_ca.acquisition
│   │   ├── qc.py                      # Missingness, outliers, seasonality
│   │   ├── features.py                # Time-series feature engineering
│   │   ├── evaluation.py              # Stability checks, heuristics
│   │   ├── visualize.py               # Plots & anomaly overlays
│   │   ├── io.py                      # Load/save artifacts (parquet/csv)
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── isolation_forest.py    # Train/score/persist, deterministic
│   │       └── autoencoder.py         # Optional (stub implementation)
│   │
│   ├── tests/                         # Test suite
│   │   ├── test_ingest_smoke.py       # Ingestion smoke tests
│   │   ├── test_features.py           # Feature engineering tests
│   │   └── test_isolation_forest.py   # Model tests
│   │
│   └── reports/                       # Generated outputs (gitignored)
│       ├── figures/                   # Exported plots
│       ├── results/                   # CSV anomaly tables
│       └── models/                    # Saved model artifacts
│
├── scripts/                           # Automation scripts
│   ├── run_case_study_hospital_anomalies.sh
│   └── export_report_assets.sh
│
├── .github/workflows/
│   └── tests.yml                      # CI configuration
│
└── docs/case_studies/
    └── hospital_anomalies.md          # Portfolio write-up
```

## Key Features

### 1. Separation of Concerns
- **Data Acquisition** (`src/publicdata_ca/`): Reusable package for fetching Canadian public data
- **Case Study** (`case_studies/hospital_anomalies/`): Specific anomaly detection implementation
- **Documentation** (`docs/`): Portfolio-ready write-ups

### 2. Importable Code
- All logic in Python modules under `src/`
- Notebooks call functions from modules (no notebook-only logic)
- Enables testing, CI, and reusability

### 3. Deterministic & Reproducible
- Fixed random seeds throughout (`random_state=42`)
- Parquet for intermediate data (stable binary format)
- CSV for final results (human-readable)
- All dependencies pinned in `pyproject.toml`

### 4. CLI-First Design
```bash
# Single command to run entire pipeline
python -m case_studies.hospital_anomalies.cli run \
  --config case_studies/hospital_anomalies/config/default.yaml
```

### 5. Test Coverage
- 16 tests across 3 modules
- Covers ingestion, feature engineering, and modeling
- Integrated with GitHub Actions CI

## Usage

### Installation
```bash
# Install package in development mode
pip install -e .
```

### Running the Pipeline
```bash
# Via CLI
python -m case_studies.hospital_anomalies.cli run \
  --config case_studies/hospital_anomalies/config/default.yaml

# Via shell script
./scripts/run_case_study_hospital_anomalies.sh
```

### Running Tests
```bash
pytest case_studies/hospital_anomalies/tests/ -v
```

### Notebooks
```bash
jupyter notebook case_studies/hospital_anomalies/notebooks/
```

## Configuration

The `config/default.yaml` file controls:
- **Datasets**: Which CIHI datasets to fetch
- **Date Range**: Analysis time period
- **Features**: Rolling windows, lags, seasonal features
- **Model Parameters**: Isolation Forest settings (n_estimators, contamination, etc.)
- **Evaluation**: Top-K anomalies, persistence windows
- **Output**: Save locations for figures, results, models

## Outputs

After running the pipeline:

1. **Data**:
   - `data/raw/cihi/*.parquet` - Raw ingested data
   - `data/processed/cihi/features.parquet` - Engineered features

2. **Models**:
   - `case_studies/hospital_anomalies/reports/models/isolation_forest.joblib` - Trained model

3. **Results**:
   - `case_studies/hospital_anomalies/reports/results/anomalies.csv` - Top anomalies with scores

4. **Visualizations**:
   - `case_studies/hospital_anomalies/reports/figures/*.png` - Time-series plots with anomaly overlays

## Development Workflow

1. **Make Changes**: Edit code in `src/` or `case_studies/*/src/`
2. **Run Tests**: `pytest case_studies/hospital_anomalies/tests/ -v`
3. **Test Pipeline**: `python -m case_studies.hospital_anomalies.cli run`
4. **Explore**: Open notebooks to visualize results
5. **Document**: Update `docs/` with findings

## Dependencies

Core dependencies (see `pyproject.toml`):
- **Data**: pandas, numpy, pyarrow
- **ML**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Notebooks**: jupyter, notebook
- **Testing**: pytest
- **Other**: requests, pyyaml

## Design Principles

1. **Minimal Normalization**: Use existing MVP acquisition helpers
2. **No Manual Downloads**: All data fetched programmatically
3. **Deterministic Seeds**: Reproducible results across runs
4. **Evaluation Without Labels**: Persistence, seasonality, stability metrics
5. **CI Integration**: Automated testing on every push

## Future Extensions

- Add more CIHI datasets
- Implement full autoencoder with TensorFlow/PyTorch
- Multi-region analysis with separate models
- Real-time monitoring capabilities
- SHAP explainability for feature importance
