# Hospital Resource Usage Anomalies - Case Study

## Overview

This case study demonstrates unsupervised anomaly detection on hospital resource usage data from the Canadian Institute for Health Information (CIHI). The goal is to identify unusual patterns in hospital utilization that may indicate resource strain or operational challenges.

## Motivation

Hospitals operate close to capacity, and unexpected surges in admissions, bed occupancy, or ICU usage can overwhelm staff and infrastructure. Early detection of anomalous demand patterns enables proactive staffing, resource reallocation, and escalation planning.

## Data Sources

This case study uses publicly available hospital utilization data from CIHI:
- Monthly inpatient admissions
- Average length of stay
- Bed occupancy rates
- ICU utilization (where available)

All data is acquired using the `publicdata_ca.acquisition` package - no manual downloads or custom processing required.

## Models

Two unsupervised anomaly detection approaches are implemented:

1. **Isolation Forest** - Fast, interpretable baseline for anomaly scoring
2. **Autoencoder** (optional) - Neural network approach modeling normal patterns

## Running the Case Study

### Prerequisites

Install dependencies from the project root:
```bash
pip install -e .
```

### Command Line Interface

Run the complete pipeline:
```bash
python -m case_studies.hospital_anomalies.cli run --config case_studies/hospital_anomalies/config/default.yaml
```

### Jupyter Notebooks

Interactive exploration and analysis:
1. `01_ingest_and_qc.ipynb` - Data ingestion and quality checks
2. `02_baseline_model.ipynb` - Isolation Forest model and visualizations
3. `03_autoencoder.ipynb` - Optional autoencoder extension

## Output

The pipeline generates:
- Time-series plots with anomaly overlays (`reports/figures/`)
- Ranked anomaly windows with severity scores (`reports/results/anomalies.csv`)
- Model artifacts and intermediate data (`data/processed/cihi/`)

## Evaluation

Since this is unsupervised learning without labeled anomalies, evaluation focuses on:
- Anomaly persistence (stability across time windows)
- Seasonal sanity checks (expected patterns vs. detected anomalies)
- Top-k anomaly tables for manual review

## Key Features

- ✅ Deterministic random seeds for reproducibility
- ✅ All logic in importable modules (notebooks call functions)
- ✅ Parquet for intermediate data, CSV for final results
- ✅ CI integration with automated tests
