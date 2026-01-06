# Hospital Resource Usage Anomalies - Case Study

## Executive Summary

This case study demonstrates practical application of unsupervised machine learning to detect anomalous patterns in Canadian hospital resource utilization. Using publicly available data from the Canadian Institute for Health Information (CIHI), we identify unusual spikes in admissions, bed occupancy, and ICU utilization that may indicate operational strain or emerging health crises.

## Business Context

### Problem Statement

Hospitals operate near capacity under normal conditions. Unexpected surges in demand can:
- Overwhelm clinical staff and infrastructure
- Compromise patient care quality
- Require emergency resource reallocation
- Indicate emerging public health threats

Early detection of anomalous utilization patterns enables proactive responses including staffing adjustments, capacity planning, and escalation protocols.

### Approach

**Unsupervised Anomaly Detection** - We treat this as an unsupervised learning problem because:
1. Labeled "crisis events" are sparse and inconsistently defined
2. Normal variation is region and season-dependent
3. The goal is to flag deviations from local historical patterns, not classify known event types

## Data

### Sources

All data acquired through the `publicdata_ca` package using standardized ingestion:
- **CIHI Hospital Admissions** - Monthly inpatient admission counts
- **CIHI Bed Occupancy** - Daily/weekly bed occupancy rates
- **CIHI ICU Utilization** - ICU bed usage and capacity metrics

Date range: 2019-2023 (5 years including pre/post pandemic patterns)

### Data Quality

Quality control includes:
- Missing data checks (threshold: <30% missing per column)
- Outlier detection (3-sigma threshold)
- Temporal coverage validation
- Seasonal pattern verification

## Methodology

### Feature Engineering

Time-series features engineered for each metric:
- **Rolling statistics** - 7, 14, and 30-day windows (mean, std, min, max)
- **Lag features** - 1, 7, and 30-day lags
- **Seasonal indicators** - Month, quarter, day-of-week cyclical encoding
- **Difference features** - First-order differences and percent changes

### Models

#### 1. Isolation Forest (Primary)

**Rationale:** Fast, interpretable, works well with multivariate time-series features

**Configuration:**
- 100 estimators
- Contamination rate: 5% (expected anomaly proportion)
- Deterministic seed: 42 (reproducibility)

**Advantages:**
- No assumptions about distribution
- Naturally handles multiple features
- Provides anomaly scores for ranking

#### 2. Autoencoder (Optional Extension)

**Rationale:** Can learn complex normal patterns through reconstruction

**Status:** Stub implementation - can be extended with TensorFlow/PyTorch

### Evaluation

Since we lack labeled anomalies, evaluation focuses on:

1. **Persistence Check** - Do anomalies cluster in time (indicating real events)?
2. **Seasonal Sanity** - Are anomalies evenly distributed or concentrated in expected high-stress periods?
3. **Stability Metrics** - Is the model consistent across different time windows?
4. **Top-K Review** - Do the highest-scoring anomalies correspond to known events?

## Results

### Key Findings

_(In a real implementation, this section would include):_

- Total anomalies detected: X events
- Persistent anomalies (>3 days): Y events
- Seasonal distribution: Z% in winter months
- Top-10 anomaly windows with dates and severity scores

### Visualizations

Generated outputs include:
- Time-series plots with anomaly overlays
- Anomaly score distributions
- Seasonal anomaly patterns
- Regional comparisons

### Operational Insights

_(Example insights that would be included):_

- Winter surge patterns: Elevated anomalies in Q1 consistently across years
- Regional variations: Urban centers show more volatile patterns
- ICU strain indicators: High reconstruction error correlates with reported capacity issues

## Technical Implementation

### Code Organization

```
case_studies/hospital_anomalies/
├── src/              # All logic in importable modules
│   ├── pipeline.py   # Orchestration
│   ├── models/       # Isolation Forest + Autoencoder
│   └── ...
├── notebooks/        # Interactive exploration (calls src/ functions)
├── tests/            # pytest suite
└── cli.py            # Command-line interface
```

### Running the Analysis

**Command Line:**
```bash
python -m case_studies.hospital_anomalies.cli run \
  --config case_studies/hospital_anomalies/config/default.yaml
```

**Or via script:**
```bash
./scripts/run_case_study_hospital_anomalies.sh
```

**Notebooks:**
1. `01_ingest_and_qc.ipynb` - Data ingestion and validation
2. `02_baseline_model.ipynb` - Isolation Forest with visualizations
3. `03_autoencoder.ipynb` - Optional deep learning extension

### Reproducibility

- ✅ Deterministic random seeds throughout
- ✅ All dependencies pinned in `pyproject.toml`
- ✅ Automated tests in CI (GitHub Actions)
- ✅ Parquet for intermediate data (stable format)

## Skills Demonstrated

### Data Engineering
- Automated data acquisition pipeline
- Robust quality control checks
- Efficient storage (Parquet for large datasets, CSV for summaries)

### Machine Learning
- Unsupervised anomaly detection
- Feature engineering for time-series
- Model persistence and reproducibility

### Software Engineering
- Modular, testable code architecture
- CLI tooling for reproducible runs
- CI/CD integration
- Clear separation of concerns (acquisition, modeling, evaluation)

### Domain Expertise
- Public health data context
- Seasonality and temporal patterns
- Practical evaluation without labeled data

## Future Extensions

1. **Multi-dataset fusion** - Combine admissions, occupancy, and ICU into unified features
2. **Regional modeling** - Separate models per region to capture local patterns
3. **Real-time monitoring** - Adapt for streaming data and online updates
4. **Deep learning** - Complete autoencoder implementation with TensorFlow
5. **Explainability** - SHAP values to interpret feature contributions to anomalies

## References

- Canadian Institute for Health Information (CIHI): https://www.cihi.ca/
- Isolation Forest: Liu et al., "Isolation Forest" (2008)
- Anomaly Detection Survey: Chandola et al., "Anomaly Detection: A Survey" (2009)

---

**Code:** [GitHub Repository](https://github.com/ajharris/hospital-resource-usage-anomolies)

**Contact:** Available for discussion of methodology, results, and extensions
