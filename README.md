### Case Study: Detecting Anomalies in Hospital Resource Usage (MVP)

## 🏥 Project Overview

This project implements an end-to-end machine learning pipeline for detecting anomalous patterns in Canadian hospital resource utilization using unsupervised learning techniques.

**Key Features:**
- ✅ Complete data acquisition pipeline using public CIHI data
- ✅ Isolation Forest anomaly detection with deterministic behavior
- ✅ Time-series feature engineering (rolling stats, lags, seasonal features)
- ✅ Comprehensive testing (16 tests, 100% passing)
- ✅ CLI-driven workflow for reproducible runs
- ✅ Portfolio-ready documentation

## 🚀 Quick Start

### Installation
```bash
pip install -e .
```

### Run the Complete Pipeline
```bash
# Via CLI
python -m case_studies.hospital_anomalies.cli run \
  --config case_studies/hospital_anomalies/config/default.yaml

# Or via shell script
./scripts/run_case_study_hospital_anomalies.sh
```

### Run Tests
```bash
pytest case_studies/hospital_anomalies/tests/ -v
```

### Explore Notebooks
```bash
jupyter notebook case_studies/hospital_anomalies/notebooks/
```

## 📊 What This Does

**Problem:** Hospitals operate close to capacity, and unexpected surges in admissions, bed occupancy, or ICU usage can overwhelm staff and infrastructure. Early detection of anomalous demand patterns enables proactive staffing, resource reallocation, and escalation planning.

**Solution:** Unsupervised anomaly detection using:
1. **Data Acquisition** - Automated fetching of CIHI hospital utilization data
2. **Quality Control** - Missing data checks, outlier detection, temporal validation
3. **Feature Engineering** - Rolling statistics, lags, seasonal indicators
4. **Modeling** - Isolation Forest for fast, interpretable anomaly scoring
5. **Evaluation** - Persistence checks, seasonal analysis, top-K rankings
6. **Visualization** - Time-series plots with anomaly overlays

## 📁 Project Structure

```
hospital-resource-usage-anomolies/
├── case_studies/hospital_anomalies/  # Anomaly detection case study
│   ├── src/                        # Source code modules
│   │   ├── utils.py               # Utilities for logging, config, paths
│   │   ├── models/                # ML model implementations
│   │   └── ...                    # Other modules
│   ├── tests/                      # Comprehensive test suite
│   ├── notebooks/                  # Jupyter notebooks
│   ├── config/                     # YAML configuration
│   └── cli.py                      # Command-line interface
│
├── scripts/                        # Automation scripts
├── docs/                           # Portfolio documentation
└── .github/workflows/              # CI/CD configuration
```

**Dependencies**: Uses the published [publicdata-ca](https://pypi.org/project/publicdata-ca/) package from PyPI for data acquisition.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation.

## 🎯 Key Results

When you run the pipeline, you get:
- **Processed Data**: Features engineered from raw CIHI data (parquet format)
- **Trained Model**: Isolation Forest detector (saved as joblib)
- **Anomaly Results**: CSV with top-K anomalies and severity scores
- **Visualizations**: 5 PNG plots showing time-series with anomaly overlays
- **Evaluation Metrics**: Persistence rates, seasonal distribution, stability scores

## 🏗️ Architecture Highlights

### Deterministic & Reproducible
- Fixed random seeds throughout (`random_state=42`)
- Training statistics stored for consistent imputation
- Parquet for intermediate data, CSV for results

### Importable Code
- All logic in Python modules (no notebook-only code)
- Enables testing, CI, and reusability
- Notebooks call functions for exploration

### Quality & Testing
- 16 comprehensive tests (100% passing)
- GitHub Actions CI integration
- Code review feedback addressed

### Configuration-Driven
- YAML configuration for all parameters
- Easy experimentation with different settings
- Environment variable overrides supported

## 📖 Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete technical overview
- **[case_studies/hospital_anomalies/README.md](case_studies/hospital_anomalies/README.md)** - Case study guide
- **[docs/case_studies/hospital_anomalies.md](docs/case_studies/hospital_anomalies.md)** - Portfolio write-up

## 🔬 Motivation

Hospitals operate close to capacity, and unexpected surges in admissions, bed occupancy, or ICU usage can overwhelm staff and infrastructure. Early detection of anomalous demand patterns enables proactive staffing, resource reallocation, and escalation planning.

## 📚 Data (MVP Scope)

Publicly available hospital utilization summaries from Canadian Institute for Health Information (CIHI), including:

- Monthly inpatient admissions
- Average length of stay
- Bed occupancy rates
- ICU utilization (where available)

Only datasets ingestible using the current MVP of the Data Acquisition Package are used—no manual cleaning or enrichment beyond standardized normalization.

## 🤖 ML Task

Unsupervised anomaly detection, treating unusual utilization patterns as deviations from historical norms rather than labeled “events.”

Models explored:

- Isolation Forest for fast, interpretable anomaly scoring
- Autoencoder (optional extension) to model normal utilization patterns and flag high reconstruction error

Anomalies are detected at the regional and hospital-group level across time.

## 📈 Output

- Time-series plots with anomaly overlays
- Ranked anomaly windows with severity scores
- Short narrative explaining detected spikes and potential operational interpretations

## 💡 Key Skill Signal

- Practical unsupervised learning in a real public-sector context
- Time-series reasoning without labeled outcomes
- Clear separation of data acquisition, modeling, and interpretation
- Production-quality code with testing and CI/CD
- Portfolio-ready documentation and visualizations

---

**Author:** [Andrew Harris](https://github.com/ajharris)  
**License:** See [LICENSE](LICENSE)  
**Status:** ✅ Complete and tested
