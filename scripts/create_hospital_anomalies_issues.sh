#!/usr/bin/env bash
set -euo pipefail

# Creates GitHub issues for the Hospital Anomalies case study.
# Requires: GitHub CLI (gh) and authentication (gh auth login)

REPO="${REPO:-}"  # optional: "owner/repo" to create issues in a different repo

create_issue () {
  local title="$1"
  local body="$2"

  if [[ -n "${REPO}" ]]; then
    gh issue create --repo "${REPO}" --title "${title}" --body "${body}"
  else
    gh issue create --title "${title}" --body "${body}"
  fi
}

create_issue \
  "Project scaffolding: hospital_anomalies structure and configs" \
  "$(cat <<'EOF'
Goal
- Add a clean, self-contained project structure for the Hospital Anomalies case study.

Deliverables
- Directory structure:
  - case_studies/hospital_anomalies/{src,tests,notebooks,reports,config}
- default.yaml with dataset IDs/date ranges/model params placeholders
- Minimal README with how to run ingest/model/viz

Acceptance criteria
- Repo contains the case_studies/hospital_anomalies directory
- Running `python -m case_studies.hospital_anomalies.cli --help` works (stub is fine)
EOF
)" \

create_issue \
  "Ingest CIHI utilization data via publicdata_ca MVP" \
  "$(cat <<'EOF'
Goal
- Use publicdata_ca (MVP) to acquire CIHI utilization summaries used in this case study.

Tasks
- Implement case_studies/hospital_anomalies/src/ingest.py
- Store raw/interim/processed outputs under data/ (gitignored)
- Add basic dataset metadata (source name, dataset id, retrieval date) into a sidecar JSON

Acceptance criteria
- `python -m case_studies.hospital_anomalies.cli ingest --config ...` downloads data reproducibly
- Artifacts saved to data/raw/cihi and data/processed/cihi
- Ingested table has stable column names + date column parsed
EOF
)" \

create_issue \
  "Data validation + QC checks (missingness, bounds, basic sanity)" \
  "$(cat <<'EOF'
Goal
- Implement lightweight QC so the pipeline fails fast on bad inputs.

Tasks
- case_studies/hospital_anomalies/src/qc.py
- Checks:
  - required columns exist
  - date column monotonic after sorting
  - missingness summary + fail threshold (configurable)
  - basic numeric bounds (no negative admissions, occupancy in [0,1] if rate)
- Persist QC report JSON

Acceptance criteria
- QC runs as part of CLI pipeline and produces a report file
- QC fails with a clear error message if required columns are missing
EOF
)" \

create_issue \
  "Feature engineering for anomaly detection (rolling stats, lags)" \
  "$(cat <<'EOF'
Goal
- Create time-series features that work without labels.

Tasks
- case_studies/hospital_anomalies/src/features.py
- Features (configurable):
  - rolling mean/std (e.g., 3, 6, 12)
  - percent change / difference
  - lag features (1, 2, 3)
  - optional seasonal index (month-of-year as integer)
- Ensure features handle missing values deterministically

Acceptance criteria
- `build_features(df)` returns a feature matrix with no NaNs (after defined handling)
- Unit tests cover at least one rolling + one lag feature
EOF
)" \

create_issue \
  "Isolation Forest baseline model: train, score, persist artifacts" \
  "$(cat <<'EOF'
Goal
- Implement a fast baseline anomaly detector with reproducible scoring.

Tasks
- case_studies/hospital_anomalies/src/models/isolation_forest.py
- Add deterministic seed support and config-driven hyperparameters
- Output:
  - per-row anomaly score
  - boolean flag for top-k or threshold-based anomalies
  - model artifact persisted (joblib)

Acceptance criteria
- CLI command `model --method isolation_forest` runs end-to-end
- Produces a results CSV (date, group keys, score, is_anomaly)
- Re-running with same config produces identical scores
EOF
)" \

create_issue \
  "Visualization: time series plots with anomaly overlays + top anomaly table" \
  "$(cat <<'EOF'
Goal
- Create clear visuals that communicate anomalies quickly.

Tasks
- case_studies/hospital_anomalies/src/visualize.py
- Plot:
  - metric over time with anomaly markers
  - optionally score over time
- Export to reports/figures
- Generate a ranked table (top N anomaly windows)

Acceptance criteria
- `visualize` step exports at least 2 figures
- A CSV exists with top anomaly windows sorted by severity
EOF
)" \

create_issue \
  "Evaluation heuristics: stability, persistence, and seasonality guardrails" \
  "$(cat <<'EOF'
Goal
- Provide label-free evaluation signals to avoid obvious false positives.

Tasks
- case_studies/hospital_anomalies/src/evaluation.py
- Metrics/heuristics:
  - anomaly persistence (consecutive anomalies)
  - stability across small hyperparameter changes (optional)
  - seasonality check (compare to same month prior years, if data supports)
- Save evaluation report JSON

Acceptance criteria
- Evaluation report generated and referenced in case study README
- Includes at least persistence and seasonality checks (when feasible)
EOF
)" \

create_issue \
  "Autoencoder model: dataset prep and baseline architecture" \
  "$(cat <<'EOF'
Goal
- Add an autoencoder-based anomaly detector as an extension to the baseline.

Tasks
- case_studies/hospital_anomalies/src/models/autoencoder.py
- Dataset prep:
  - train/val split by time (no leakage)
  - scaling (fit on train only)
- Model:
  - simple feed-forward AE (MLP)
  - reconstruction error as anomaly score
- Save model + scaler artifacts

Acceptance criteria
- CLI supports `model --method autoencoder`
- Produces reconstruction error scores and anomaly flags
- Time-based split prevents training on future data
EOF
)" \

create_issue \
  "Autoencoder tuning: thresholding, calibration, and comparison to baseline" \
  "$(cat <<'EOF'
Goal
- Make AE results usable and comparable to Isolation Forest.

Tasks
- Implement threshold strategies:
  - percentile on train/val reconstruction error
  - robust z-score threshold (optional)
- Compare overlap and disagreement with Isolation Forest
- Add a small report: strengths, failure modes, when to prefer each

Acceptance criteria
- A comparison table exists (overlap %, top anomalies per method)
- README includes a short discussion of differences
EOF
)" \

create_issue \
  "Pipeline CLI: ingest -> qc -> features -> model -> evaluate -> visualize" \
  "$(cat <<'EOF'
Goal
- Provide a single CLI entrypoint for reproducible runs.

Tasks
- case_studies/hospital_anomalies/cli.py and src/pipeline.py
- Commands:
  - ingest
  - qc
  - features
  - model (method arg)
  - evaluate
  - visualize
  - run (all steps)
- Config-driven, no notebook-only logic

Acceptance criteria
- `python -m case_studies.hospital_anomalies.cli run --config ...` completes
- Each step can be run independently
EOF
)" \

create_issue \
  "Tests: smoke tests for ingest/qc/features and model determinism" \
  "$(cat <<'EOF'
Goal
- Add enough tests to catch breakage and prove determinism.

Tasks
- tests for:
  - feature output shape + no NaNs after handling
  - isolation forest determinism (same seed -> same output)
  - AE train step runs on small synthetic data (fast test)
  - qc fails on missing required columns

Acceptance criteria
- `pytest` passes locally and in CI
- Tests complete in reasonable time (no large training runs)
EOF
)" \

create_issue \
  "Documentation: portfolio write-up (methods, results, limitations)" \
  "$(cat <<'EOF'
Goal
- Produce a polished case study document suitable for GitHub + LinkedIn article.

Content
- Motivation and operational framing
- Data source and limitations (public summaries, aggregation)
- Methods:
  - Isolation Forest baseline
  - Autoencoder extension
- Results: key plots + top anomalies table
- Evaluation heuristics and caveats

Acceptance criteria
- docs/case_studies/hospital_anomalies.md exists
- case_studies/hospital_anomalies/README.md points to it
EOF
)" \

echo "Done. Issues created."
