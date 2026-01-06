#!/bin/bash
# Script to run the hospital anomalies case study

set -e

echo "=========================================="
echo "Hospital Anomalies Case Study Pipeline"
echo "=========================================="

# Default config path
CONFIG_PATH="${1:-case_studies/hospital_anomalies/config/default.yaml}"

echo ""
echo "Using configuration: $CONFIG_PATH"
echo ""

# Run the pipeline
python -m case_studies.hospital_anomalies.cli run --config "$CONFIG_PATH"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Figures: case_studies/hospital_anomalies/reports/figures/"
echo "  - Results: case_studies/hospital_anomalies/reports/results/"
echo "  - Models: case_studies/hospital_anomalies/reports/models/"
