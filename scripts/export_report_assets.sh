#!/bin/bash
# Script to export report assets for portfolio/documentation

set -e

echo "Exporting report assets..."

# Define paths
REPORT_DIR="case_studies/hospital_anomalies/reports"
EXPORT_DIR="docs/case_studies/hospital_anomalies_assets"

# Create export directory
mkdir -p "$EXPORT_DIR"

# Copy figures
if [ -d "$REPORT_DIR/figures" ]; then
    echo "Copying figures..."
    cp -r "$REPORT_DIR/figures" "$EXPORT_DIR/"
fi

# Copy results summaries
if [ -d "$REPORT_DIR/results" ]; then
    echo "Copying results..."
    cp -r "$REPORT_DIR/results" "$EXPORT_DIR/"
fi

# Create a summary README
cat > "$EXPORT_DIR/README.md" << EOF
# Hospital Anomalies Case Study - Assets

This directory contains exported assets from the hospital anomalies case study.

## Contents

- \`figures/\` - Visualization plots showing anomalies and time-series analysis
- \`results/\` - CSV files with anomaly scores and rankings

## Generated

$(date)
EOF

echo "Export complete! Assets saved to: $EXPORT_DIR"
