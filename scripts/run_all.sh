#!/usr/bin/env bash
set -e  # exit on any error
set -u  # treat unset variables as error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Routing Audit: Full Reproducibility Run ==="
echo "Working dir: $REPO_ROOT"
echo ""
echo "Regenerates every current figure from the raw Seattle Building Energy Benchmarking"
echo "CSV. Requires data/raw/seattle_building_energy_benchmarking.csv (see"
echo "scripts/download_seattle_data.py) and data/processed/king_county_tracts_demographics.geojson"
echo "(shared with the geospatial-xai-attacks repo -- see docs/data_dependencies.md)."
echo ""

if [ ! -f "data/raw/seattle_building_energy_benchmarking.csv" ]; then
    echo "Missing data/raw/seattle_building_energy_benchmarking.csv -- see" \
         "scripts/download_seattle_data.py or docs/data_dependencies.md" >&2
    exit 1
fi
if [ ! -f "data/processed/king_county_tracts_demographics.geojson" ]; then
    echo "Missing data/processed/king_county_tracts_demographics.geojson -- see" \
         "docs/data_dependencies.md" >&2
    exit 1
fi

echo "Step 1: Fit the honest model + tract-demographics raster"
python3 scripts/build_model.py --config configs/model.yaml

echo ""
echo "Step 2: Run the production audit protocol (naive multi-scale dip scan, on-manifold probes)"
python3 scripts/run_seattle_audit.py --config configs/main_audit.yaml \
    --out results/seattle_audit.csv

echo ""
echo "Step 3: Honest-model smoothness check (supporting figure)"
python3 scripts/run_lengthscale_check.py

echo ""
echo "Step 4: Dip-discriminator toy validation (supporting figure, self-contained)"
python3 scripts/run_dip_discriminator_toy.py

echo ""
echo "Step 5: Current poster figures + conformal coverage"
python3 scripts/make_figures.py

echo ""
echo "=== Done. See results/figures/ for the current figure set. ==="
echo "Supporting experiments (query reallocation, sigma* value, spatial-randomization"
echo "radius sweep) are not run by default -- see scripts/experiments/ and README.md."
