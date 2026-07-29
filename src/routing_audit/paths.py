"""Central data-file locations.

Everything here used to be hardcoded to sandbox-session paths
(`/sessions/.../mnt/...`) in the original scripts. This module resolves
paths relative to the repo root instead, with environment-variable
overrides for anyone who keeps the data elsewhere.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = Path(os.environ.get("ROUTING_AUDIT_DATA_RAW", ROOT / "data" / "raw"))
DATA_PROCESSED = Path(os.environ.get("ROUTING_AUDIT_DATA_PROCESSED", ROOT / "data" / "processed"))
RESULTS = Path(os.environ.get("ROUTING_AUDIT_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("ROUTING_AUDIT_FIGURES", ROOT / "results" / "figures"))

SEATTLE_EUI_CSV = DATA_RAW / "seattle_building_energy_benchmarking.csv"
TRACT_DEMOGRAPHICS_GEOJSON = DATA_PROCESSED / "king_county_tracts_demographics.geojson"

SEATTLE_MODEL_PKL = DATA_PROCESSED / "seattle_model.pkl"
SEATTLE_BUILDINGS_CSV = DATA_PROCESSED / "seattle_buildings.csv"
SEATTLE_PREP_NPZ = DATA_PROCESSED / "seattle_prep.npz"


def ensure_dirs():
    for d in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
