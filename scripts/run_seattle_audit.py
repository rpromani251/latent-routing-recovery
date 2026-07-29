#!/usr/bin/env python3
"""
Run the current production audit protocol (naive multi-scale dip scan + K=2 recovery,
on-manifold probes) over all Seattle buildings. See src/routing_audit/audit.py for the
per-building logic and docs/results_2026-07-28.md S6d for the headline numbers this
reproduces.

Usage:
  python scripts/run_seattle_audit.py --config configs/main_audit.yaml \
      --out results/seattle_audit.csv
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.routing_audit import paths
from src.routing_audit.audit import AuditConfig, run_audit_batch
from src.routing_audit.location_terms import build_location_terms, M_LAT, M_LON
from src.routing_audit.probes import Probe


def load_config(path):
    cfg = yaml.safe_load(open(path))
    a = cfg["audit"]
    return AuditConfig(
        scales_m=np.geomspace(a["scale_min_m"], a["scale_max_m"], a["n_scales"]),
        m_dip=a["m_dip"],
        m_rec=a["m_rec"],
        tau_obs=a.get("tau_obs", 0.010),
        tau_b=a.get("tau_b", 0.25),
        penalty=a.get("penalty", 0.30),
        alpha=a.get("alpha", 0.05),
        min_component_mass=a.get("min_component_mass", 20),
    ), cfg.get("seed", 777)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--indices-file", type=Path, default=None,
                     help="Optional .npy of building indices to restrict to "
                          "(e.g. results/weak_band_idx.npy)")
    ap.add_argument("--i0", type=int, default=0)
    ap.add_argument("--i1", type=int, default=None)
    args = ap.parse_args()

    cfg, seed = load_config(args.config)

    df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
    model, num = pickle.load(open(paths.SEATTLE_MODEL_PKL, "rb"))
    loc = build_location_terms(model, num, df)
    loc["lat0"] = df["Latitude"].to_numpy()
    loc["lon0"] = df["Longitude"].to_numpy()

    prep = np.load(paths.SEATTLE_PREP_NPZ)
    ras = {"B": prep["B"], "lats": prep["lats"], "lons": prep["lons"]}

    lat_ref, lon_ref = loc["lat0"].min(), loc["lon0"].min()
    XY = np.column_stack([(loc["lon0"] - lon_ref) * M_LON, (loc["lat0"] - lat_ref) * M_LAT])
    probe = Probe(cKDTree(XY), np.column_stack([loc["lat0"], loc["lon0"]]),
                  "onmanifold", M_LAT, M_LON)
    probe.lat0_ref, probe.lon0_ref = lat_ref, lon_ref

    if args.indices_file is not None:
        indices = np.load(args.indices_file)
    else:
        i1 = args.i1 if args.i1 is not None else len(df)
        indices = np.arange(args.i0, min(i1, len(df)))

    rng = np.random.default_rng(seed)
    rows = run_audit_batch(indices, loc, ras, probe, rng, cfg)
    for row, k in zip(rows, indices):
        row["b"] = float(prep["b_bldg"][k])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}  rows={len(rows)}")


if __name__ == "__main__":
    main()
