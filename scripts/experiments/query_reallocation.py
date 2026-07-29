#!/usr/bin/env python3
"""
Query-allocation experiment (docs/results_2026-07-28.md S6e): on the weak-signal band
(pi_true in [0.05, 0.10), see select_weak_band.py), compare the headline protocol
(6 scales x m=500) against a reallocated one (3 scales x m=1000, same total query
budget) on power and honest-model false-positive rate.

Usage:
  python scripts/experiments/query_reallocation.py \
      --indices-file results/weak_band_idx.npy \
      --main-config configs/main_audit.yaml \
      --realloc-config configs/query_reallocation.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.routing_audit import paths
from src.routing_audit.audit import run_audit_batch
from src.routing_audit.location_terms import build_location_terms, M_LAT, M_LON
from src.routing_audit.probes import Probe
from scripts.run_seattle_audit import load_config  # noqa: E402  (sys.path set above)


def run(config_path, indices, loc, ras, probe):
    cfg, seed = load_config(config_path)
    rng = np.random.default_rng(seed)
    rows = run_audit_batch(indices, loc, ras, probe, rng, cfg)
    return pd.DataFrame(rows), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices-file", type=Path, default=paths.RESULTS / "weak_band_idx.npy")
    ap.add_argument("--main-config", default=str(Path(__file__).resolve().parents[2]
                                                 / "configs" / "main_audit.yaml"))
    ap.add_argument("--realloc-config", default=str(Path(__file__).resolve().parents[2]
                                                     / "configs" / "query_reallocation.yaml"))
    args = ap.parse_args()

    df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
    model, num = pickle.load(open(paths.SEATTLE_MODEL_PKL, "rb"))
    loc = build_location_terms(model, num, df)
    loc["lat0"] = df["Latitude"].to_numpy(); loc["lon0"] = df["Longitude"].to_numpy()

    prep = np.load(paths.SEATTLE_PREP_NPZ)
    ras = {"B": prep["B"], "lats": prep["lats"], "lons": prep["lons"]}

    lat_ref, lon_ref = loc["lat0"].min(), loc["lon0"].min()
    XY = np.column_stack([(loc["lon0"] - lon_ref) * M_LON, (loc["lat0"] - lat_ref) * M_LAT])
    probe = Probe(cKDTree(XY), np.column_stack([loc["lat0"], loc["lon0"]]),
                  "onmanifold", M_LAT, M_LON)
    probe.lat0_ref, probe.lon0_ref = lat_ref, lon_ref

    indices = np.load(args.indices_file)
    print(f"scoring on {len(indices)} weak-band anchors\n")

    hdr = f"{'protocol':<28}{'queries/anchor':>16}{'power':>10}{'honest FP':>12}"
    print(hdr); print("-" * len(hdr))
    for label, path in (("6 scales x m=500", args.main_config),
                        ("3 scales x m=1000", args.realloc_config)):
        d, cfg = run(path, indices, loc, ras, probe)
        queries = int(len(cfg.scales_m) * (cfg.m_dip + cfg.m_rec))
        power = float(d["g_flag"].mean())
        honest_fp = float(d["h_flag"].mean())
        print(f"{label:<28}{queries:>16}{power:>10.3f}{honest_fp:>12.4f}")


if __name__ == "__main__":
    main()
