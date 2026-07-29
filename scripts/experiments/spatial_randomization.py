#!/usr/bin/env python3
"""
Radius sweep for the spatial-randomization alignment claim (docs/results_2026-07-28.md
S7c). The test statistic must match the method's reach: flags from the naive multi-scale
scan sit further from boundaries because the method detects from further away, so a
"within 100 m" statistic penalizes the extra reach. 800 m (the operating point in
scripts/make_figures.py) is taken from the independently measured power-vs-distance
curve, not chosen for significance -- report the full sweep so that's auditable.

Usage:
  python scripts/experiments/spatial_randomization.py \
      --audit-csv results/seattle_audit.csv --config configs/spatial_randomization.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np, pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detect_recover_interpret import paths
from src.detect_recover_interpret import spatial_randomization as SR
from src.detect_recover_interpret.model import TAU_B

DELTA_FILTER = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", type=Path, default=paths.RESULTS / "seattle_audit.csv")
    ap.add_argument("--config", type=Path,
                     default=Path(__file__).resolve().parents[2]
                     / "configs" / "spatial_randomization.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = pd.read_csv(args.audit_csv)
    flag = d["g_flag"].fillna(False).to_numpy(bool)
    strict = flag & (d["g_delta_med"] > DELTA_FILTER).fillna(False).to_numpy()

    tid, bvals, lats, lons = SR.tract_index_raster(
        paths.TRACT_DEMOGRAPHICS_GEOJSON, tuple(cfg["lat_range"]), tuple(cfg["lon_range"]))
    lat = d["lat"].to_numpy(); lon = d["lon"].to_numpy()

    print(f"{'radius':>8}{'observed':>12}{'null mean':>12}{'p':>10}")
    for radius_m in cfg["radii_m"]:
        rng = np.random.default_rng(cfg["seed"])
        obs, null, p = SR.alignment_test(strict, lat, lon, tid, bvals, lats, lons,
                                         TAU_B, radius_m=float(radius_m),
                                         n_perm=cfg["n_perm"], rng=rng)
        print(f"{radius_m:>7}m{obs:>12.3f}{null.mean():>12.3f}{p:>10.4f}")


if __name__ == "__main__":
    main()
