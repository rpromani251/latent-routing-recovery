#!/usr/bin/env python3
"""
Select an anchor subset for the sigma*-value experiment (docs/results_2026-07-28.md S6c
Role 1, n=906 on-manifold anchors). The original anchor list was chosen ad hoc and not
scripted; this reconstructs a documented equivalent: a random sample of buildings, sized
to match. If you have a prior audit run and want anchors near boundaries specifically,
pass --near-boundary-csv (a CSV with a `d_bound` column, e.g. legacy/seattle_audit_v1.py
output) and --max-dist-m.

Usage:
  python scripts/experiments/select_sigma_star_anchors.py --n 906 --out results/sigma_star_anchors_idx.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detect_recover_interpret import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=906)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "sigma_star_anchors_idx.npy")
    ap.add_argument("--near-boundary-csv", type=Path, default=None)
    ap.add_argument("--max-dist-m", type=float, default=400.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.near_boundary_csv is not None:
        d = pd.read_csv(args.near_boundary_csv)
        pool = d.loc[d["d_bound"] <= args.max_dist_m, "idx"].to_numpy()
    else:
        df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
        pool = np.arange(len(df))

    n = min(args.n, len(pool))
    idx = rng.choice(pool, size=n, replace=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, idx)
    print(f"selected {n} anchors from a pool of {len(pool)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
