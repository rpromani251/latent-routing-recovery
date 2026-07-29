#!/usr/bin/env python3
"""
Select the "failing band" anchors from a main-audit run: pi_true in [0.05, 0.10), where
detection power under the headline protocol is weakest (docs/results_2026-07-28.md S6e).
Writes an .npy index file for --indices-file to scripts/run_seattle_audit.py, so the
query-reallocation experiment (configs/query_reallocation.yaml) can be scored on exactly
this band.

Usage:
  python scripts/experiments/select_weak_band.py \
      --audit-csv results/seattle_audit.csv --out results/weak_band_idx.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detect_recover_interpret import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", type=Path, default=paths.RESULTS / "seattle_audit.csv")
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "weak_band_idx.npy")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.10)
    args = ap.parse_args()

    d = pd.read_csv(args.audit_csv)
    sel = (d["g_pi_true_max"] >= args.lo) & (d["g_pi_true_max"] < args.hi)
    idx = d.loc[sel, "idx"].to_numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, idx)
    print(f"selected {len(idx)} anchors with pi_true in [{args.lo}, {args.hi})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
