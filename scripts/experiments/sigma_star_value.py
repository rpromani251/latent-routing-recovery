#!/usr/bin/env python3
"""
Does picking the scale from the dispersion curve actually help? (docs/results_2026-07-28.md
S6c Role 1 -- it doesn't: sigma* is WORSE than a randomly chosen scale, 0.715 vs 0.798
power. This is why audit.py's production pipeline drops scale selection entirely and
naive-scans every scale in the ladder instead.)

The claim under test: the dispersion statistic's job is SCALE SELECTION -- it tells you
which probe scale to point the dip test at. Four ways of choosing where to dip, all on
the same buildings and the same fresh samples:

  sigma*        the scale where the dispersion curve peaks
  best fixed    the single scale that works best averaged over ALL buildings
                (cheating slightly: chosen with hindsight, so it is an upper bound
                 on what any fixed-scale rule could achieve)
  median fixed  the middle scale of the ladder, i.e. a reasonable blind guess
  random        a randomly chosen valid scale per building

Usage:
  python scripts/experiments/sigma_star_value.py \
      --indices-file results/sigma_star_anchors_idx.npy --out results/sigma_star.csv
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np, pandas as pd
import diptest
from scipy import stats
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.routing_audit import paths
from src.routing_audit.location_terms import build_location_terms, M_LAT, M_LON
from src.routing_audit.probes import Probe, respond

SIG_M = np.geomspace(25.0, 1200.0, 12)
M_PROBE = 200
M_DIP = 600
TAU_OBS = 0.010
MIN_VALID = 4


def run_one(k, loc, ras, probe, rng, gated):
    lat0, lon0, hb = loc["lat0"][k], loc["lon0"][k], loc["h_base"][k]
    r = np.zeros(len(SIG_M)); lam = np.zeros(len(SIG_M)); avail = np.ones(len(SIG_M), bool)
    for t, sm in enumerate(SIG_M):
        la, lo = probe.draw(lat0, lon0, sm, M_PROBE, rng)
        if la is None:
            avail[t] = False; continue
        y = respond(la, lo, hb, loc, ras, rng, gated, tau_obs=TAU_OBS)
        lam[t] = np.var(y, ddof=1) / TAU_OBS**2
        r[t] = np.sqrt(max(lam[t] - 1.0, 0.0)) / sm
    valid = avail & (stats.chi2.sf((M_PROBE - 1) * lam, df=M_PROBE - 1) < 0.05)
    if valid.sum() < MIN_VALID:
        return None
    # fresh dip sample at EVERY valid scale, so all selection rules are scored
    # against identical data and differ only in which scale they point at
    dp = np.full(len(SIG_M), np.nan)
    for t, sm in enumerate(SIG_M):
        if not valid[t]:
            continue
        la, lo = probe.draw(lat0, lon0, sm, M_DIP, rng)
        if la is None:
            continue
        y = respond(la, lo, hb, loc, ras, rng, gated, tau_obs=TAU_OBS)
        dp[t] = diptest.diptest(np.ascontiguousarray(y))[1]
    t_star = int(np.argmax(np.where(valid, r, -np.inf)))
    vt = np.where(valid & np.isfinite(dp))[0]
    return {"t_star": t_star, "p_star": dp[t_star],
            "dp": dp, "t_rand": int(rng.choice(vt)) if vt.size else -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices-file", type=Path,
                     default=paths.RESULTS / "sigma_star_anchors_idx.npy")
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "sigma_star.csv")
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

    sel = np.load(args.indices_file)
    rng = np.random.default_rng(4242)
    rows = []
    for k in sel:
        k = int(k)
        rec = {"idx": k}
        for gated, pre in ((True, "g"), (False, "h")):
            o = run_one(k, loc, ras, probe, rng, gated)
            if o is None:
                continue
            rec[f"{pre}_t_star"] = o["t_star"]
            rec[f"{pre}_p_star"] = o["p_star"]
            rec[f"{pre}_t_rand"] = o["t_rand"]
            rec[f"{pre}_p_rand"] = o["dp"][o["t_rand"]] if o["t_rand"] >= 0 else np.nan
            for t in range(len(SIG_M)):
                rec[f"{pre}_p_{t}"] = o["dp"][t]
        rows.append(rec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}  rows={len(rows)}")


if __name__ == "__main__":
    main()
