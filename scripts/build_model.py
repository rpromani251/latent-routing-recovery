#!/usr/bin/env python3
"""
Fit the honest EUI model and the tract Black/Latino-share raster, and cache the
per-building working set. This is the step every audit script assumes has already run
(they all `pickle.load(seattle_model.pkl)` / `np.load(seattle_prep.npz)`), pulled out
into its own script here since none of the original 21 scripts did it explicitly -- it
was run ad hoc.

Writes, under data/processed/:
  seattle_buildings.csv   filtered building table (see model.load_buildings)
  seattle_model.pkl       (sklearn pipeline, numeric_cols) honest model
  seattle_prep.npz        B/lats/lons raster + b_bldg (b at each building's own location)
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.routing_audit import model as m
from src.routing_audit import paths


def main(config_path):
    cfg = yaml.safe_load(open(config_path)) if config_path else {}
    paths.ensure_dirs()

    print(f"loading buildings from {paths.SEATTLE_EUI_CSV}")
    df = m.load_buildings()
    print(f"  {len(df)} buildings after filtering")

    print("fitting honest model (spline + ridge)...")
    model, num_cols = m.fit_honest(
        df,
        n_knots=cfg.get("model", {}).get("n_knots", 6),
        degree=cfg.get("model", {}).get("degree", 3),
        ridge_alpha=cfg.get("model", {}).get("ridge_alpha", 2.0),
    )
    r2 = model.score(
        df[num_cols + ["LargestPropertyUseType"]].assign(
            PropertyGFATotal=lambda x: np.log(x["PropertyGFATotal"].clip(lower=1))),
        df["log_eui"],
    )
    print(f"  train R2 = {r2:.3f}")

    lat_lo, lat_hi = df["Latitude"].min() - 0.02, df["Latitude"].max() + 0.02
    lon_lo, lon_hi = df["Longitude"].min() - 0.02, df["Longitude"].max() + 0.02
    grid_n = cfg.get("model", {}).get("grid_n", m.GRID_N)
    print(f"rasterizing tract demographics at {grid_n}x{grid_n}...")
    ras = m.build_b_raster(lat_lo, lat_hi, lon_lo, lon_hi, n=grid_n)
    b_bldg = m.b_lookup(ras, df["Latitude"].to_numpy(), df["Longitude"].to_numpy())
    print(f"  {int((b_bldg >= m.TAU_B).sum())} / {len(df)} buildings in gated tracts "
          f"({(b_bldg >= m.TAU_B).mean():.1%})")

    df.to_csv(paths.SEATTLE_BUILDINGS_CSV, index=False)
    pickle.dump((model, num_cols), open(paths.SEATTLE_MODEL_PKL, "wb"))
    np.savez(paths.SEATTLE_PREP_NPZ, B=ras["B"], lats=ras["lats"], lons=ras["lons"],
             b_bldg=b_bldg)
    print(f"wrote {paths.SEATTLE_BUILDINGS_CSV}")
    print(f"wrote {paths.SEATTLE_MODEL_PKL}")
    print(f"wrote {paths.SEATTLE_PREP_NPZ}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                            / "configs" / "model.yaml"))
    args = ap.parse_args()
    main(args.config)
