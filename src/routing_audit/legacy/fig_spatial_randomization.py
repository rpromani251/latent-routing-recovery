"""
Spatial randomization test for the alignment claim, run standalone against the v1/probe-
comparison output (`legacy_onman_seattle.csv`, BH-flagged).
SUPERSEDED — see legacy/README.md. The underlying alignment_test/tract_index_raster
utilities are NOT superseded and live in ..spatial_randomization; this script is only the
standalone figure built on the superseded protocol's flag column. Current usage is inside
scripts/make_figures.py and scripts/experiments/spatial_randomization.py.

Run:  python -m src.routing_audit.legacy.fig_spatial_randomization
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..model import TAU_B
from .. import spatial_randomization as SR
from .. import paths

N_PERM = 999
NEAR_M = 200.0
LAT_RANGE = (47.48, 47.75)
LON_RANGE = (-122.44, -122.24)


def main():
    d = pd.read_csv(paths.RESULTS / "legacy_onman_seattle.csv")
    tid, bvals, lats, lons = SR.tract_index_raster(
        paths.TRACT_DEMOGRAPHICS_GEOJSON, LAT_RANGE, LON_RANGE)
    print(f"tract raster {tid.shape}, {len(bvals)} tracts, "
          f"{int((bvals>=TAU_B).sum())} gated")

    lat = d["lat"].to_numpy(); lon = d["lon"].to_numpy()
    rng = np.random.default_rng(2026)
    out = {}

    for kind in ("ambient", "onmanifold"):
        flag = d[f"{kind}_flag_bh"].to_numpy(bool)
        obs, null, p = SR.alignment_test(flag, lat, lon, tid, bvals, lats, lons,
                                         TAU_B, radius_m=NEAR_M, n_perm=N_PERM, rng=rng)
        out[kind] = (obs, null, p)
        print(f"\n{kind}:  {int(flag.sum())} flagged")
        print(f"  observed frac within {NEAR_M:.0f} m of a boundary : {obs:.4f}")
        print(f"  permutation null  mean {null.mean():.4f}   p95 {np.quantile(null,0.95):.4f}"
              f"   max {null.max():.4f}")
        print(f"  p = {p:.4f}   (B = {N_PERM} permutations)")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for a, kind in zip(ax, ("ambient", "onmanifold")):
        obs, null, p = out[kind]
        a.hist(null, bins=40, color="#9AA0A6", label="permuted tract labels")
        a.axvline(obs, color="#B9432F", lw=2.4, label=f"observed  ({obs:.3f})")
        a.set_title(f"{kind}   p = {p:.4f}", fontsize=10)
        a.set_xlabel(f"frac of flags within {NEAR_M:.0f} m of a gate boundary")
        a.legend(fontsize=8, frameon=False)
    ax[0].set_ylabel("permutations")
    fig.suptitle("Spatial randomization: flags align with the gate, "
                 "valid under arbitrary spatial dependence", fontsize=10)
    fig.tight_layout()
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = paths.FIGURES / "legacy_fig_spatial_randomization.png"
    fig.savefig(out_path, dpi=175)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
