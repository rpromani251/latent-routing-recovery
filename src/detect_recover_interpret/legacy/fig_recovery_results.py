"""
Three-panel results figure under the v1/probe-comparison protocol: K=2 recovery,
probe-family comparison, spatial randomization.
SUPERSEDED — see legacy/README.md. Current figures come from scripts/make_figures.py.

Expects `legacy_onman_seattle.csv` (see legacy/probe_family_comparison.py, with a
`{kind}_flag_bh` column added per probe family via `dispersion.benjamini_hochberg` on
`{kind}_g_dip_p`) under results/.

Run:  python -m src.detect_recover_interpret.legacy.fig_recovery_results
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..model import TAU_B, PENALTY
from .. import spatial_randomization as SR
from .. import paths

LAT_RANGE = (47.48, 47.75)
LON_RANGE = (-122.44, -122.24)


def main():
    d = pd.read_csv(paths.RESULTS / "legacy_onman_seattle.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # ---- A: K=2 recovered penalty
    KIND = "onmanifold"
    fl = d[f"{KIND}_flag_bh"].to_numpy(bool)
    bins = np.linspace(0, 0.45, 70)
    ax[0].hist(d.loc[d[f"{KIND}_h_delta"].notna(), f"{KIND}_h_delta"], bins=bins,
               color="#9AA0A6", alpha=0.85, label="honest model (no gate)")
    ax[0].hist(d.loc[fl, f"{KIND}_g_delta"], bins=bins,
               color="#B9432F", alpha=0.9, label="gated, dip-flagged")
    ax[0].axvline(PENALTY, color="#111", ls="--", lw=2, label=f"true penalty = {PENALTY}")
    med = np.median(d.loc[fl, f"{KIND}_g_delta"].dropna())
    ax[0].set_xlabel(r"recovered penalty  $\hat\Delta$  (log-EUI)")
    ax[0].set_ylabel("buildings")
    ax[0].set_title(f"A.  K=2 regime recovery\n"
                    f"$\\hat\\Delta$ median {med:.3f}  vs  true {PENALTY:.3f}", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)

    # ---- B: detection radius, ambient vs on-manifold
    edges = [0, 50, 100, 200, 400, 800, 1600]
    lab = ["0–50", "50–100", "100–200", "200–400", "400–800", "800–1600"]
    xs = np.arange(len(lab)); w = 0.38
    for k, (kind, col) in enumerate((("ambient", "#4C72B0"), ("onmanifold", "#B9432F"))):
        ys = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d["d_bound"] >= lo) & (d["d_bound"] < hi)
            ys.append(np.nanmean(d.loc[m, f"{kind}_g_dip_p"] < 0.05) if m.sum() else np.nan)
        ax[1].bar(xs + (k - 0.5) * w, ys, w, color=col,
                  label={"ambient": "ambient probe", "onmanifold": "on-manifold (kNN) probe"}[kind])
    far = {k_: np.nanmean(d.loc[d["d_bound"].isna(), f"{k_}_g_dip_p"] < 0.05)
           for k_ in ("ambient", "onmanifold")}
    ax[1].axhline(far["onmanifold"], ls=":", c="#555", lw=1.2,
                  label=f"far from any boundary ({far['onmanifold']:.3f})")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(lab, fontsize=8, rotation=35, ha="right")
    ax[1].set_xlabel("true distance to gate boundary (m)")
    ax[1].set_ylabel("dip flag rate")
    ax[1].set_title("B.  Probe family comparison\nequal close in, on-manifold decays faster",
                    fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)

    # ---- C: spatial randomization
    tid, bvals, lats, lons = SR.tract_index_raster(
        paths.TRACT_DEMOGRAPHICS_GEOJSON, LAT_RANGE, LON_RANGE)
    lat = d["lat"].to_numpy(); lon = d["lon"].to_numpy()
    rng = np.random.default_rng(2026); B = 399

    res = {}
    for kind in ("ambient", "onmanifold"):
        f = d[f"{kind}_flag_bh"].to_numpy(bool)
        o, nl, p = SR.alignment_test(f, lat, lon, tid, bvals, lats, lons,
                                     TAU_B, radius_m=100.0, n_perm=B, rng=rng)
        res[kind] = (o, nl, p)

    ax[2].hist(res["ambient"][1], bins=32, color="#4C72B0", alpha=0.45, label="null (ambient)")
    ax[2].hist(res["onmanifold"][1], bins=32, color="#B9432F", alpha=0.45, label="null (on-manifold)")
    ax[2].axvline(res["ambient"][0], color="#4C72B0", lw=2.4,
                  label=f"ambient obs, p={res['ambient'][2]:.3f}")
    ax[2].axvline(res["onmanifold"][0], color="#B9432F", lw=2.4,
                  label=f"on-manifold obs, p={res['onmanifold'][2]:.3f}")
    ax[2].set_xlabel("frac of flags within 100 m of a gate boundary")
    ax[2].set_ylabel("permutations")
    ax[2].set_title("C.  Spatial randomization\npermuted tract demographics, "
                    f"B={B}", fontsize=10)
    ax[2].legend(fontsize=7.5, frameon=False)

    fig.tight_layout()
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / "legacy_fig_recovery_results.png"
    fig.savefig(out, dpi=175)
    print(f"wrote {out}")
    for k_, (o, nl, p) in res.items():
        print(f"  {k_:<12} obs={o:.4f}  null mean={nl.mean():.4f}  p={p:.4f}")


if __name__ == "__main__":
    main()
