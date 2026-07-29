"""
Three-panel Seattle map under the v1 (sigma*) protocol, shared basemap and shared tract
overlay:
  A  b(x): tract Black/Latino share, with the tau = 0.25 gate region outlined
  B  R^log dispersion statistic, hex-binned over buildings
  C  dip statistic at sigma*, hex-binned over buildings
SUPERSEDED — see legacy/README.md. Current figures come from scripts/make_figures.py.

Expects `legacy_audit_seattle.csv` (see legacy/seattle_audit_v1.py) under results/.

Run:  python -m src.detect_recover_interpret.legacy.fig_three_panel_map
"""
import json

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ..dispersion import benjamini_hochberg
from ..model import TAU_B
from .. import paths

Q_BH = 0.10
LAT = (47.49, 47.74)
LON = (-122.44, -122.24)
ASPECT = 1 / np.cos(np.deg2rad(47.61))


def main():
    d = pd.read_csv(paths.RESULTS / "legacy_audit_seattle.csv")
    ok = d["g_dip_p"].notna()

    # ---------------------------------------------------------------- BH screening
    raw = (d["g_dip_p"] < 0.05) & ok
    flag_bh = np.zeros(len(d), bool)
    flag_bh[np.where(ok)[0]] = benjamini_hochberg(d.loc[ok, "g_dip_p"].to_numpy(), q=Q_BH)
    print(f"n tested            {int(ok.sum())}")
    print(f"raw p<0.05 flags    {int(raw.sum())}   ({raw.sum()/ok.sum():.3f})")
    print(f"BH q={Q_BH} flags     {int(flag_bh.sum())}   ({flag_bh.sum()/ok.sum():.3f})")
    print(f"expected null flags at p<0.05 if all null: {0.05*ok.sum():.0f}")
    print(f"honest-model raw flags: {int((d['h_dip_p'] < 0.05).sum())} / {int(d['h_dip_p'].notna().sum())}")
    d["flag_bh"] = flag_bh
    d.to_csv(paths.RESULTS / "legacy_audit_seattle.csv", index=False)

    # ---------------------------------------------------------------- tracts
    gj = json.load(open(paths.TRACT_DEMOGRAPHICS_GEOJSON))
    tracts = []
    for ft in gj["features"]:
        v = ft["properties"].get("pct_black_latino")
        if v is None:
            continue
        geom = ft["geometry"]
        rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else \
                [p[0] for p in geom["coordinates"]]
        for ring in rings:
            a = np.asarray(ring, float)[:, :2]
            if a[:, 0].max() < LON[0] or a[:, 0].min() > LON[1]: continue
            if a[:, 1].max() < LAT[0] or a[:, 1].min() > LAT[1]: continue
            tracts.append((a, float(v)))
    print(f"tracts drawn: {len(tracts)}   gated: {sum(1 for _, v in tracts if v >= TAU_B)}")

    def overlay(ax, lw=1.5):
        """Gate region outline, drawn identically on every panel so they are comparable."""
        for a, v in tracts:
            if v >= TAU_B:
                ax.add_patch(MplPoly(a, closed=True, facecolor="none",
                                     edgecolor="#B9432F", lw=lw, zorder=6))

    def frame(ax, title):
        ax.set_xlim(*LON); ax.set_ylim(*LAT); ax.set_aspect(ASPECT)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.6))

    # ---- A: b(x) choropleth
    norm = Normalize(0, max(v for _, v in tracts))
    cmap = plt.get_cmap("YlOrRd")
    for a, v in tracts:
        ax[0].add_patch(MplPoly(a, closed=True, facecolor=cmap(norm(v)),
                                edgecolor="white", lw=0.5, zorder=2))
    overlay(ax[0])
    frame(ax[0], f"A.  $b(x)$ — tract Black/Latino share\nred outline: gate region $b\\geq{TAU_B}$")
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax[0],
                 fraction=0.037, pad=0.02).set_label("$b(x)$", fontsize=9)

    # ---- B: R^log hexbin
    for a, v in tracts:
        ax[1].add_patch(MplPoly(a, closed=True, facecolor="#F7F7F7",
                                edgecolor="#E0E0E0", lw=0.4, zorder=1))
    sub = d[ok]
    hb = ax[1].hexbin(sub["lon"], sub["lat"], C=sub["g_R_log"], gridsize=42,
                      reduce_C_function=np.mean, mincnt=1, cmap="viridis",
                      vmin=np.nanpercentile(sub["g_R_log"], 5),
                      vmax=np.nanpercentile(sub["g_R_log"], 95), zorder=3, linewidths=0.2)
    overlay(ax[1])
    frame(ax[1], "B.  $R^{\\log}$ dispersion statistic\ndetection radius $\\approx$ 100 m")
    plt.colorbar(hb, ax=ax[1], fraction=0.037, pad=0.02).set_label("mean $R^{\\log}$", fontsize=9)

    # ---- C: dip hexbin
    for a, v in tracts:
        ax[2].add_patch(MplPoly(a, closed=True, facecolor="#F7F7F7",
                                edgecolor="#E0E0E0", lw=0.4, zorder=1))
    hc = ax[2].hexbin(sub["lon"], sub["lat"], C=sub["g_dip"], gridsize=42,
                      reduce_C_function=np.mean, mincnt=1, cmap="magma_r",
                      vmin=np.nanpercentile(sub["g_dip"], 5),
                      vmax=np.nanpercentile(sub["g_dip"], 95), zorder=3, linewidths=0.2)
    overlay(ax[2])
    frame(ax[2], f"C.  Hartigan dip at $\\sigma^\\star$\ndetection radius $\\approx$ 800 m  ·  "
                  f"{int(flag_bh.sum())} flagged at BH $q$={Q_BH}")
    plt.colorbar(hc, ax=ax[2], fraction=0.037, pad=0.02).set_label("mean dip", fontsize=9)

    fig.tight_layout()
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / "legacy_fig_three_panel_map.png"
    fig.savefig(out, dpi=175, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
