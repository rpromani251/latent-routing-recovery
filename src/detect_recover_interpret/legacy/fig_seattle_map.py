"""
Poster figures under the v1 (sigma*) protocol: Seattle map with both statistics, and the
conformal coverage gap.
SUPERSEDED — see legacy/README.md. Current figures come from scripts/make_figures.py.

Expects the batches written by legacy/seattle_audit_v1.py, concatenated into a single
`legacy_audit_seattle.csv` under results/ (concatenation across index batches is a manual
step, same as in the original pipeline).

Run:  python -m src.detect_recover_interpret.legacy.fig_seattle_map
"""
import json

import numpy as np, pandas as pd, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.lines import Line2D

from ..model import TAU_B, PENALTY
from ..conformal import conformal_report
from .. import paths

ALPHA_DIP = 0.05
COV = 0.90


def main():
    d = pd.read_csv(paths.RESULTS / "legacy_audit_seattle.csv")
    df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
    model, num = pickle.load(open(paths.SEATTLE_MODEL_PKL, "rb"))
    prep = np.load(paths.SEATTLE_PREP_NPZ)
    b_bldg = prep["b_bldg"]

    X = df[num + ["LargestPropertyUseType"]].copy()
    X["PropertyGFATotal"] = np.log(X["PropertyGFATotal"].clip(lower=1))
    h = model.predict(X)
    f = h - PENALTY * (b_bldg >= TAU_B)          # deployed (gated) predictor
    y = df["log_eui"].to_numpy()
    cf = conformal_report(f, b_bldg >= TAU_B, y, coverage=COV)

    print(f"split conformal at nominal {COV:.0%}   interval half-width {cf['q']:.3f} log-EUI")
    print(f"  marginal coverage            {cf['marginal']:.3f}")
    print(f"  coverage | penalized branch  {cf['penalized']:.3f}   (n={cf['n_pen']})")
    print(f"  coverage | unpenalized       {cf['unpenalized']:.3f}   (n={cf['n_unp']})")
    print(f"  conditional coverage gap     {cf['unpenalized']-cf['penalized']:+.3f}")

    gj = json.load(open(paths.TRACT_DEMOGRAPHICS_GEOJSON))
    LAT = (47.49, 47.74); LON = (-122.44, -122.24)

    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.9], wspace=0.22)
    axm = fig.add_subplot(gs[0, 0]); axr = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2])

    def draw_tracts(ax):
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
                gated = v >= TAU_B
                ax.add_patch(MplPoly(a, closed=True,
                                     facecolor="#F2C9C0" if gated else "#F2F2F2",
                                     edgecolor="#B9432F" if gated else "#CFCFCF",
                                     lw=1.4 if gated else 0.4, zorder=1))
        ax.set_xlim(*LON); ax.set_ylim(*LAT)
        ax.set_aspect(1 / np.cos(np.deg2rad(47.61)))
        ax.set_xticks([]); ax.set_yticks([])

    ok = d["g_dip_p"].notna()
    flag = ok & (d["g_dip_p"] < ALPHA_DIP)

    # panel 1: modality (the discriminator)
    draw_tracts(axm)
    axm.scatter(d.loc[ok & ~flag, "lon"], d.loc[ok & ~flag, "lat"], s=3.5,
                c="#9AA0A6", alpha=0.55, lw=0, zorder=2)
    axm.scatter(d.loc[flag, "lon"], d.loc[flag, "lat"], s=13,
                c="#111111", lw=0, zorder=3)
    axm.set_title("Dip test at $\\sigma^\\star$  —  flags trace the gate\n"
                  f"{int(flag.sum())} flagged  ·  0 false positives on the honest model",
                  fontsize=10)
    axm.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=9, mfc="#F2C9C0", mec="#B9432F",
               label="tract $b(x)\\geq0.25$ (penalized)"),
        Line2D([], [], marker="o", ls="", ms=5, color="#111111", label="dip-flagged building"),
        Line2D([], [], marker="o", ls="", ms=4, color="#9AA0A6", label="not flagged"),
    ], fontsize=7.5, loc="lower left", frameon=True, framealpha=0.9)

    # panel 2: dispersion (no discriminating power)
    draw_tracts(axr)
    v = d.loc[ok, "g_R_log"].to_numpy()
    sc = axr.scatter(d.loc[ok, "lon"], d.loc[ok, "lat"], s=5, c=v, lw=0,
                     cmap="viridis", vmin=np.nanpercentile(v, 2),
                     vmax=np.nanpercentile(v, 98), zorder=2)
    plt.colorbar(sc, ax=axr, fraction=0.035, pad=0.02).set_label(r"$R^{\log}$", fontsize=8)
    axr.set_title("Log-range dispersion  —  no spatial structure", fontsize=10)

    # panel 3: detection radius
    edges = [0, 50, 100, 200, 400, 800, 1600]
    lab = ["0–50", "50–100", "100–200", "200–400", "400–800", "800–1600"]
    ys, ns, keep = [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (d["d_bound"] >= lo) & (d["d_bound"] < hi) & ok
        if m.sum() == 0: continue
        ys.append(float((d.loc[m, "g_dip_p"] < ALPHA_DIP).mean()))
        ns.append(int(m.sum())); keep.append(lab[i])
    xs = np.arange(len(ys))
    m = d["d_bound"].isna() & ok
    far = float((d.loc[m, "g_dip_p"] < ALPHA_DIP).mean()) if m.sum() else float("nan")
    axc.plot(xs, ys, "o-", color="#111111", lw=2, ms=7, zorder=3)
    for x, y_, n in zip(xs, ys, ns):
        axc.annotate(f"n={n}", (x, y_), textcoords="offset points", xytext=(0, 9),
                     fontsize=7, ha="center")
    axc.axhline(far, ls=":", c="#888", lw=1.3, label=f"far from any boundary ({far:.3f})")
    axc.axhline(0.0, ls="--", c="#B9432F", lw=1.3, label="honest model (0.000)")
    axc.set_xticks(xs); axc.set_xticklabels(keep, fontsize=8, rotation=35, ha="right")
    axc.set_ylim(-0.05, 1.12)
    axc.set_xlabel("true distance to gate boundary (m)", fontsize=9)
    axc.set_ylabel("dip flag rate", fontsize=9)
    axc.set_title("Detection radius", fontsize=10)
    axc.legend(fontsize=7.5, frameon=False, loc="upper right")

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / "legacy_fig_seattle_map.png"
    fig.savefig(out, dpi=175, bbox_inches="tight")
    print(f"\nwrote {out}")

    fig2, ax2 = plt.subplots(figsize=(4.6, 4.0))
    labels = ["marginal", "unpenalized\nbranch", "penalized\nbranch"]
    vals = [cf["marginal"], cf["unpenalized"], cf["penalized"]]
    cols = ["#4C72B0", "#9AA0A6", "#B9432F"]
    ax2.bar(labels, vals, color=cols)
    ax2.axhline(COV, ls="--", c="k", lw=1.2, label=f"nominal {COV:.0%}")
    for i, v_ in enumerate(vals):
        ax2.text(i, v_ + 0.012, f"{v_:.3f}", ha="center", fontsize=9)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("empirical coverage")
    ax2.set_title("Split conformal on the gated model:\nvalid marginally, broken conditionally",
                  fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    fig2.tight_layout()
    out2 = paths.FIGURES / "fig_coverage.png"
    fig2.savefig(out2, dpi=175)
    print(f"wrote {out2}")

    pd.DataFrame([cf]).to_csv(paths.RESULTS / "conformal_result.csv", index=False)


if __name__ == "__main__":
    main()
