#!/usr/bin/env python3
"""
Current poster figure set, all regenerated from the FINAL protocol (results/seattle_audit.csv,
see scripts/run_seattle_audit.py with configs/main_audit.yaml): on-manifold probes, naive
scan over log-spaced scales, Bonferroni across scales, K=2 mixture recovery.

  fig_poster_1_mechanism.png   one clump vs two clumps -- the whole idea, real data
  fig_poster_2_map.png         b(x) | flagged buildings | recovered penalty
  fig_poster_3_results.png     recovery / power vs mixing fraction / spatial randomization
  fig_coverage.png             conformal (protocol-independent)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.routing_audit import spatial_randomization as SR
from src.routing_audit import paths
from src.routing_audit.model import TAU_B, PENALTY, b_lookup
from src.routing_audit.conformal import conformal_report
from src.routing_audit.location_terms import build_location_terms, M_LAT, M_LON
from src.routing_audit.probes import Probe, respond

LAT = (47.49, 47.74); LON = (-122.44, -122.24)
ASPECT = 1 / np.cos(np.deg2rad(47.61))
DELTA_FILTER = 0.15


def main():
    d = pd.read_csv(paths.RESULTS / "seattle_audit.csv")
    df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
    prep = np.load(paths.SEATTLE_PREP_NPZ)
    ras = {"B": prep["B"], "lats": prep["lats"], "lons": prep["lons"]}
    model, num = pickle.load(open(paths.SEATTLE_MODEL_PKL, "rb"))
    loc = build_location_terms(model, num, df)
    loc["lat0"] = df["Latitude"].to_numpy(); loc["lon0"] = df["Longitude"].to_numpy()

    lat_ref, lon_ref = loc["lat0"].min(), loc["lon0"].min()
    XY = np.column_stack([(loc["lon0"] - lon_ref) * M_LON, (loc["lat0"] - lat_ref) * M_LAT])
    probe = Probe(cKDTree(XY), np.column_stack([loc["lat0"], loc["lon0"]]),
                  "onmanifold", M_LAT, M_LON)
    probe.lat0_ref, probe.lon0_ref = lat_ref, lon_ref
    rng = np.random.default_rng(11)

    flag = d["g_flag"].fillna(False).to_numpy(bool)
    strict = flag & (d["g_delta_med"] > DELTA_FILTER).fillna(False).to_numpy()
    detect = (d["g_pi_true_max"] >= 0.05).to_numpy()

    paths.FIGURES.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- FIG 1
    near = np.where(flag & (d["g_pi_true_max"] > 0.25))[0]
    far = np.where((~flag) & (d["g_pi_true_max"] < 0.01))[0]
    k_near, k_far = int(near[0]), int(far[0])

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    for a, k, title, col in ((ax[0], k_far, "Interior building\nno hidden boundary in reach", "#4C72B0"),
                             (ax[1], k_near, "Boundary-adjacent building\nanswers split in two", "#B9432F")):
        sel = probe.draw(loc["lat0"][k], loc["lon0"][k], 600.0, 4000, rng)
        y = respond(sel[0], sel[1], loc["h_base"][k], loc, ras, rng, gated=True)
        a.hist(y, bins=70, color=col, alpha=0.9)
        a.set_title(title, fontsize=10.5)
        a.set_xlabel("model's answer  (log energy intensity)")
    ax[0].set_ylabel("number of probe queries")
    ax[1].annotate("", xy=(np.percentile(y, 5), ax[1].get_ylim()[1]*0.75),
                   xytext=(np.percentile(y, 95), ax[1].get_ylim()[1]*0.75),
                   arrowprops=dict(arrowstyle="<->", color="k", lw=1.4))
    ax[1].text(0.5, 0.80, f"gap $\\approx$ {PENALTY}\n= the hidden penalty",
               transform=ax[1].transAxes, ha="center", fontsize=9)
    fig.suptitle("Nudge a building to nearby real locations and read the answers back",
                 fontsize=11)
    fig.tight_layout()
    out1 = paths.FIGURES / "fig_poster_1_mechanism.png"
    fig.savefig(out1, dpi=175)
    print(f"wrote {out1}")

    # ---------------------------------------------------------------- FIG 2
    gj = json.load(open(paths.TRACT_DEMOGRAPHICS_GEOJSON))
    tracts = []
    for ft in gj["features"]:
        v = ft["properties"].get("pct_black_latino")
        if v is None: continue
        g = ft["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        for r_ in rings:
            a_ = np.asarray(r_, float)[:, :2]
            if a_[:, 0].max() < LON[0] or a_[:, 0].min() > LON[1]: continue
            if a_[:, 1].max() < LAT[0] or a_[:, 1].min() > LAT[1]: continue
            tracts.append((a_, float(v)))

    def base(ax_, fill=None):
        for a_, v in tracts:
            ax_.add_patch(MplPoly(a_, closed=True,
                                  facecolor=fill(v) if fill else "#F7F7F7",
                                  edgecolor="white" if fill else "#E4E4E4", lw=0.5, zorder=1))
        for a_, v in tracts:
            if v >= TAU_B:
                ax_.add_patch(MplPoly(a_, closed=True, facecolor="none",
                                      edgecolor="#B9432F", lw=1.6, zorder=6))
        ax_.set_xlim(*LON); ax_.set_ylim(*LAT); ax_.set_aspect(ASPECT)
        ax_.set_xticks([]); ax_.set_yticks([])

    fig, ax = plt.subplots(1, 3, figsize=(16, 6.6))
    norm = Normalize(0, max(v for _, v in tracts)); cmap = plt.get_cmap("YlOrRd")
    base(ax[0], fill=lambda v: cmap(norm(v)))
    ax[0].set_title("A.  The hidden rule\nneighbourhoods with $b(x)\\geq0.25$ are penalised",
                    fontsize=10.5)
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax[0],
                 fraction=0.037, pad=0.02).set_label("Black/Latino population share", fontsize=8)

    base(ax[1])
    ax[1].scatter(d.loc[~flag, "lon"], d.loc[~flag, "lat"], s=3, c="#B8BCC2", lw=0, zorder=3)
    ax[1].scatter(d.loc[strict, "lon"], d.loc[strict, "lat"], s=13, c="#111111", lw=0, zorder=4)
    h_fp_rate = float(d["h_flag"].mean())
    ax[1].set_title(f"B.  What the audit finds\n{int(strict.sum())} buildings flagged  ·  "
                    f"{h_fp_rate:.1%} false alarms on the clean model",
                    fontsize=10.5)
    ax[1].legend(handles=[
        Line2D([], [], marker="o", ls="", ms=5, color="#111111", label="flagged"),
        Line2D([], [], marker="o", ls="", ms=4, color="#B8BCC2", label="not flagged"),
        Line2D([], [], marker="s", ls="", ms=9, mfc="none", mec="#B9432F", label="penalised area")],
        fontsize=8, loc="lower left", framealpha=0.92)

    base(ax[2])
    sub = d[strict]
    sc = ax[2].scatter(sub["lon"], sub["lat"], s=11, c=sub["g_delta_med"], cmap="magma_r",
                       vmin=0.15, vmax=0.35, lw=0, zorder=4)
    ax[2].set_title("C.  How big is the hidden penalty?\nrecovered per building, true value 0.30",
                    fontsize=10.5)
    plt.colorbar(sc, ax=ax[2], fraction=0.037, pad=0.02).set_label(r"recovered $\hat\Delta$", fontsize=8)
    fig.tight_layout()
    out2 = paths.FIGURES / "fig_poster_2_map.png"
    fig.savefig(out2, dpi=175, bbox_inches="tight")
    print(f"wrote {out2}")

    # ---------------------------------------------------------------- FIG 3
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    bins = np.linspace(0, 0.45, 70)
    ax[0].hist(d["h_delta_med"].dropna(), bins=bins, color="#9AA0A6", alpha=0.9,
               label="clean model (no hidden rule)")
    ax[0].hist(d.loc[flag, "g_delta_med"].dropna(), bins=bins, color="#B9432F", alpha=0.9,
               label="rigged model, flagged")
    ax[0].axvline(PENALTY, color="#111", ls="--", lw=2, label=f"true penalty {PENALTY}")
    ax[0].set_xlabel(r"recovered penalty $\hat\Delta$")
    ax[0].set_ylabel("buildings")
    ax[0].set_title(f"A.  We recover the size of the rule\nmedian "
                    f"{np.nanmedian(d.loc[flag,'g_delta_med']):.3f} vs true {PENALTY}", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)

    edges = [0.0, 0.05, 0.10, 0.20, 0.35, 0.51]
    lab = ["<5%", "5-10%", "10-20%", "20-35%", ">35%"]
    ys, ns = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d["g_pi_true_max"] >= lo) & (d["g_pi_true_max"] < hi)
        ys.append(d.loc[m, "g_flag"].mean()); ns.append(int(m.sum()))
    ax[1].bar(np.arange(len(ys)), ys, 0.62, color=["#C9CCD1"] + ["#B9432F"]*4)
    for i, (y_, n_) in enumerate(zip(ys, ns)):
        ax[1].annotate(f"n={n_}", (i, y_), textcoords="offset points", xytext=(0, 5),
                       ha="center", fontsize=7)
    ax[1].set_xticks(range(len(lab))); ax[1].set_xticklabels(lab, fontsize=8)
    ax[1].set_xlabel("share of probes that cross the boundary")
    ax[1].set_ylabel("detection rate"); ax[1].set_ylim(0, 1.12)
    ax[1].set_title("B.  Detection is near-perfect once\n10% of probes cross", fontsize=10)

    tid, bvals, lats, lons = SR.tract_index_raster(paths.TRACT_DEMOGRAPHICS_GEOJSON, LAT, LON)
    lat_a = d["lat"].to_numpy(); lon_a = d["lon"].to_numpy()
    r2 = np.random.default_rng(2026); B = 399

    # Radius matched to the method's DETECTION RADIUS, measured independently from the
    # power-vs-distance curve -- not tuned to p. See docs/results_2026-07-28.md S7c.
    RAD = 800.0
    obs, null, p = SR.alignment_test(strict, lat_a, lon_a, tid, bvals, lats, lons,
                                     TAU_B, radius_m=RAD, n_perm=B, rng=r2)
    ax[2].hist(null, bins=34, color="#9AA0A6", label="if demographics were random")
    ax[2].axvline(obs, color="#B9432F", lw=2.6, label="what we actually see")
    ax[2].set_xlabel(f"share of flags within {RAD:.0f} m of a boundary")
    ax[2].set_ylabel("random redrawings")
    ax[2].set_title(f"C.  Flags track the real boundaries\np = {p:.3f}", fontsize=10)
    ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out3 = paths.FIGURES / "fig_poster_3_results.png"
    fig.savefig(out3, dpi=175)
    print(f"wrote {out3}   (randomization p = {p:.4f}, obs={obs:.3f})")

    print(f"\nflagged (dip+delta): {int(strict.sum())}   dip only: {int(flag.sum())}")
    print(f"clean-model false alarm rate: {h_fp_rate:.4f}")
    print(f"detectable anchors: {int(detect.sum())}   power: {d.loc[detect,'g_flag'].mean():.3f}")

    # ---------------------------------------------------------------- coverage
    X = df[num + ["LargestPropertyUseType"]].copy()
    X["PropertyGFATotal"] = np.log(X["PropertyGFATotal"].clip(lower=1))
    h = model.predict(X)
    f_deployed = h - PENALTY * (prep["b_bldg"] >= TAU_B)
    y_true = df["log_eui"].to_numpy()
    cf = conformal_report(f_deployed, prep["b_bldg"] >= TAU_B, y_true, coverage=0.90)
    print(f"\nsplit conformal at nominal 90%   interval half-width {cf['q']:.3f} log-EUI")
    print(f"  marginal coverage            {cf['marginal']:.3f}")
    print(f"  coverage | penalized branch  {cf['penalized']:.3f}   (n={cf['n_pen']})")
    print(f"  coverage | unpenalized       {cf['unpenalized']:.3f}   (n={cf['n_unp']})")
    print(f"  conditional coverage gap     {cf['unpenalized']-cf['penalized']:+.3f}")

    fig2, ax2 = plt.subplots(figsize=(4.6, 4.0))
    labels = ["marginal", "unpenalized\nbranch", "penalized\nbranch"]
    vals = [cf["marginal"], cf["unpenalized"], cf["penalized"]]
    cols = ["#4C72B0", "#9AA0A6", "#B9432F"]
    ax2.bar(labels, vals, color=cols)
    ax2.axhline(0.90, ls="--", c="k", lw=1.2, label="nominal 90%")
    for i, v_ in enumerate(vals):
        ax2.text(i, v_ + 0.012, f"{v_:.3f}", ha="center", fontsize=9)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("empirical coverage")
    ax2.set_title("Split conformal on the gated model:\nvalid marginally, broken conditionally",
                  fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    fig2.tight_layout()
    out4 = paths.FIGURES / "fig_coverage.png"
    fig2.savefig(out4, dpi=175)
    print(f"wrote {out4}")
    pd.DataFrame([cf]).to_csv(paths.RESULTS / "conformal_result.csv", index=False)


if __name__ == "__main__":
    main()
