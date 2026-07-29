#!/usr/bin/env python3
"""
Measure the curvature lengthscale ell of the honest EUI model, and locate it relative to
the probe ladder. Writes results/figures/fig_lengthscale.png (a CURRENT supporting figure,
see docs/results_2026-07-28.md S4b).

The dispersion statistic is only informative when the honest model is straight at every
scale probed, i.e. ell > sigma_T. This measures ell two independent ways:

(A) PARAMETRIC -- fit the closed form to the honest model's own dispersion curves.
    Var(sigma) = sigma^2 beta^2 + s^2 (1 - (1 + 2 sigma^2/ell^2)^(-d/2)),  d = 2 (lat, lon)
    This is the same fit the (superseded) null generator used, run on gate-free
    responses, so ell_min is set negligibly small to avoid clipping the thing measured.

(B) NON-PARAMETRIC -- evaluate the honest location surface on a grid, remove the
    linear trend, and read off the lag at which spatial correlation falls to exp(-1/2).
    Assumes no functional form, so agreement with (A) is evidence the SE-GP null
    class is adequate rather than merely convenient.
"""
import pickle
import sys
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.routing_audit import paths
from src.routing_audit.legacy.null_generator import fit_null_hyper
from src.routing_audit.location_terms import build_location_terms, M_LAT, M_LON

SIG_M = np.geomspace(25.0, 1200.0, 12)
M_PROBE = 200
TAU_OBS = 0.010
N_ANCHOR = 500
D_PROBE = 2


def honest_curve(k, loc, rng):
    lat0, lon0, hb = loc["lat0"][k], loc["lon0"][k], loc["h_base"][k]
    r = np.zeros(len(SIG_M)); lam = np.zeros(len(SIG_M))
    for t, sm in enumerate(SIG_M):
        la = lat0 + rng.normal(0, sm / M_LAT, M_PROBE)
        lo = lon0 + rng.normal(0, sm / M_LON, M_PROBE)
        y = hb + np.interp(la, loc["lat_g"], loc["g_lat"]) \
               + np.interp(lo, loc["lon_g"], loc["g_lon"]) \
               + rng.normal(0, TAU_OBS, M_PROBE)
        lam[t] = np.var(y, ddof=1) / TAU_OBS**2
        r[t] = np.sqrt(max(lam[t] - 1.0, 0.0)) / sm
    valid = stats.chi2.sf((M_PROBE - 1) * lam, df=M_PROBE - 1) < 0.05
    return r, lam, valid


def main():
    df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
    model, num = pickle.load(open(paths.SEATTLE_MODEL_PKL, "rb"))
    loc = build_location_terms(model, num, df)
    loc["lat0"] = df["Latitude"].to_numpy(); loc["lon0"] = df["Longitude"].to_numpy()
    rng = np.random.default_rng(99)

    # ---------------------------------------------------------------- (A)
    idx = rng.choice(len(df), min(N_ANCHOR, len(df)), replace=False)
    ells, ss, betas = [], [], []
    for k in idx:
        r, lam, v = honest_curve(k, loc, rng)
        if v.sum() < 4:
            continue
        h = fit_null_hyper(r, SIG_M, v, TAU_OBS, D_PROBE, ell_min=1.0, ell_max=1e6)
        if h is None:
            continue
        ells.append(h["ell"]); ss.append(h["s"]); betas.append(h["beta"])
    ells = np.array(ells); ss = np.array(ss)

    print(f"(A) parametric fit to honest dispersion curves   n = {len(ells)}")
    for q in (10, 25, 50, 75, 90):
        print(f"      ell  p{q:<2} = {np.percentile(ells, q):>10.0f} m")
    print(f"      amplitude s  median = {np.median(ss):.3f} log-EUI")
    print(f"\n    probe ladder: {SIG_M[0]:.0f} m ... {SIG_M[-1]:.0f} m")
    print(f"    fraction of anchors with ell inside the ladder : "
          f"{np.mean((ells >= SIG_M[0]) & (ells <= SIG_M[-1])):.3f}")
    print(f"    fraction with ell  >  sigma_T (condition holds): "
          f"{np.mean(ells > SIG_M[-1]):.3f}")

    # ---------------------------------------------------------------- (B)
    n = 220
    lat_g = np.linspace(df["Latitude"].min(), df["Latitude"].max(), n)
    lon_g = np.linspace(df["Longitude"].min(), df["Longitude"].max(), n)
    LO, LA = np.meshgrid(lon_g, lat_g)
    Z = (np.interp(LA.ravel(), loc["lat_g"], loc["g_lat"])
         + np.interp(LO.ravel(), loc["lon_g"], loc["g_lon"]))
    P = np.column_stack([(LO.ravel() - lon_g[0]) * M_LON,
                         (LA.ravel() - lat_g[0]) * M_LAT])
    A = np.column_stack([np.ones(len(P)), P])           # remove the linear trend
    Z = Z - A @ np.linalg.lstsq(A, Z, rcond=None)[0]
    Z = Z - Z.mean()

    sub = rng.choice(len(P), 4000, replace=False)
    Ps, Zs = P[sub], Z[sub]
    dist = np.sqrt(((Ps[:, None, :] - Ps[None, :, :]) ** 2).sum(-1))
    prod = Zs[:, None] * Zs[None, :]
    iu = np.triu_indices(len(sub), 1)
    dv, pv = dist[iu], prod[iu]
    var = Zs.var()

    edges = np.geomspace(20, 12000, 26)
    ctr, corr = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (dv >= lo_) & (dv < hi_)
        if m.sum() < 200:
            continue
        ctr.append(np.sqrt(lo_ * hi_)); corr.append(pv[m].mean() / var)
    ctr, corr = np.array(ctr), np.array(corr)

    target = np.exp(-0.5)
    below = np.where(corr < target)[0]
    ell_np = np.interp(target, [corr[below[0]], corr[below[0] - 1]],
                       [ctr[below[0]], ctr[below[0] - 1]]) if below.size and below[0] > 0 else np.nan

    print(f"\n(B) non-parametric correlation of the honest location surface")
    print(f"      lag where corr drops to exp(-1/2) = {ell_np:,.0f} m")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
    ax[0].hist(np.log10(ells), bins=40, color="#4C72B0")
    for s_, c_, lab in ((SIG_M[0], "#B9432F", r"$\sigma_1$"),
                        (SIG_M[-1], "#B9432F", r"$\sigma_T$")):
        ax[0].axvline(np.log10(s_), color=c_, ls="--", lw=1.6)
        ax[0].annotate(lab, (np.log10(s_), ax[0].get_ylim()[1] * 0.92),
                       color=c_, fontsize=9, ha="center")
    ax[0].axvspan(np.log10(SIG_M[0]), np.log10(SIG_M[-1]), color="#B9432F", alpha=0.09)
    ax[0].set_xlabel(r"$\log_{10}\ \hat\ell$  (m)"); ax[0].set_ylabel("anchors")
    ax[0].set_title("(A) fitted curvature lengthscale\nshaded = probe ladder", fontsize=10)

    ax[1].semilogx(ctr, corr, "o-", color="#111111", lw=2, ms=4)
    ax[1].axhline(target, ls=":", c="#888", label=r"$e^{-1/2}$")
    if np.isfinite(ell_np):
        ax[1].axvline(ell_np, ls="-", c="#2E7D32", lw=1.6,
                      label=fr"$\ell \approx {ell_np:,.0f}$ m")
    ax[1].axvspan(SIG_M[0], SIG_M[-1], color="#B9432F", alpha=0.12, label="probe ladder")
    ax[1].axhline(0, c="#ccc", lw=0.8)
    ax[1].set_xlabel("lag (m)"); ax[1].set_ylabel("spatial correlation")
    ax[1].set_title("(B) honest location surface, trend removed", fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / "fig_lengthscale.png"
    fig.savefig(out, dpi=175)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
