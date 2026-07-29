"""
Baseline comparison on the 2-D known-regimes simulation.

Methods compared on identical models (sim2d: wavy spatial gate, Delta = 0.30):

  dip-scan (ours)    naive multi-scale dip scan + K=2 recovery (from sim2d run)
  global-cluster     detrend anchor responses on a global smooth fit, then
                     1-D GMM with K in {1,2} selected by BIC  (v2 sec. 12)
  residual-outlier   flag anchors with |residual| > 2.5 sd from the smooth fit
  spa-profile        SPA profile-mean recovery: per-anchor probe means,
                     detrended; simplex vertices = robust extremes; pi_hat by
                     interpolation (the June-notes route; v2 sec. 11 baseline)

The pEx-SBM graph baseline requires the geospatial-xai-attacks repo and is run
separately when available (see repo staging notes).

Common metrics: existence call on the honest model (false alarm), existence on
the gated model, partition accuracy up to permutation (all anchors / detectable
subset), Delta_hat error, queries per anchor.
"""
import numpy as np
import pandas as pd
from sim_core import gmm2_fit, _norm_logpdf
from sim2d_known_regimes import (anchors_all, honest_f, gated_f, gate_ind,
                                 TAU_OBS, DELTA, SEED)

Q_REPEAT = 25          # queries/anchor for anchor-level baselines
M_SPA = 50             # probes/anchor for SPA profile means
SIG_SPA = 0.10


def poly_design(X, deg=3):
    """Monomial design matrix up to total degree `deg` in 2 variables."""
    cols = [np.ones(len(X))]
    for i in range(1, deg + 1):
        for j in range(i + 1):
            cols.append(X[:, 0] ** (i - j) * X[:, 1] ** j)
    return np.column_stack(cols)


def ridge_detrend(X, y, lam=1e-4, deg=3):
    D = poly_design(X, deg)
    w = np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ y)
    return y - D @ w


def bic_gmm(y, rng):
    """BIC-selected K in {1,2}; returns (K, fit_or_None)."""
    n = len(y)
    mu, sd = y.mean(), max(y.std(), 1e-12)
    ll1 = float(_norm_logpdf(y, mu, sd).sum())
    bic1 = -2 * ll1 + 2 * np.log(n)
    fit = gmm2_fit(y, rng)
    bic2 = -2 * fit["loglik"] + 5 * np.log(n)
    return (2, fit) if bic2 < bic1 else (1, None)


def acc_perm(pred, truth):
    """Binary partition accuracy up to label permutation."""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    a = (pred == truth).mean()
    return float(max(a, 1 - a))


def main():
    rng = np.random.default_rng(SEED + 999)
    A = anchors_all()
    truth = gate_ind(A)
    det = None  # detectable mask loaded from sim2d run

    ours = pd.read_csv("sim2d_anchors.csv")
    og = ours[ours.model == "gated"].sort_values("idx")
    det = og.detectable.to_numpy()

    rows = []

    # ---------------- ours (from the sim2d run)
    pred = (og.flag & (og.z_hat == 0)).to_numpy()
    oh = ours[ours.model == "honest"]
    fl = og[og.flag & np.isfinite(og.delta_at_pmin)]
    rows.append(dict(
        method="dip-scan (ours)",
        exist_honest=float(oh.flag.mean()),
        exist_gated=float(og.flag.mean()),
        part_acc_all=acc_perm(pred, truth),
        part_acc_detectable=acc_perm(pred[det], truth[det]),
        delta_err=float(abs(fl.delta_at_pmin.median() - DELTA)),
        queries_per_anchor=6 * 1000 + 1,
    ))

    # ---------------- anchor-level responses for the cheap baselines
    for model, f in (("gated", gated_f), ("honest", honest_f)):
        y = f(A) + rng.normal(0, TAU_OBS / np.sqrt(Q_REPEAT), len(A))
        e = ridge_detrend(A, y)
        if model == "gated":
            e_g = e
        else:
            e_h = e

    # global clustering
    kg, fg = bic_gmm(e_g, rng)
    kh, fh = bic_gmm(e_h, rng)
    if kg == 2:
        pred_gc = fg["resp"][:, 0] > 0.5          # lower component = penalized
        d_gc = float(fg["mu"][1] - fg["mu"][0])
    else:
        pred_gc = np.zeros(len(A), bool)
        d_gc = np.nan
    rows.append(dict(
        method="global-cluster",
        exist_honest=float(kh == 2),
        exist_gated=float(kg == 2),
        part_acc_all=acc_perm(pred_gc, truth),
        part_acc_detectable=acc_perm(pred_gc[det], truth[det]),
        delta_err=float(abs(d_gc - DELTA)) if np.isfinite(d_gc) else np.nan,
        queries_per_anchor=Q_REPEAT,
    ))

    # residual-outlier
    thr = 2.5
    fo_g = np.abs(e_g) > thr * np.std(e_g)
    fo_h = np.abs(e_h) > thr * np.std(e_h)
    rows.append(dict(
        method="residual-outlier",
        exist_honest=float(fo_h.mean()),
        exist_gated=float(fo_g.mean()),
        part_acc_all=acc_perm(fo_g, truth),
        part_acc_detectable=acc_perm(fo_g[det], truth[det]),
        delta_err=np.nan,
        queries_per_anchor=Q_REPEAT,
    ))

    # SPA profile-mean
    def spa(f):
        prof = np.empty(len(A))
        for i, x0 in enumerate(A):
            P = x0[None, :] + rng.normal(0, SIG_SPA, size=(M_SPA, 2))
            prof[i] = (f(P) + rng.normal(0, TAU_OBS, M_SPA)).mean()
        e = ridge_detrend(A, prof)
        v_lo, v_hi = np.quantile(e, [0.02, 0.98])
        pi = np.clip((e - v_lo) / max(v_hi - v_lo, 1e-12), 0, 1)
        return pi, float(v_hi - v_lo)

    pi_g, d_spa_g = spa(gated_f)
    pi_h, d_spa_h = spa(honest_f)
    pred_spa = pi_g < 0.5                          # low end = penalized vertex
    rows.append(dict(
        method="spa-profile",
        exist_honest=np.nan,                       # SPA has no calibrated test
        exist_gated=np.nan,
        part_acc_all=acc_perm(pred_spa, truth),
        part_acc_detectable=acc_perm(pred_spa[det], truth[det]),
        delta_err=float(abs(d_spa_g - DELTA)),
        queries_per_anchor=M_SPA,
    ))
    # SPA's uncalibrated 'gap' on the honest model — structure found where none is
    rows[-1]["delta_honest_spurious"] = d_spa_h

    out = pd.DataFrame(rows)
    out.to_csv("baseline_comparison.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
