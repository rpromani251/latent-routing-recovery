"""
E1 — Perturbing the full input vector.

PROPOSITION P1. Let the gate be a threshold on coordinate 1 at distance d from
the anchor, with honest slopes beta_j = beta on every coordinate, dimension D.
(a) Isotropic probe with FIXED TOTAL budget (per-coord sd sigma/sqrt(D)):
    crossing rate pi = Phi(-d sqrt(D)/sigma) -> collapses with D.
(b) Isotropic probe with FIXED PER-COORD scale sigma: pi is D-independent, but
    within-branch response spread grows as sigma*beta*sqrt(D), so the resolvable
    gap ratio Delta / (sigma beta sqrt(D)) shrinks and the dip loses power.
(c) Local-linear residualization of y on the displacement removes the off-axis
    spread exactly (it is linear), at the cost of absorbing part of the step
    into the fitted slope; most of (b)'s loss is recovered.
(d) Coordinate-targeted scanning (probe one coordinate at a time, Bonferroni
    over D x scales) restores ~1-D power at ~D x query cost.

Ordering predicted at fixed budget-per-test: (d) ~ (c) > (b) >> (a) for D >= 8.
"""
import os
import sys
import numpy as np
import pandas as pd
from sim_core import gmm2_fit
from ext_core import (ols_residualize, trimmed_residualize,
                      min_signal_ok, dip_bonferroni, ALPHA)
from dip import dip_pvalue

BETA = 0.15
DELTA = 0.30
TAU = 0.02
SCALES = [0.05, 0.10, 0.20]
M = 1000
DIMS = [2, 4, 8, 16]
N_ANCHOR = 100
SEED = 20260730


def make_models(D):
    b = np.full(D, BETA)
    def honest(X):  return X @ b
    def gated(X):   return X @ b - DELTA * (X[:, 0] >= 0.5)
    def gind(X):    return X[:, 0] >= 0.5
    return honest, gated, gind


def audit(f, gind, x0, D, strategy, rng):
    """Returns flag, p_min, pi_true_max for one anchor under one strategy."""
    samples, pis = [], []
    if strategy in ("iso_total", "iso_coord", "iso_resid", "iso_resid_rob"):
        sd_scale = (lambda s: s / np.sqrt(D)) if strategy == "iso_total" else (lambda s: s)
        for s in SCALES:
            X = np.atleast_2d(x0) + rng.normal(0, sd_scale(s), size=(M, D))
            fr = float(np.mean(gind(X)))
            pis.append(min(fr, 1 - fr))
            y = f(X) + rng.normal(0, TAU, M)
            if not min_signal_ok(y, TAU):
                continue
            if strategy == "iso_resid":
                samples.append(ols_residualize(y, X, x0))
            elif strategy == "iso_resid_rob":
                samples.append(trimmed_residualize(y, X, x0))
            else:
                samples.append(y)
    elif strategy == "coord_scan":
        for s in SCALES:
            for j in range(D):
                X = np.tile(np.atleast_2d(x0), (M, 1))
                X[:, j] += rng.normal(0, s, M)
                fr = float(np.mean(gind(X)))
                pis.append(min(fr, 1 - fr))
                y = f(X) + rng.normal(0, TAU, M)
                if not min_signal_ok(y, TAU):
                    continue
                samples.append(y)
    flag, p_min, _ = dip_bonferroni(samples, ALPHA)
    return flag, p_min, (max(pis) if pis else np.nan), len(samples)


def run_cell(D, strategy, model):
    out = f"e1_D{D}_{strategy}_{model}.csv"
    if os.path.exists(out):
        return False
    rng = np.random.default_rng(SEED + D * 100 + hash(strategy) % 50 + (0 if model == "gated" else 7))
    honest, gated, gind = make_models(D)
    f = gated if model == "gated" else honest
    anchors = np.linspace(0.35, 0.65, N_ANCHOR)
    rows = []
    for a in anchors:
        x0 = np.full(D, 0.5); x0[0] = a
        flag, p, pi, ns = audit(f, gind, x0, D, strategy, rng)
        rows.append(dict(D=D, strategy=strategy, model=model, anchor=a,
                         dist=abs(a - 0.5), flag=flag, p_min=p,
                         pi_true_max=pi, n_tests=ns))
    pd.DataFrame(rows).to_csv(out, index=False)
    return True


CELLS = [(D, s, m) for D in DIMS
         for s in ("iso_total", "iso_coord", "iso_resid", "iso_resid_rob",
                   "coord_scan")
         for m in ("gated", "honest")]


def do_summarize():
    df = pd.concat([pd.read_csv(f"e1_D{D}_{s}_{m}.csv") for D, s, m in CELLS])
    df.to_csv("e1_anchors.csv", index=False)
    g = df[df.model == "gated"]; h = df[df.model == "honest"]
    det = g[g.pi_true_max >= 0.05]
    summ = (det.groupby(["D", "strategy"])
               .agg(n_det=("flag", "size"), power=("flag", "mean")).reset_index())
    fp = (h.groupby(["D", "strategy"]).flag.mean().rename("fp_honest").reset_index())
    summ = summ.merge(fp, on=["D", "strategy"])
    # queries per anchor per strategy
    summ["queries"] = np.where(summ.strategy == "coord_scan",
                               summ.D * len(SCALES) * M, len(SCALES) * M)
    summ.to_csv("e1_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    # crossing-rate check for P1(a): empirical pi vs Phi(-d sqrt(D)/sigma)
    from math import erf, sqrt
    sub = df[(df.model == "gated") & (df.strategy == "iso_total")]
    chk = sub.groupby("D").pi_true_max.median().reset_index()
    for _, r in chk.iterrows():
        d_med = 0.075  # median |anchor-0.5| over the band
        pred = 0.5 * (1 - erf(d_med * np.sqrt(r.D) / 0.2 / sqrt(2)))
        print(f"D={int(r.D)}: median pi_true={r.pi_true_max:.3f}  "
              f"P1(a) prediction at median dist={pred:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        do_summarize()
    else:
        import time
        t0 = time.time()
        for D, s, m in CELLS:
            if run_cell(D, s, m):
                print(f"e1 D={D} {s} {m} ({time.time()-t0:.0f}s)", flush=True)
                if time.time() - t0 > 25:
                    print("CHUNK LIMIT"); sys.exit(0)
        print("E1 ALL DONE")
