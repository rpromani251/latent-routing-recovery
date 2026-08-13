"""
E4 — Non-geographic anchor graphs: clumpy covariates and the A12 repair.

The attribute-graph experiment failed because building attributes are clumpy
(construction booms, standard floor counts): an on-manifold probe inherits the
data's own multimodality and the dip reads it as model multimodality.

PROPOSITION P4. Let the covariate distribution be a C-cluster mixture with
inter-cluster spacing L and the honest response A11-smooth. On-manifold probes
at scale sigma:
(a) For sigma comparable to L, the probe straddles clusters and the honest
    model is flagged far above nominal -- the A12 violation, reproduced.
(b) Local-linear residualization of y on the probe displacement removes the
    trend-induced multimodality (the honest surface is near-linear across the
    probe span, by A11), restoring near-nominal FP while the gate's jump --
    which no linear function fits -- retains power.
(c) On an evenly distributed (uniform) manifold the raw dip is already valid
    at all scales: the failure is a property of the DATA distribution, not of
    on-manifold probing.

Design: 2-D attribute space; data = 6-cluster Gaussian mixture (sd 0.05,
centers >= 0.3 apart), n = 4000, vs uniform control of the same size. Honest
model linear + long-lengthscale GP (A11 holds). Gate: x1 >= 0.55, Delta 0.30.
kNN on-manifold probes (min 12 distinct neighbours). Strategies: raw dip |
residualized dip, each on clumpy and uniform manifolds.
"""
import os
import sys
import numpy as np
import pandas as pd
from sim_core import rff_gp_path
from ext_core import (KnnProbe, ols_residualize, trimmed_residualize,
                      min_signal_ok, dip_bonferroni, ALPHA)

DELTA = 0.30
TAU = 0.02
SCALES = [0.05, 0.10, 0.20]
M = 1000
N_DATA = 4000
N_ANCHOR = 130
SEED = 20260733

_gp = rff_gp_path(2, 0.02, 0.5, 256, np.random.default_rng(999))
B = np.array([0.50, 0.50])   # strong attribute trend: inter-cluster
                             # response gaps ~0.1-0.35, rivaling Delta

CENTERS = np.array([[0.2, 0.2], [0.2, 0.75], [0.5, 0.45],
                    [0.8, 0.2], [0.8, 0.8], [0.45, 0.85]])


def honest_f(X):  return X @ B + _gp(X)
def gated_f(X):   return honest_f(X) - DELTA * (X[:, 0] >= 0.55)
def gind(X):      return X[:, 0] >= 0.55


def make_manifold(kind, rng):
    if kind == "uniform":
        return rng.uniform(0.02, 0.98, size=(N_DATA, 2))
    z = rng.integers(0, len(CENTERS), N_DATA)
    return np.clip(CENTERS[z] + rng.normal(0, 0.05, size=(N_DATA, 2)), 0, 1)


def run_cell(manifold, method, model):
    out = f"e4b_{manifold}_{method}_{model}.csv"
    if os.path.exists(out):
        return False
    rng = np.random.default_rng(SEED + hash((manifold, method, model)) % 9999)
    pts = make_manifold(manifold, np.random.default_rng(SEED))   # shared cloud
    probe = KnnProbe(pts)
    f = gated_f if model == "gated" else honest_f
    a_idx = np.random.default_rng(SEED + 1).choice(N_DATA, N_ANCHOR, replace=False)
    rows = []
    for k in a_idx:
        x0 = pts[k]
        samples, pis = [], []
        for s in SCALES:
            X = probe.draw(x0, s, M, rng)
            if X is None:
                continue
            fr = float(np.mean(gind(X)))
            pis.append(min(fr, 1 - fr))
            y = f(X) + rng.normal(0, TAU, M)
            if not min_signal_ok(y, TAU):
                continue
            if method == "resid":
                samples.append(ols_residualize(y, X, x0))
            elif method == "resid_rob":
                samples.append(trimmed_residualize(y, X, x0))
            else:
                samples.append(y)
        flag, p_min, _ = dip_bonferroni(samples, ALPHA)
        rows.append(dict(manifold=manifold, method=method, model=model,
                         x1=float(x0[0]), x2=float(x0[1]),
                         gated_side=bool(gind(x0[None, :])[0]),
                         pi_true_max=(max(pis) if pis else np.nan),
                         n_tests=len(samples), flag=flag, p_min=p_min,
                         abstain=len(samples) == 0))
    pd.DataFrame(rows).to_csv(out, index=False)
    return True


CELLS = [(mf, me, mo) for mf in ("clumpy", "uniform")
         for me in ("raw", "resid", "resid_rob") for mo in ("gated", "honest")]


def do_summarize():
    df = pd.concat([pd.read_csv(f"e4b_{a}_{b}_{c}.csv") for a, b, c in CELLS])
    df.to_csv("e4_anchors.csv", index=False)
    h = df[df.model == "honest"]
    g = df[(df.model == "gated") & (df.pi_true_max >= 0.05)]
    summ = (h.groupby(["manifold", "method"])
              .agg(fp_honest=("flag", "mean"), abstain=("abstain", "mean"))
              .reset_index())
    pw = (g.groupby(["manifold", "method"]).flag.mean()
            .rename("power_detectable").reset_index())
    summ = summ.merge(pw, on=["manifold", "method"])
    summ.to_csv("e4_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        do_summarize()
    else:
        import time
        t0 = time.time()
        for a, b, c in CELLS:
            if run_cell(a, b, c):
                print(f"e4 {a} {b} {c} ({time.time()-t0:.0f}s)", flush=True)
                if time.time() - t0 > 25:
                    print("CHUNK LIMIT"); sys.exit(0)
        print("E4 ALL DONE")
