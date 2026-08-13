"""
E3 — Non-scalar outputs: the projection choice for the modality test.

Model: input x in R, output in R^V (V=3). Within-branch variation lies along a
unit direction u (response = u * beta * x); the gate subtracts Delta * v on the
gated side, with v at angle theta to u. Isotropic noise tau (whitening trivial,
as in the scalar pipeline; estimated whitening is future work).

PROPOSITION P3. (a) The dip on the top principal component fails as
theta -> 90 deg whenever within-branch variance (beta*sigma)^2 exceeds the
between-branch contribution (~ Delta^2/4 at balanced mixing): PC1 locks to u
and the projected gap scales as Delta|cos angle(PC1, v)|.
(b) A prespecified projection set -- top two PCs, Bonferroni over the set --
restores power whenever the response variation is effectively rank <= 2 (it is:
span{u, v}), at a factor-2 multiplicity cost.
(c) Coordinate axes (Bonferroni V) work only when v is axis-aligned; J random
projections pay |cos| concentration ~ 1/sqrt(V), a real but graceful cost.
(d) Vector recovery: responsibilities from the 1-D mixture on the best
projection transfer to the full space; the component-conditional mean
difference recovers the penalty VECTOR (norm and direction), not just a gap.

Strategies: pc1 | pc2_bonf (top-2) | axes_bonf (V) | rand8_bonf.
Sweep theta in {0, 30, 60, 90} deg at V=3; V=8 variant at theta=90.
"""
import os
import sys
import numpy as np
import pandas as pd
from sim_core import gmm2_fit
from ext_core import min_signal_ok, dip_bonferroni, ALPHA

BETA = 3.0
DELTA = 0.30
TAU = 0.02
SCALES = [0.05, 0.10, 0.20]
M = 1000
M_REC = 1000
N_ANCHOR = 60
SEED = 20260732
THETAS = [0, 30, 60, 90]


def dirs(V, theta_deg, rng):
    u = np.zeros(V); u[0] = 1.0
    t = np.deg2rad(theta_deg)
    v = np.zeros(V); v[0] = np.cos(t); v[1] = np.sin(t)
    return u, v


def make_models(V, theta, rng):
    u, v = dirs(V, theta, rng)
    def honest(x):  # x: (n,)  -> (n, V)
        return np.outer(BETA * x, u)
    def gated(x):
        return honest(x) - DELTA * np.outer((x >= 0.5).astype(float), v)
    return honest, gated, u, v


def projections(Y, strategy, rng):
    """Return list of 1-D projected samples for the dip."""
    Yc = Y - Y.mean(0)
    if strategy in ("pc1", "pc2_bonf"):
        _, _, VT = np.linalg.svd(Yc, full_matrices=False)
        k = 1 if strategy == "pc1" else 2
        return [Yc @ VT[i] for i in range(k)]
    if strategy == "axes_bonf":
        return [Yc[:, j] for j in range(Y.shape[1])]
    if strategy == "rand8_bonf":
        P = rng.normal(size=(8, Y.shape[1]))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        return [Yc @ p for p in P]
    raise ValueError(strategy)


def run_cell(V, theta, strategy, model):
    out = f"e3c_V{V}_t{theta}_{strategy}_{model}.csv"
    if os.path.exists(out):
        return False
    rng = np.random.default_rng(SEED + V * 1000 + theta * 7 + hash(strategy) % 97
                                + (0 if model == "gated" else 3))
    honest, gated, u, v = make_models(V, theta, rng)
    f = gated if model == "gated" else honest
    anchors = np.linspace(0.44, 0.56, N_ANCHOR)
    rows = []
    for a in anchors:
        samples = []
        for s in SCALES:
            x = a + rng.normal(0, s, M)
            Y = f(x) + rng.normal(0, TAU, size=(M, V))
            # min-signal on the total variance (scalar rule per scale)
            if not min_signal_ok(np.linalg.norm(Y - Y.mean(0), axis=1), TAU * np.sqrt(V)):
                continue
            samples.extend(projections(Y, strategy, rng))
        flag, p_min, _ = dip_bonferroni(samples, ALPHA)

        # (d) vector recovery on a fresh draw at the mid scale
        dvec_norm, dvec_cos = np.nan, np.nan
        if flag and model == "gated":
            x = a + rng.normal(0, SCALES[1], M_REC)
            Y = f(x) + rng.normal(0, TAU, size=(M_REC, V))
            projs = projections(Y, strategy, rng)
            _, _, kbest = dip_bonferroni(projs, ALPHA)
            z = projs[max(kbest, 0)]
            fit = gmm2_fit(z, rng)
            r = fit["resp"][:, 0] > 0.5
            if 20 <= r.sum() <= M_REC - 20:
                # PENALTY VECTOR = difference of the two group-conditional
                # local-linear predictions AT THE ANCHOR (delta = 0). A raw
                # group-mean difference is contaminated by the trend times the
                # difference in where the groups sit in input space.
                dx = (x - a)
                dv = np.empty(V)
                for out_dim in range(V):
                    icpt = []
                    for grp in (r, ~r):
                        Z = np.column_stack([np.ones(grp.sum()), dx[grp]])
                        coef, *_ = np.linalg.lstsq(Z, Y[grp, out_dim], rcond=None)
                        icpt.append(coef[0])
                    dv[out_dim] = icpt[1] - icpt[0]
                c = abs(float(dv @ v) / (np.linalg.norm(dv) + 1e-12))
                dvec_norm, dvec_cos = float(np.linalg.norm(dv)), c
        rows.append(dict(V=V, theta=theta, strategy=strategy, model=model,
                         anchor=a, flag=flag, p_min=p_min,
                         dvec_norm=dvec_norm, dvec_cos=dvec_cos))
    pd.DataFrame(rows).to_csv(out, index=False)
    return True


CELLS = ([(3, t, s, m) for t in THETAS
          for s in ("pc1", "pc2_bonf", "axes_bonf", "rand8_bonf")
          for m in ("gated", "honest")]
         + [(8, 90, s, m) for s in ("pc1", "pc2_bonf", "rand8_bonf")
            for m in ("gated", "honest")])


def do_summarize():
    df = pd.concat([pd.read_csv(f"e3c_V{V}_t{t}_{s}_{m}.csv") for V, t, s, m in CELLS])
    df.to_csv("e3_anchors.csv", index=False)
    g = df[df.model == "gated"]; h = df[df.model == "honest"]
    summ = (g.groupby(["V", "theta", "strategy"])
              .agg(power=("flag", "mean"),
                   dnorm=("dvec_norm", "median"),
                   dcos=("dvec_cos", "median")).reset_index())
    fp = (h.groupby(["V", "theta", "strategy"]).flag.mean()
            .rename("fp_honest").reset_index())
    summ = summ.merge(fp, on=["V", "theta", "strategy"])
    summ.to_csv("e3_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        do_summarize()
    else:
        import time
        t0 = time.time()
        for V, t, s, m in CELLS:
            if run_cell(V, t, s, m):
                print(f"e3 V={V} th={t} {s} {m} ({time.time()-t0:.0f}s)", flush=True)
                if time.time() - t0 > 25:
                    print("CHUNK LIMIT"); sys.exit(0)
        print("E3 ALL DONE")
