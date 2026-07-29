"""
Known-regimes simulation, 2-D: a spatial gate with a curved boundary.

Mimics the Seattle geometry class: a threshold gate whose boundary is a curve
in the plane, an honest model that is locally linear with a mild long-lengthscale
smooth term (amplitude far below the penalty, as measured for Seattle), and
location-only probing.

  boundary   x2 = 0.5 + 0.08 sin(4 pi x1)            (wavy, curvature + corners)
  gate       g(x) = 1[x2 >= boundary(x1)]
  honest     h(x) = b1 x1 + b2 x2 + GP(amp 0.02, ell 0.5)   (precondition holds:
             ell = 0.5 > ladder top 0.2; amp 15x below the penalty)
  gated      f(x) = h(x) - 0.30 g(x)

Protocol identical to sim1d (naive 3-scale dip scan, Bonferroni, K=2 recovery).
Truth: pi_true per scale; exact distance to the boundary for reporting.
Resumable in anchor blocks; `summarize` aggregates.
"""
import os
import sys
import numpy as np
import pandas as pd
from sim_core import audit_anchor, rff_gp_path, summarize

DELTA = 0.30
B1, B2 = 0.10, 0.10
GP_AMP, GP_ELL = 0.02, 0.50
SCALES = np.geomspace(0.02, 0.2, 3)
M_DIP = M_REC = 1000
TAU_OBS = 0.02
N_ANCHOR = 480
N_BLOCK = 8
SEED = 20260729

_gp = rff_gp_path(2, GP_AMP, GP_ELL, 256, np.random.default_rng(4242))


def boundary(x1):  return 0.5 + 0.08 * np.sin(4 * np.pi * x1)
def gate_ind(X):   return X[:, 1] >= boundary(X[:, 0])
def honest_f(X):   return B1 * X[:, 0] + B2 * X[:, 1] + _gp(X)
def gated_f(X):    return honest_f(X) - DELTA * gate_ind(X)


def anchors_all():
    rng = np.random.default_rng(SEED)
    return rng.uniform(0.02, 0.98, size=(N_ANCHOR, 2))


_curve = None
def dist_to_boundary(x):
    global _curve
    if _curve is None:
        t = np.linspace(0, 1, 4000)
        _curve = np.column_stack([t, boundary(t)])
    return float(np.sqrt(((x[None, :] - _curve) ** 2).sum(1)).min())


def run_block(bi, model):
    out = f"s2_{model}_b{bi}.csv"
    if os.path.exists(out):
        return False
    A = anchors_all()
    idx = np.array_split(np.arange(N_ANCHOR), N_BLOCK)[bi]
    f = gated_f if model == "gated" else honest_f
    rng = np.random.default_rng(SEED + 100 * bi + (0 if model == "gated" else 1))
    rows = []
    for k in idx:
        x0 = A[k]
        r = audit_anchor(f, gate_ind, x0, SCALES, rng,
                         m_dip=M_DIP, m_rec=M_REC, tau_obs=TAU_OBS)
        r.update(model=model, idx=int(k), x1=float(x0[0]), x2=float(x0[1]),
                 gated_side=bool(gate_ind(x0[None, :])[0]),
                 dist=dist_to_boundary(x0))
        rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)
    return True


def do_summarize():
    df = pd.concat([pd.read_csv(f"s2_{m}_b{b}.csv")
                    for m in ("gated", "honest") for b in range(N_BLOCK)])
    df.to_csv("sim2d_anchors.csv", index=False)

    s = summarize(df.to_dict("records"))
    pd.DataFrame([s]).to_csv("sim2d_summary.csv", index=False)
    print("=== overall ===\n", pd.DataFrame([s]).round(3).to_string())

    g = df[df.model == "gated"].copy()
    dbins = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.80]
    g["d_bin"] = pd.cut(g.dist, dbins, include_lowest=True)
    dd = (g.groupby("d_bin", observed=False)
            .agg(n=("flag", "size"), flag_rate=("flag", "mean"),
                 delta_pmin=("delta_at_pmin", "median")).reset_index())
    dd.to_csv("sim2d_distance_summary.csv", index=False)
    print("\n=== gated, by distance to boundary ===\n", dd.round(3).to_string())

    pbins = [0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.51]
    g["pi_bin"] = pd.cut(g.pi_true_max.fillna(0.0), pbins, include_lowest=True)
    pdd = (g.groupby("pi_bin", observed=False)
             .agg(n=("flag", "size"), flag_rate=("flag", "mean"),
                  delta_pmin=("delta_at_pmin", "median")).reset_index())
    pdd.to_csv("sim2d_pi_summary.csv", index=False)
    print("\n=== gated, by pi_true ===\n", pdd.round(3).to_string())

    # orientation: among flagged gated-side anchors, z_hat==0 means the anchor's
    # own response joined the LOWER (penalized) component
    fl = g[g.flag & (g.z_hat >= 0)]
    if len(fl):
        ok = ((fl.z_hat == 0) == fl.gated_side).mean()
        print(f"\norientation accuracy on flagged anchors: {ok:.3f}  (n={len(fl)})")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        do_summarize()
        return
    import multiprocessing as mp
    from dip import preload_null_table
    preload_null_table(M_DIP, f"null_table_{M_DIP}.npz")
    jobs = [(b, m) for b in range(N_BLOCK) for m in ("gated", "honest")]
    with mp.Pool(4) as p:
        for r in p.starmap(run_block, jobs):
            pass
    print("ALL DONE" if all(os.path.exists(f"s2_{m}_b{b}.csv")
                            for b, m in jobs) else "PARTIAL — rerun")


if __name__ == "__main__":
    main()
