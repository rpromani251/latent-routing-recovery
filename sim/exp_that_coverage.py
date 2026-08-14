"""
Does the repaired offset restore interval coverage on t_hat?

diag_offset_bias.py established that the pooled offset carries a MULTIPLICATIVE bias:
c_a = nu_a . t_a + t0_a uses each anchor's OWN normal, so the large "distance along the
boundary from the origin" term is attenuated by E[cos phi_a]. The bias is proportional to
the true threshold, does not shrink with N, and was invisible to Experiment P because that
run placed the boundary at c_true = 0.

Two repairs, both already implicit in the method note:
  B  project each anchor's estimated boundary POINT onto the POOLED normal, so
     per-anchor rotation error is not multiplied by the anchor's distance from the origin
  C  B, and take the distance from the anchor to the boundary from the validated crossing
     law d_hat = -sigma Phi^-1(pi_hat) rather than from the LDA midpoint of the class
     means, which is not the boundary unless pi = 0.5 (0.780 sigma against 1.282 sigma at
     pi = 0.10)

This measures what actually matters for the target sentence: the bias in t_hat, the width
of its bootstrap interval, and whether that interval covers at its nominal 95%.

The gate is axis-aligned throughout (theta = 0), because coverage of t_hat is only defined
where a single-coordinate truth exists. Ambient D = 20, intrinsic d = 2, frame supplied,
by-design placement in the pi = 0.10 shell.

    python3 exp_that_coverage.py
"""
import os
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("p", "exp_p_pooling.py")
AX = _load("ax", "exp_axis_dominance.py")

SIGMA, TAU, M_PROBE = 0.20, 0.02, 600
T_TRUE, PI_TARGET = 5.0, 0.10
BETA = 0.15 * np.array([np.cos(2.1), np.sin(2.1)])
NU_TRUE = np.array([1.0, 0.0])
TG = np.array([-NU_TRUE[1], NU_TRUE[0]])

DT_VALUES = [1.5, 2.5, 5.0]
N_LIST = [25, 50, 100, 200]
FRAME_REPS = 5
POOL = 220
N_SUB = 120                 # pooled draws per (cell, N)
B_CI = 300
SEED = 20260814
PARTS = os.path.join(HERE, "_parts_tcov")


def one_anchor(dt_ratio, rng):
    delta = dt_ratio * TAU
    dist = -SIGMA * norm.ppf(PI_TARGET) * rng.uniform(0.95, 1.05)
    side = rng.choice([-1.0, 1.0])
    along = rng.normal(0.0, 4.0 * SIGMA)
    t_a = NU_TRUE * (side * dist + T_TRUE) + TG * along

    Z = rng.normal(0.0, SIGMA, size=(M_PROBE, 2))
    T = t_a[None, :] + Z
    y = T @ BETA - delta * ((T @ NU_TRUE) > T_TRUE) + rng.normal(0.0, TAU, M_PROBE)

    resid = P.lts_residuals(Z, y)
    fit = P.gmm2_equalvar(resid, rng)
    gamma = fit["resp"][:, 0]
    nu, t0 = P.lda_direction(Z, gamma)
    if nu is None:
        return None
    if (gamma * (Z @ nu)).sum() / max(gamma.sum(), 1e-9) > (
            (1 - gamma) * (Z @ nu)).sum() / max((1 - gamma).sum(), 1e-9):
        nu, t0 = -nu, -t0
    pi_hat = float(np.clip(min(fit["w"]), 1e-6, 0.5))
    return dict(nu=nu, t0=float(t0), t_a=t_a, w=float(max(fit["lrt"], 0.0)),
                d_hat=float(-SIGMA * norm.ppf(pi_hat)))


def t_hat_of(rec, idx, U, mode):
    nv = np.array([rec[i]["nu"] for i in idx])
    w = np.array([rec[i]["w"] for i in idx])
    t0 = np.array([rec[i]["t0"] for i in idx])
    ta = np.array([rec[i]["t_a"] for i in idx])
    dh = np.array([rec[i]["d_hat"] for i in idx])
    if w.sum() <= 0:
        return np.nan, -1

    nu_hat, _ = P.pool_direction(nv, w)
    if mode == "A":
        sg = np.sign(nv @ nu_hat); sg[sg == 0] = 1
        c_hat = float((w * sg * ((nv * ta).sum(1) + t0)).sum() / w.sum())
    elif mode == "B":
        p = ta + t0[:, None] * nv
        c_hat = float((w * (p @ nu_hat)).sum() / w.sum())
    else:                                            # C
        p = ta + (np.sign(t0) * dh)[:, None] * nv
        c_hat = float((w * (p @ nu_hat)).sum() / w.sum())

    n_amb = AX.lift(nu_hat, U)
    a = int(np.argmax(np.abs(n_amb)))
    if abs(n_amb[a]) < 1e-9:
        return np.nan, a
    return float(c_hat / n_amb[a]), a


def run_unit(args):
    dt, rep = args
    path = os.path.join(PARTS, f"dt{dt}__r{rep}.csv")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(SEED + 313 * rep + int(dt * 1000))
    U, axis_i, _ = AX.build_frame(0.0, rng)          # theta = 0: axis-aligned gate
    rec = [r for r in (one_anchor(dt, rng) for _ in range(POOL)) if r is not None]
    if len(rec) < max(N_LIST):
        return path

    rows = []
    for N in N_LIST:
        for mode in ("A", "B", "C"):
            ts, cov, wid = [], [], []
            for _ in range(N_SUB):
                idx = rng.choice(len(rec), N, replace=False)
                t_pt, _ = t_hat_of(rec, idx, U, mode)
                if not np.isfinite(t_pt):
                    continue
                boot = np.array([t_hat_of(rec, idx[rng.integers(0, N, N)], U, mode)[0]
                                 for _ in range(B_CI)])
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                ts.append(t_pt); cov.append(bool(lo <= T_TRUE <= hi)); wid.append(hi - lo)
            if not ts:
                continue
            ts = np.array(ts)
            rows.append(dict(dt=dt, rep=rep, N=N, mode=mode, n_draw=len(ts),
                             t_med=float(np.median(ts)),
                             bias=float(np.median(ts) - T_TRUE),
                             sd=float(np.std(ts)),
                             ci_width=float(np.median(wid)),
                             coverage=float(np.mean(cov))))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    os.makedirs(PARTS, exist_ok=True)
    units = [(dt, r) for dt in DT_VALUES for r in range(FRAME_REPS)]
    todo = [u for u in units
            if not os.path.exists(os.path.join(PARTS, f"dt{u[0]}__r{u[1]}.csv"))]
    print(f"{len(units)} units, {len(todo)} to run", flush=True)
    if todo:
        import multiprocessing as mp
        nproc = min(int(os.environ.get("NPROC", "2")), max(1, mp.cpu_count()))
        with mp.Pool(nproc) as pool:
            for i, p in enumerate(pool.imap_unordered(run_unit, todo), 1):
                print(f"  [{i}/{len(todo)}] {os.path.basename(p)}", flush=True)
    frames = [pd.read_csv(os.path.join(HERE, PARTS, f"dt{u[0]}__r{u[1]}.csv"))
              for u in units
              if os.path.getsize(os.path.join(PARTS, f"dt{u[0]}__r{u[1]}.csv")) > 5]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(HERE, "that_coverage_rows.csv"), index=False)

    print("\nt_hat against a planted threshold of 5.0, pooled over frame replicates")
    print(f"  {'dt':>4} {'N':>5} | " + "".join(
        f"{'  '+m+': bias   cover  width':<26}" for m in "AC"))
    g = df.groupby(["dt", "N", "mode"]).agg(
        bias=("bias", "mean"), coverage=("coverage", "mean"),
        width=("ci_width", "mean")).reset_index()
    for dt in DT_VALUES:
        for N in N_LIST:
            cells = ""
            for m in "AC":
                s = g[(g.dt == dt) & (g.N == N) & (g["mode"] == m)]
                if len(s):
                    cells += (f"  {s.bias.iloc[0]:+7.4f} {s.coverage.iloc[0]:6.3f}"
                              f" {s.width.iloc[0]:7.4f}   ")
            print(f"  {dt:>4} {N:>5} | {cells}")
    print("\nwrote that_coverage_rows.csv")


if __name__ == "__main__":
    main()
