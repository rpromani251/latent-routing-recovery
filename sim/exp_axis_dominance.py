"""
Experiment T -- axis dominance and the threshold t_hat.  Step 4 of the v5 critical path.

WHAT IS BEING TESTED. Pooling returns a hyperplane: an ambient normal n_hat in R^D and
an offset c_hat. That is a weighted mix of every feature, which is true and nearly
useless. The claim an audit needs is "it routes on feature i above t_hat". Stage 7
specifies the conversion:

    axis-dominance:  is one |n_hat_i| dominant?
                     calibrate against the permutation
                     distribution of max_j |n_hat_j|
    if yes:  coordinate i, threshold t_hat = c_hat / n_hat_i, CI from the bootstrap
    if no:   report the hyperplane, no single-coordinate claim

PRE-REGISTERED PREDICTIONS, stated before running.

  T-1  The permutation calibration is STRUCTURALLY BROKEN, and near-powerless exactly
       where it matters. The recovered normal always lies in span(U) -- n_hat = U nu /
       ||U nu|| -- and the permutation shuffles residuals WITHIN an anchor, leaving the
       frame untouched. So permuted normals live in the same 2-D subspace. If the gate
       is axis-aligned AND visible at all, the frame MUST contain that axis (a component
       off the frame is unrecoverable by construction), so a permuted normal is
       cos(phi)*u1 + sin(phi)*u2 with phi uniform, and |n_hat_i| = |cos phi|. Its 95th
       percentile is sin(0.95 * pi/2) = 0.9969. The observed |n_hat_i| is cos(pooled
       orientation error) = cos(6.3 deg) = 0.994 at Delta/tau = 1.5, N = 100 -- BELOW
       the null's own 95th percentile. The test should fail to fire on a perfectly
       axis-aligned gate.
       FALSIFIED BY: fire rate materially above 0.05 at theta = 0.

  T-2  A sharp usability boundary between Delta/tau = 1.5 and 1.0. Pooled orientation
       error is 6.3 deg at 1.5 and 34.3 deg at 1.0 (Experiment P, N = 100), so the mass
       leaking off the true axis is sin(6.3) = 0.11 against sin(34.3) = 0.56. Dominance
       (however tested) should survive the first and collapse at the second.

  T-3  t_hat = c_hat / n_hat_i inherits BOTH error sources multiplicatively, so its
       error should exceed what the offset error alone implies. c_hat converges to
       0.26 sigma at Delta/tau = 1.5 and 0.12 sigma at 2.5 (Experiment P).

  T-4  The false-claim rate -- dominance firing on a genuinely oblique gate -- rises
       with the frame's own axis alignment rather than the gate's, for the same reason
       as T-1.

DESIGN. Ambient D = 20, intrinsic d = 2, frame supplied (Stage 0 deferred), a single
planted hyperplane, anchors placed by design in the orientation-useful shell (pi = 0.10)
because Experiment P showed geometric pooling is dead under random placement.

The true ambient gate normal is

    n_true(theta) = cos(theta) * e_i + sin(theta) * v,      v unit, orthogonal to e_i,
                                                            spread over the other D-1

so theta = 0 is a perfectly axis-aligned rule "x_i >= T" and theta = 90 deg is a generic
oblique one with no single-coordinate truth at all. The frame is U = [n_true, w] with w a
random direction orthogonal to n_true: the gate normal must lie in the frame or it is
unrecoverable by construction (the parallel-penalty limit), so the frame's alignment is
not a free parameter -- which is precisely why T-1 is structural and not a setup artefact.

TWO RULES COMPARED.
  perm       Stage 7 as written: max_j |n_hat_j| against the permutation null's q95.
  bootstrap  resample the pooled ANCHORS, re-pool, and ask whether the IDENTITY of the
             dominant coordinate is stable. This asks a different and more answerable
             question -- not "is this dominant against a no-signal null" but "would I
             name the same feature again on a re-draw" -- and it is the natural repair
             if T-1 holds.

    python3 exp_axis_dominance.py
"""
import os
import sys
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("p", "exp_p_pooling.py")     # anchor_primitive, pool_direction, point_estimate

# ------------------------------------------------------------------ settings
D_AMBIENT, D_INTR = 20, 2
SIGMA = 0.20                  # per-coordinate probe scale
M_PROBE = 600
TAU = 0.02
BETA_NORM = 0.15
POOL = 220                    # anchors generated per cell
N_PERM_REP = 2                # permuted replicas per anchor
T_TRUE = 5.0                  # the planted single-coordinate threshold at theta = 0
PI_TARGET = 0.10              # by-design placement, the orientation-useful shell
SEED = 20260814

THETA_DEG = [0.0, 10.0, 20.0, 30.0, 45.0, 90.0]
DELTA_OVER_TAU = [1.0, 1.5, 2.5, 5.0]
N_LIST = [25, 50, 100, 200]
FRAME_REPS = 5                # independent draws of the axis, of v, and of w
N_SUBSAMPLE = 200             # pooled draws per (cell, N) -> the rates
N_PERM_DRAW = 200             # pooled permuted draws -> the null for max|n_hat_j|
B_CI = 300                    # anchor bootstrap for the t_hat interval
N_CI_SUB = 60                 # subsamples on which the CI is actually computed
ALPHA = 0.05
STABILITY = 0.95              # bootstrap rule: name a coordinate only at this stability
TOLERANCES = [5.0, 10.0, 15.0]  # equivalence rule: claim only if the off-axis angle is
                                # provably below this many degrees

PARTS = os.path.join(HERE, "_parts_axis2")


# ------------------------------------------------------------------ geometry
def build_frame(theta_rad, rng):
    """Return (U, axis_index, n_true_ambient).

    U is D x 2 orthonormal with U[:, 0] = n_true, so the gate normal lies in the frame.
    That is forced, not chosen: any component of the gate normal off the frame is
    unrecoverable (the parallel-penalty limit), so a visible axis-aligned gate implies
    an axis-aligned frame.
    """
    i = int(rng.integers(D_AMBIENT))
    e = np.zeros(D_AMBIENT); e[i] = 1.0
    v = rng.normal(size=D_AMBIENT)
    v -= (v @ e) * e
    v /= np.linalg.norm(v)                       # unit, orthogonal to e_i, spread
    n_true = np.cos(theta_rad) * e + np.sin(theta_rad) * v
    n_true /= np.linalg.norm(n_true)
    w = rng.normal(size=D_AMBIENT)
    w -= (w @ n_true) * n_true
    w /= np.linalg.norm(w)
    return np.column_stack([n_true, w]), i, n_true


def build_pool(dt_ratio, rng):
    """Anchor pool in MANIFOLD coordinates, with the gate at nu = (1, 0), c = T_TRUE.

    Lifting x = U t with the manifold through the ambient origin makes the ambient
    offset equal the manifold offset, so t_hat = c_hat / n_hat_i recovers T_TRUE
    exactly when theta = 0 and recovery is perfect.
    """
    delta = dt_ratio * TAU
    nu_true = np.array([1.0, 0.0])
    beta = BETA_NORM * np.array([np.cos(2.1), np.sin(2.1)])
    d_shell = -SIGMA * norm.ppf(PI_TARGET)       # |distance| realising pi = 0.10

    real, perm = [], []
    for _ in range(POOL):
        dist = d_shell * rng.uniform(0.95, 1.05)
        side = rng.choice([-1.0, 1.0])
        along = rng.normal(0.0, 4.0 * SIGMA)
        t_a = nu_true * (side * dist + T_TRUE) + np.array([-nu_true[1], nu_true[0]]) * along

        Z = rng.normal(0.0, SIGMA, size=(M_PROBE, D_INTR))
        T = t_a[None, :] + Z
        crossed = (T @ nu_true) > T_TRUE
        y = T @ beta - delta * crossed + rng.normal(0.0, TAU, M_PROBE)

        out = P.anchor_primitive(Z, y, t_a, rng, permute=False)
        if out is not None:
            real.append(out)
        for _ in range(N_PERM_REP):
            o = P.anchor_primitive(Z, y, t_a, rng, permute=True)
            if o is not None:
                perm.append(o)

    def unpack(lst):
        if not lst:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        return (np.array([a[0] for a in lst]), np.array([a[1] for a in lst]),
                np.array([a[2] for a in lst]))
    return unpack(real), unpack(perm)


def lift(nu_hat, U):
    """Manifold normal -> ambient unit normal."""
    n = U @ nu_hat
    return n / np.linalg.norm(n)


# ------------------------------------------------------------------ the two rules
def dominance_stats(nv, cs, w, U):
    """Pool, lift, and return (n_ambient, c_hat, argmax, max|n_j|)."""
    nu_hat, c_hat = P.point_estimate(nv, cs, w)
    n_amb = lift(nu_hat, U)
    a = int(np.argmax(np.abs(n_amb)))
    return n_amb, c_hat, a, float(np.abs(n_amb).max())


def bootstrap_axis(nv, cs, w, U, rng, cand, B=B_CI):
    """Resample the pooled anchors.

    Returns (modal axis, its stability, t_hat draws, angle UCB to `cand`).

    The last one is the ingredient for the equivalence rule. Axis-dominance is a claim
    that the off-axis part of the normal is SMALL, so the burden belongs on showing that
    -- a one-sided upper confidence bound on the angle between n_hat and the candidate
    axis -- not on rejecting a no-signal null, which is what Stage 7 asks for and which
    can be rejected by a normal that is merely well determined while being oblique.
    """
    N = len(w)
    axes = np.empty(B, int)
    ts = np.empty(B)
    ang = np.empty(B)
    for b in range(B):
        j = rng.integers(0, N, N)
        n_amb, c_hat, a, _ = dominance_stats(nv[j], cs[j], w[j], U)
        axes[b] = a
        ts[b] = c_hat / n_amb[a] if abs(n_amb[a]) > 1e-9 else np.nan
        ang[b] = np.degrees(np.arccos(np.clip(abs(n_amb[cand]), 0.0, 1.0)))
    vals, cnt = np.unique(axes, return_counts=True)
    k = int(np.argmax(cnt))
    return int(vals[k]), float(cnt[k] / B), ts, float(np.quantile(ang, 1 - ALPHA))


# ------------------------------------------------------------------ work unit
def run_unit(args):
    theta_deg, dt, rep = args
    path = os.path.join(PARTS, f"th{int(theta_deg):02d}__dt{dt}__r{rep}.csv")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(SEED + 101 * rep + 7919 * int(theta_deg) + int(dt * 1000))
    U, axis_i, n_true = build_frame(np.radians(theta_deg), rng)
    (nR, cR, wR), (nP, cP, wP) = build_pool(dt, rng)
    if min(len(wR), len(wP)) < max(N_LIST):
        return path

    rows = []
    for N in N_LIST:
        if min(len(wR), len(wP)) < N:
            continue
        # --- the permutation null for max_j |n_hat_j|, pooled at the same N
        null_max = np.empty(N_PERM_DRAW)
        for r in range(N_PERM_DRAW):
            j = rng.choice(len(wP), N, replace=False)
            null_max[r] = dominance_stats(nP[j], cP[j], wP[j], U)[3]
        thr_perm = float(np.quantile(null_max, 1 - ALPHA))

        fired_p, fired_b, correct, tvals, cov, width = [], [], [], [], [], []
        obs_max, ucbs = [], []
        fired_e = {t: [] for t in TOLERANCES}
        for s in range(N_SUBSAMPLE):
            j = rng.choice(len(wR), N, replace=False)
            n_amb, c_hat, a, mx = dominance_stats(nR[j], cR[j], wR[j], U)
            obs_max.append(mx)
            fired_p.append(mx > thr_perm)
            correct.append(a == axis_i)
            if s < N_CI_SUB:
                mode_a, stab, ts, ucb = bootstrap_axis(nR[j], cR[j], wR[j], U, rng, a)
                fired_b.append(stab >= STABILITY)
                ucbs.append(ucb)
                for t in TOLERANCES:
                    # the equivalence rule also requires naming the RIGHT axis
                    fired_e[t].append(bool(ucb < t and a == axis_i))
                if theta_deg == 0.0:
                    t_hat = c_hat / n_amb[a] if abs(n_amb[a]) > 1e-9 else np.nan
                    tvals.append(t_hat)
                    lo, hi = np.nanpercentile(ts, [2.5, 97.5])
                    cov.append(bool(lo <= T_TRUE <= hi))
                    width.append(float(hi - lo))
        row = dict(
            theta_deg=theta_deg, dt_ratio=dt, rep=rep, N=N, axis_true=axis_i,
            thr_perm=thr_perm, null_max_med=float(np.median(null_max)),
            obs_max_med=float(np.median(obs_max)),
            fire_perm=float(np.mean(fired_p)),
            fire_boot=float(np.mean(fired_b)) if fired_b else np.nan,
            ucb_med=float(np.median(ucbs)) if ucbs else np.nan,
            correct_axis=float(np.mean(correct)),
            t_hat_med=float(np.nanmedian(tvals)) if tvals else np.nan,
            t_err_med=float(np.nanmedian(np.abs(np.array(tvals) - T_TRUE))) if tvals else np.nan,
            ci_coverage=float(np.mean(cov)) if cov else np.nan,
            ci_width_med=float(np.median(width)) if width else np.nan,
            n_pool=int(len(wR)))
        for t in TOLERANCES:
            row[f"fire_equiv_{int(t)}"] = float(np.mean(fired_e[t])) if fired_e[t] else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    os.makedirs(PARTS, exist_ok=True)
    units = [(th, dt, r) for th in THETA_DEG for dt in DELTA_OVER_TAU
             for r in range(FRAME_REPS)]
    todo = [u for u in units
            if not os.path.exists(os.path.join(
                PARTS, f"th{int(u[0]):02d}__dt{u[1]}__r{u[2]}.csv"))]
    print(f"{len(units)} units, {len(todo)} to run", flush=True)
    if todo:
        import multiprocessing as mp
        nproc = min(int(os.environ.get("NPROC", "2")), max(1, mp.cpu_count()))
        with mp.Pool(nproc) as pool:
            for i, p in enumerate(pool.imap_unordered(run_unit, todo), 1):
                print(f"  [{i}/{len(todo)}] {os.path.basename(p)}", flush=True)
    frames = []
    for u in units:
        p = os.path.join(PARTS, f"th{int(u[0]):02d}__dt{u[1]}__r{u[2]}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 5:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(HERE, "axis_dominance_rows.csv"), index=False)
    print(f"\nwrote axis_dominance_rows.csv ({len(df)} rows)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        import time
        t0 = time.time()
        p = run_unit((0.0, 2.5, 99))
        print(f"  smoke unit in {time.time()-t0:.1f}s -> {p}")
        print(pd.read_csv(p).to_string(index=False))
    else:
        main()
