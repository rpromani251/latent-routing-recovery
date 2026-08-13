"""
Experiment P, tasks 6 and 7.

  #6  the offset c_hat as an estimand in its own right, and N50/N95 on a finer
      N grid with interpolation rather than read off a five-point ladder.

  #7  THE COMB TEST -- pre-registered prediction P-2.

P-2, as stated before running:
  A real gate produces a SINGLE MODE in the pooled offset c. Resonant curvature
  pools COHERENTLY in the normal -- a sine has a preferred direction, so every
  anchor on it yields a normal aligned with the same axis -- but places its
  offsets at the sine's steep parts, spaced by the wavelength. So it should
  produce a COMB of modes spaced by ell.

  If this fails, smooth rejection returns to the per-anchor rules (Stage 5) and
  Experiment A comes back onto the critical path.

The statistic. Take the sign-resolved, centred offsets and form the distribution
of PAIRWISE differences. A gate concentrates that distribution at zero. A comb
places secondary mass at multiples of ell. So

    comb score = (kernel density of pairwise |dc| at the best non-zero lag)
                 / (density at zero lag)

is low for a gate and high for a comb, and it needs no knowledge of ell.

Surfaces compared, all with anchors spread across the sine's phase so a comb can
form at all:
    gate        y = beta.t - Delta * 1[n.t > c]          (a real boundary)
    resonant    y = beta.t + A * sin(2 pi (u1.t) / ell)   (no boundary at all)
    honest      y = beta.t                                (control)
"""
import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("p", "/tmp/pexp/exp_p_pooling.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

SIGMA, TAU, M_PROBE = P.SIGMA, P.TAU, P.M_PROBE
SEED = 20260813
POOL = 250
N_GRID = [5, 8, 12, 18, 25, 35, 50, 70, 100]
N_REPEAT = 300
ALPHA = 0.05

N_TRUE = np.array([np.cos(0.7), np.sin(0.7)])
BETA = P.BETA_NORM * np.array([np.cos(2.1), np.sin(2.1)])
ALONG_SD = 4.0 * SIGMA                 # spread of anchors along the boundary
PROBE_DIAM = 2.0 * SIGMA * np.sqrt(2)  # ~0.57: the resonance band is around this


def make_pool(kind, rng, dt_ratio=2.5, pi_target=0.10, amp=None, ell=None,
              n_pool=POOL):
    """Anchors on one of the three surfaces. Returns (n, c, w) arrays."""
    from scipy.stats import norm
    delta = dt_ratio * TAU
    d_shell = -SIGMA * norm.ppf(pi_target)
    tang = np.array([-N_TRUE[1], N_TRUE[0]])
    out = []
    for _ in range(n_pool):
        if kind == "gate":
            dist = d_shell * rng.uniform(0.95, 1.05) * rng.choice([-1.0, 1.0])
            along = rng.normal(0.0, ALONG_SD)
        else:
            # no boundary: spread anchors across the sine's phase
            dist = rng.normal(0.0, ALONG_SD)
            along = rng.normal(0.0, ALONG_SD)
        t_a = N_TRUE * dist + tang * along
        Z = rng.normal(0.0, SIGMA, size=(M_PROBE, 2))
        T = t_a[None, :] + Z
        y = T @ BETA + rng.normal(0.0, TAU, M_PROBE)
        if kind == "gate":
            y = y - delta * ((T @ N_TRUE) > 0.0)
        elif kind == "resonant":
            y = y + amp * np.sin(2 * np.pi * (T @ N_TRUE) / ell)
        o = P.anchor_primitive(Z, y, t_a, rng)
        if o is not None:
            out.append(o)
    if not out:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    return (np.array([a[0] for a in out]), np.array([a[1] for a in out]),
            np.array([a[2] for a in out]))


# ------------------------------------------------------------------ comb score
def signed_offsets(nvecs, cs, w):
    nhat, R = P.pool_direction(nvecs, w)
    s = np.sign(nvecs @ nhat); s[s == 0] = 1
    cc = s * cs
    return cc - np.average(cc, weights=w), nhat, R


def comb_score(nvecs, cs, w, h=0.15 * PROBE_DIAM, lag_max=3.0 * PROBE_DIAM):
    """Density of pairwise offset differences at the best non-zero lag,
    relative to the density at zero lag. Low for a gate, high for a comb."""
    cc, _, _ = signed_offsets(nvecs, cs, w)
    d = np.abs(cc[:, None] - cc[None, :])
    W = np.outer(w, w); np.fill_diagonal(d, np.nan); np.fill_diagonal(W, 0.0)
    d = d[~np.isnan(d)]; W = W[W > 0] if W.sum() else W
    W = np.outer(w, w)[~np.eye(len(w), dtype=bool)]
    lags = np.linspace(0.0, lag_max, 61)
    dens = np.array([(W * np.exp(-0.5 * ((d - L) / h) ** 2)).sum() for L in lags])
    zero = dens[0]
    if zero <= 0:
        return np.nan, np.nan
    off = lags > 0.6 * PROBE_DIAM          # ignore the shoulder of the zero peak
    if not off.any():
        return np.nan, np.nan
    j = np.argmax(dens[off])
    return float(dens[off][j] / zero), float(lags[off][j])


def interp_N(grid, vals, target):
    """First N at which a monotone-ish curve crosses `target`, by interpolation."""
    g, v = np.array(grid, float), np.array(vals, float)
    for i in range(1, len(v)):
        if v[i] >= target:
            if v[i - 1] >= target:
                return float(g[0]) if i == 1 else float(g[i - 1])
            f = (target - v[i - 1]) / max(v[i] - v[i - 1], 1e-12)
            return float(np.exp(np.log(g[i - 1]) + f * (np.log(g[i]) - np.log(g[i - 1]))))
    return np.nan


def main():
    rng = np.random.default_rng(SEED)
    rows, comb_rows = [], []

    # ---------- task 6: recovery curves + interpolated N50/N95, gate surface
    for dt in [1.5, 2.5, 5.0]:
        nG, cG, wG = make_pool("gate", rng, dt_ratio=dt)
        nH, cH, wH = make_pool("honest", rng)
        if min(len(wG), len(wH)) < 2 * max(N_GRID):
            continue
        pw, oe, ce = [], [], []
        for N in N_GRID:
            sr, sn, errs = [], [], []
            for _ in range(N_REPEAT):
                i = rng.choice(len(wG), N, replace=False)
                j = rng.choice(len(wH), N, replace=False)
                sr.append(P.estimator_C(nG[i], cG[i], wG[i]))
                sn.append(P.estimator_C(nH[j], cH[j], wH[j]))
                nh, ch = P.point_estimate(nG[i], cG[i], wG[i])
                errs.append((np.degrees(np.arccos(np.clip(abs(nh @ N_TRUE), -1, 1))),
                             abs(ch)))
            thr = np.quantile(sn, 1 - ALPHA)
            e = np.array(errs)
            pw.append(float(np.mean(np.array(sr) > thr)))
            oe.append(float(np.median(e[:, 0]))); ce.append(float(np.median(e[:, 1])))
            rows.append(dict(dt_ratio=dt, N=N, power=pw[-1],
                             orient_err=oe[-1], offset_err=ce[-1]))
        print(f"  gate dt={dt}: N50={interp_N(N_GRID, pw, .50):.1f} "
              f"N95={interp_N(N_GRID, pw, .95):.1f} | "
              f"N for orient<5deg={interp_N(N_GRID, -np.array(oe), -5.0):.1f} "
              f"| N for offset<0.5sigma={interp_N(N_GRID, -np.array(ce), -0.5*SIGMA):.1f}",
              flush=True)

    # ---------- task 7: the comb test
    print("\n  comb test")
    cells = [("gate dt=2.5", dict(kind="gate", dt_ratio=2.5)),
             ("gate dt=5.0", dict(kind="gate", dt_ratio=5.0)),
             ("honest", dict(kind="honest"))]
    for ell in [0.3, 0.5, 0.8]:
        cells.append((f"resonant ell={ell}", dict(kind="resonant", amp=2.5 * TAU, ell=ell)))
    for label, kw in cells:
        kind = kw.pop("kind")
        nv, cv, wv = make_pool(kind, rng, **kw)
        if len(wv) < 60:
            continue
        sc, lg, Rs, fires = [], [], [], []
        nH, cH, wH = make_pool("honest", rng, n_pool=150)
        null_stat = [P.estimator_C(*[a[rng.choice(len(wH), 50, replace=False)]
                                     for a in (nH, cH, wH)]) for _ in range(200)]
        thr = np.quantile(null_stat, 1 - ALPHA)
        for _ in range(200):
            i = rng.choice(len(wv), min(50, len(wv)), replace=False)
            s, L = comb_score(nv[i], cv[i], wv[i])
            _, _, R = signed_offsets(nv[i], cv[i], wv[i])
            sc.append(s); lg.append(L); Rs.append(R)
            fires.append(P.estimator_C(nv[i], cv[i], wv[i]) > thr)
        comb_rows.append(dict(surface=label, comb_score=float(np.nanmedian(sc)),
                              best_lag=float(np.nanmedian(lg)),
                              dir_coherence=float(np.median(Rs)),
                              C_fires=float(np.mean(fires))))
        print(f"    {label:18s} comb={np.nanmedian(sc):.3f}  lag={np.nanmedian(lg):.2f}  "
              f"dir coherence R={np.median(Rs):.3f}  C fires={np.mean(fires):.3f}",
              flush=True)

    pd.DataFrame(rows).to_csv("/tmp/pexp/p_task6_rows.csv", index=False)
    pd.DataFrame(comb_rows).to_csv("/tmp/pexp/p_comb_rows.csv", index=False)
    print("\nwrote p_task6_rows.csv, p_comb_rows.csv")


if __name__ == "__main__":
    main()
