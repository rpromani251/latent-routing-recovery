"""
Why t_hat is biased, and two repairs.

The axis-dominance run recovers t_hat ~ 4.88 against a planted 5.0, with a bias that does
NOT shrink with N -- so the bootstrap interval tightens around a displaced centre and
coverage collapses. Bias survives pooling; that is exactly why step 2 measured the
bias/variance split for the normal. Nobody has done the same for the OFFSET.

TWO CANDIDATE DEFECTS, and they are independent.

  (1) ORIGIN AMPLIFICATION -- a multiplicative bias.
      The per-anchor offset is c_a = nu_a . t_a + t0_a, using that anchor's OWN normal.
      Write t_a = (T +- d) nu_true + along * tangent and let phi_a be the anchor's
      orientation error. Then

          nu_a . t_a  =  (T +- d) cos(phi_a)  +  along * sin(phi_a)

      The second term averages away. The first does not: E[cos phi] < 1, so the pooled
      offset is ATTENUATED TOWARD THE ORIGIN by a factor E[cos phi], and the absolute
      bias grows with T. Experiment P measured the offset at c_true = 0, where a
      multiplicative bias is invisible by construction -- 0 times anything is 0.

  (2) LDA MIDPOINT -- an additive bias.
      t0 is the midpoint between the two projected class means. For a half-space cut of
      a Gaussian at distance d, those means are -sigma phi/Phi and +sigma phi/(1-Phi),
      and their midpoint is NOT d unless the split is balanced. At pi = 0.10 it sits at
      0.780 sigma against a true 1.282 sigma. It cancels between anchors on opposite
      sides of the gate, so it costs variance rather than bias -- unless placement is
      one-sided.

THREE ESTIMATORS COMPARED.
  A  current      c_hat = sum_a w_a sg_a (nu_a . t_a + t0_a) / sum w
  B  pooled-normal projection: form each anchor's estimated boundary POINT
                    p_a = t_a + t0_a nu_a,  then c_hat = sum_a w_a (n_hat . p_a) / sum w
                  The big T term is now projected on the POOLED normal, so per-anchor
                  rotation error is no longer multiplied by |t_a|.
  C  B, with the LDA's distance replaced by the validated crossing-law estimator
                    d_hat_a = -sigma Phi^-1(pi_hat_a),  sign taken from t0_a
                  i.e. direction from separability (validated unbiased), distance from
                  the crossing law (validated r = 0.996). This is the pairing v5 sec.11, Stage 5(c),
                  already describes as two routes to the offset -- but calls a
                  specification check, on the grounds that they agree "by construction
                  under a correctly specified model".

The T sweep is the discriminator: defect (1) scales with T, defect (2) does not.
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

SIGMA, TAU, M_PROBE = 0.20, 0.02, 600
PI_TARGET = 0.10
BETA = 0.15 * np.array([np.cos(2.1), np.sin(2.1)])
NU_TRUE = np.array([1.0, 0.0])
TG = np.array([-NU_TRUE[1], NU_TRUE[0]])
POOL, N_POOL_DRAW = 200, 200
T_VALUES = [0.0, 1.0, 5.0, 20.0]
DT_VALUES = [1.5, 2.5, 5.0]
SEED = 20260814


def one_anchor(dt_ratio, T_true, rng):
    delta = dt_ratio * TAU
    dist = -SIGMA * norm.ppf(PI_TARGET) * rng.uniform(0.95, 1.05)
    side = rng.choice([-1.0, 1.0])
    along = rng.normal(0.0, 4.0 * SIGMA)
    t_a = NU_TRUE * (side * dist + T_true) + TG * along

    Z = rng.normal(0.0, SIGMA, size=(M_PROBE, 2))
    T = t_a[None, :] + Z
    y = T @ BETA - delta * ((T @ NU_TRUE) > T_true) + rng.normal(0.0, TAU, M_PROBE)

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
    d_hat = -SIGMA * norm.ppf(pi_hat)                  # magnitude, from the crossing law
    return dict(nu=nu, t0=float(t0), t_a=t_a, w=float(max(fit["lrt"], 0.0)),
                pi_hat=pi_hat, d_hat=float(d_hat), side=float(side),
                phi=float(np.degrees(np.arccos(np.clip(abs(nu @ NU_TRUE), -1, 1)))))


def pool_three(rec, idx):
    """Return (c_A, c_B, c_C) for a subsample of anchors."""
    nv = np.array([rec[i]["nu"] for i in idx])
    w = np.array([rec[i]["w"] for i in idx])
    t0 = np.array([rec[i]["t0"] for i in idx])
    ta = np.array([rec[i]["t_a"] for i in idx])
    dh = np.array([rec[i]["d_hat"] for i in idx])

    nhat, _ = P.pool_direction(nv, w)
    sg = np.sign(nv @ nhat); sg[sg == 0] = 1

    c_a = (nv * ta).sum(1) + t0                        # A: per-anchor normal
    cA = float((w * sg * c_a).sum() / max(w.sum(), 1e-12))

    p_lda = ta + t0[:, None] * nv                      # B: estimated boundary point
    cB = float((w * (p_lda @ nhat)).sum() / max(w.sum(), 1e-12))

    p_dist = ta + (np.sign(t0) * dh)[:, None] * nv     # C: distance from the crossing law
    cC = float((w * (p_dist @ nhat)).sum() / max(w.sum(), 1e-12))
    return cA, cB, cC


def main():
    d_true_sig = -norm.ppf(PI_TARGET)
    phi_d = norm.pdf(d_true_sig)
    mid = 0.5 * (phi_d / PI_TARGET - phi_d / (1 - PI_TARGET))
    print(f"ANALYTIC (defect 2), pi = {PI_TARGET}:")
    print(f"  true distance {d_true_sig:.4f} sigma,  LDA midpoint {mid:.4f} sigma,"
          f"  error {mid - d_true_sig:+.4f} sigma = {(mid-d_true_sig)*SIGMA:+.4f}\n")

    rows = []
    for dt in DT_VALUES:
        for T_true in T_VALUES:
            rng = np.random.default_rng(SEED + int(dt * 10) + int(T_true * 7))
            rec = [r for r in (one_anchor(dt, T_true, rng) for _ in range(POOL))
                   if r is not None]
            if len(rec) < 100:
                continue
            phis = np.array([r["phi"] for r in rec])
            est = {"A": [], "B": [], "C": []}
            for _ in range(N_POOL_DRAW):
                idx = rng.choice(len(rec), 100, replace=False)
                a, b, c = pool_three(rec, idx)
                est["A"].append(a); est["B"].append(b); est["C"].append(c)
            row = dict(dt=dt, T_true=T_true, n_anchor=len(rec),
                       phi_med=float(np.median(phis)),
                       cos_phi_mean=float(np.mean(np.cos(np.radians(phis)))))
            for k in "ABC":
                v = np.abs(np.array(est[k]))            # sign convention absorbed
                row[f"c_{k}"] = float(np.median(v))
                row[f"bias_{k}"] = float(np.median(v) - T_true)
            rows.append(row)
            print(f"  dt={dt:<4} T={T_true:<5} per-anchor |phi| med {row['phi_med']:5.1f} deg"
                  f"   bias  A {row['bias_A']:+8.4f}   B {row['bias_B']:+8.4f}"
                  f"   C {row['bias_C']:+8.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "offset_bias_rows.csv"), index=False)

    print("\nIS THE BIAS MULTIPLICATIVE? (defect 1 predicts bias/T = const = E[cos phi] - 1)")
    print(f"  {'dt':>4} {'E[cos phi]-1':>13} | " + " ".join(f"{'T='+str(t):>10}" for t in T_VALUES))
    for dt, g in df.groupby("dt"):
        g = g.set_index("T_true")
        cells = []
        for t in T_VALUES:
            if t in g.index and t > 0:
                cells.append(f"{g.loc[t,'bias_A']/t:>10.4f}")
            else:
                cells.append(f"{'--':>10}")
        print(f"  {dt:>4} {g.cos_phi_mean.iloc[0]-1:>13.4f} | " + " ".join(cells))

    print("\n  If the A column matches E[cos phi] - 1 and is constant across T, the")
    print("  current offset is attenuated toward the origin -- a bias no amount of")
    print("  pooling removes, and one that Experiment P could not have seen at c = 0.")
    print("\nwrote offset_bias_rows.csv")


if __name__ == "__main__":
    main()
