"""
Distance-to-boundary estimator: d_hat = -sigma * Phi^{-1}(pi_hat_cross)

Tests the inversion of the E1(a) crossing law, using the registered 1-D
known-regimes setting (SLOPE 0.15, DELTA 0.30, gate at x = 0.5,
ladder geomspace(0.02, 0.2, 3), m_dip = m_rec = 1000, alpha 0.05).

Records PER SCALE (the existing sim only stored a median across scales):
  sigma, dip p, Delta_hat, w, pi_hat_cross, d_hat, pi_true, true d

pi_hat_cross is oriented query-only: the anchor's own response y0 says which
component it sits in; the crossing mass is the OTHER component's weight.

Models:
  gated        f = SLOPE*x - DELTA*1[x >= 0.5]      -- the estimator should work
  honest       f = SLOPE*x                          -- no boundary; control
  gp_*_bad     honest + GP path, ell < ladder top   -- A11-violating confounder
  gp_*_ok      honest + GP path, ell > ladder top   -- precondition satisfied
  kink         continuous, non-routed               -- sharp but not a step

Prediction (pre-stated):
  P1  On gated anchors with pi_hat in [0.05, 0.5], d_hat tracks true distance.
  P2  d_hat is approximately CONSTANT ACROSS RUNGS for the gated model.
  P3  On smooth confounders that produce spurious mixtures, the implied
      distance WANDERS across rungs (higher CV than gated).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sim_core import gmm2_fit, gmm2_posterior, rff_gp_path, _chi2_sf, NMIN_SOFT
from dip import dip_pvalue

SLOPE, DELTA, GATE = 0.15, 0.30, 0.5
SCALES = np.geomspace(0.02, 0.2, 3)
M = 1000
ALPHA = 0.05
SEED = 20260813


def make_models():
    m = {}
    m["gated"] = lambda X: SLOPE * X[:, 0] - DELTA * (X[:, 0] >= GATE)
    m["honest"] = lambda X: SLOPE * X[:, 0]
    m["kink"] = lambda X: SLOPE * X[:, 0] + 1.5 * np.maximum(X[:, 0] - GATE, 0.0)
    for name, amp, ell in [("gp_A.10_l.30_ok", 0.10, 0.30),
                           ("gp_A.10_l.05_bad", 0.10, 0.05),
                           ("gp_A.20_l.05_bad", 0.20, 0.05)]:
        g = rff_gp_path(1, amp, ell, 256, np.random.default_rng(abs(hash(name)) % 2**31))
        m[name] = (lambda gg: (lambda X: SLOPE * X[:, 0] + gg(X)))(g)
    return m


def audit_per_scale(f, x0, scales, rng, tau_obs):
    """One anchor. Returns one row per scale that survives the guards."""
    rows = []
    for s in scales:
        X = x0 + rng.normal(0.0, s, size=(M, 1))
        pi_true = float(np.mean(X[:, 0] >= GATE))
        pi_true = min(pi_true, 1.0 - pi_true)
        y = f(X) + rng.normal(0.0, tau_obs, M)

        # per-scale minimum-signal rule
        lam = np.var(y, ddof=1) / tau_obs ** 2
        if _chi2_sf((M - 1) * lam, M - 1) >= 0.05:
            continue
        _, p = dip_pvalue(y)

        Xr = x0 + rng.normal(0.0, s, size=(M, 1))          # fresh recovery draw
        y2 = f(Xr) + rng.normal(0.0, tau_obs, M)
        fit = gmm2_fit(y2, rng)
        if min(fit["w"]) * M < NMIN_SOFT:                  # minimum-mass rule
            continue

        # orientation, query-only: which component does the anchor itself sit in?
        y0 = f(np.array([[x0]]))[0] + rng.normal(0.0, tau_obs)
        z_anchor = int(np.argmax(gmm2_posterior(fit, y0)))
        pi_cross = float(fit["resp"][:, 1 - z_anchor].mean())

        d_hat = -s * norm.ppf(pi_cross) if 0.0 < pi_cross < 1.0 else np.nan
        rows.append(dict(sigma=s, dip_p=p,
                         delta_hat=float(abs(fit["mu"][1] - fit["mu"][0])),
                         pi_cross=pi_cross, d_hat=d_hat,
                         pi_true=pi_true, w_min=float(min(fit["w"]))))
    return rows


def main():
    models = make_models()
    rng = np.random.default_rng(SEED)
    out = []
    # anchors: dense near the boundary, out to well past the ladder top
    dists = np.concatenate([np.linspace(0.005, 0.10, 40), np.linspace(0.11, 0.35, 25)])
    for mname, f in models.items():
        for tau in ([0.005, 0.02, 0.05] if mname in ("gated", "honest") else [0.02]):
            for d_true in dists:
                for side in (-1, +1):
                    x0 = GATE + side * d_true
                    if not (0.02 < x0 < 0.98):
                        continue
                    for r in audit_per_scale(f, x0, SCALES, rng, tau):
                        r.update(model=mname, tau_obs=tau, anchor=x0,
                                 d_true=d_true, side=side)
                        out.append(r)
    df = pd.DataFrame(out)
    df.to_csv("distance_estimator_rows.csv", index=False)
    print(f"rows: {len(df)}   models: {df.model.nunique()}")
    return df


if __name__ == "__main__":
    main()
