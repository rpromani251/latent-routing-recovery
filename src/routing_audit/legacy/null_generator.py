"""
The null generator (routing_audit_v2 3.3) and the studentized scale statistic (3.2 S2).
SUPERSEDED — see legacy/README.md. Kept for provenance; not used by run_all.sh.

WHY THIS EXISTS
---------------
Shape statistics do not work. Under the null the EXPECTED dispersion curve is monotone
non-increasing, but an individual GP sample path wiggles, and a wiggle at one scale is
indistinguishable in shape from a gate crossing. Only the simulated null's SPREAD
s_0(sigma) tells you how much wiggle is ordinary. That is the whole point of 3.3.

DESIGN
------
1. Per anchor, fit the null hyperparameters (beta, s, ell) to its own dispersion curve
   using the closed form for a linear mean plus a stationary SE-GP under a Gaussian probe:

       Var(sigma) = sigma^2 beta^2 + s^2 ( 1 - (1 + 2 sigma^2/ell^2)^(-d/2) )
       r(sigma)   = sqrt(Var(sigma)) / (sigma * tau_obs)

   ell is bounded below by a REGISTERED ell_min. This bound is what defines "smooth";
   without it a GP fits the gate itself with a tiny lengthscale and the null swallows
   the alternative. It must be declared up front and reported with a sensitivity sweep.

2. Trim the top TRIM fraction by a preliminary statistic, then fit a population law over
   (log beta, log s, log ell) -- the hierarchical part. A few gated anchors cannot drag it.

3. Cross-fit: anchors are split into folds; each fold is tested against a population
   fitted on the others.

4. Simulate B synthetic audits from the population (hyperparameters redrawn each
   replicate, GP path drawn by random Fourier features) to get the null distribution of
   the SHAPE curve.

5. Studentize. Anchors differ in local slope, so we remove scale first: the shape curve
   is c_i(sigma_t) = log r_i(sigma_t) - log r_i(sigma_T). Then

       T_scale_i = max_t ( c_i(sigma_t) - mu_0(sigma_t) ) / s_0(sigma_t)

   and the screening p-value is the null tail probability of T_scale.
"""
import numpy as np
from scipy.optimize import least_squares

from ..dispersion import dispersion_curve
from ..gp_utils import rff_gp_path

__all__ = ["null_r_curve", "fit_null_hyper", "shape_curve", "fit_population",
           "simulate_null_shapes", "studentize"]


def null_r_curve(sigmas, beta, s, ell, tau_obs, d):
    """Closed-form expected dispersion curve for linear mean + stationary SE-GP."""
    sig2 = np.asarray(sigmas, float) ** 2
    var = sig2 * beta**2 + s**2 * (1.0 - (1.0 + 2.0 * sig2 / ell**2) ** (-d / 2.0))
    return np.sqrt(np.maximum(var, 1e-300)) / (np.asarray(sigmas, float) * tau_obs)


def fit_null_hyper(r_obs, sigmas, valid, tau_obs, d, ell_min, ell_max=None):
    """Least squares in log space for (beta, s, ell), with ell >= ell_min."""
    sg = np.asarray(sigmas, float)[valid]
    ro = np.asarray(r_obs, float)[valid]
    if ro.size < 4:
        return None
    ell_max = ell_max if ell_max is not None else 100.0 * float(sg.max())
    lo = np.log([1e-8, 1e-8, ell_min])
    hi = np.log([1e6, 1e6, ell_max])

    def resid(theta):
        beta, s, ell = np.exp(theta)
        return np.log(null_r_curve(sg, beta, s, ell, tau_obs, d) + 1e-300) - np.log(ro + 1e-300)

    best, best_cost = None, np.inf
    for ell0 in (ell_min * 2, float(np.median(sg)), float(sg.max())):
        x0 = np.log([max(ro[-1] * tau_obs, 1e-6),
                     max(ro[0] * tau_obs * sg[0], 1e-6),
                     np.clip(ell0, ell_min * 1.01, ell_max * 0.99)])
        try:
            out = least_squares(resid, x0, bounds=(lo, hi), max_nfev=400)
        except Exception:
            continue
        if out.cost < best_cost:
            best, best_cost = out, out.cost
    if best is None:
        return None
    beta, s, ell = np.exp(best.x)
    return {"beta": float(beta), "s": float(s), "ell": float(ell), "cost": float(best_cost)}


def shape_curve(r, valid, eps=1e-12):
    """log r normalized by its value at the largest valid scale.

    Removes the anchor's local slope, which varies across anchors for reasons that
    have nothing to do with routing, and leaves the shape.
    """
    v = np.where(valid, r, np.nan)
    idx = np.where(valid)[0]
    if idx.size < 4:
        return None
    ref = np.log(v[idx[-1]] + eps)
    c = np.full(len(r), np.nan)
    c[idx] = np.log(v[idx] + eps) - ref
    return c


def fit_population(hypers, trim_stat, trim=0.20):
    """Multivariate normal over (log beta, log s, log ell), fitted on the
    (1 - trim) least extreme anchors by `trim_stat`."""
    keep = [h is not None and np.isfinite(t) for h, t in zip(hypers, trim_stat)]
    idx = np.where(keep)[0]
    if idx.size < 10:
        return None
    ts = np.asarray(trim_stat, float)[idx]
    cut = np.quantile(ts, 1.0 - trim)
    idx = idx[ts <= cut]
    Z = np.array([[np.log(hypers[i]["beta"]), np.log(hypers[i]["s"]), np.log(hypers[i]["ell"])]
                  for i in idx])
    mu = Z.mean(0)
    C = np.cov(Z.T) + 1e-6 * np.eye(3)
    return {"mu": mu, "cov": C, "n": int(idx.size)}


def simulate_null_shapes(pop, sigmas, m, tau_obs, d, n_rep, rng,
                         n_feat=256, ell_min=1e-6):
    """B synthetic audits from the fitted population. Returns (n_rep, T) shape curves."""
    sigmas = np.asarray(sigmas, float)
    L = np.linalg.cholesky(pop["cov"])
    out = []
    for _ in range(n_rep):
        beta, s, ell = np.exp(pop["mu"] + L @ rng.normal(size=3))
        ell = max(ell, ell_min)
        g = rff_gp_path(d, s, ell, n_feat, rng)
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction) + 1e-12

        def f(Z, g=g, beta=beta, direction=direction):
            return Z @ (beta * direction) + g(Z)

        dc = dispersion_curve(f, np.zeros(d), sigmas, m, tau_obs, rng)
        c = shape_curve(dc["r"], dc["valid"])
        if c is not None:
            out.append(c)
    return np.array(out) if out else None


def studentize(c_obs, null_shapes):
    """T_scale = max_t (c_obs - mu_0)/s_0, plus the null distribution of the same
    statistic (leave-one-out over replicates) for the screening p-value."""
    ok = np.isfinite(c_obs) & np.all(np.isfinite(null_shapes), axis=0)
    if ok.sum() < 3:
        return np.nan, np.nan
    C = null_shapes[:, ok]
    mu = C.mean(0)
    sd = C.std(0, ddof=1) + 1e-9
    t_obs = float(np.max((c_obs[ok] - mu) / sd))
    t_null = np.max((C - mu) / sd, axis=1)
    p = (1.0 + np.sum(t_null >= t_obs)) / (len(t_null) + 1.0)
    return t_obs, float(p)
