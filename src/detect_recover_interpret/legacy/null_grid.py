"""
Anchor-specific null via a precomputed lookup grid.
SUPERSEDED — see legacy/README.md. Kept for provenance; not used by run_all.sh.

KEY REDUCTION
-------------
Write the normalized dispersion curve

    q(sigma) = log( r(sigma) * tau_obs / beta ).

Under the registered null (linear mean beta.delta + stationary SE-GP amplitude s,
lengthscale ell) the response variance over a Gaussian probe of scale sigma is

    Var(sigma) = sigma^2 beta^2 + s^2 ( 1 - (1 + 2 sigma^2/ell^2)^(-d/2) )

so q depends on (beta, s) only through the ratio rho = s/beta:

    q(sigma) = 0.5 log( 1 + rho^2 (1 - (1+2 sigma^2/ell^2)^(-d/2)) / sigma^2 ).

Two parameters, not three. So the null mean mu_0 and the GP-realization spread
s_GP can be simulated ONCE on a (log rho, log ell) grid and interpolated per anchor,
instead of running B replicates at every anchor.

SPREAD HAS TWO SOURCES, and they are handled separately:
  s_GP(sigma)  realization-to-realization wiggle of the GP sample path.
               Simulated on the grid with observation noise OFF, so it is pure
               GP variability and carries no SNR dependence.
  s_MC(sigma)  Monte-Carlo error in estimating the variance from m probes.
               Analytic: Var(log lam) ~ 2/(m-1), and d log r/d lam = 0.5/(lam-1),
               giving s_MC = 0.5 * lam * sqrt(2/(m-1)) / (lam - 1).
               This IS anchor-specific -- it blows up as lam -> 1, i.e. exactly at
               low-signal scales, which is where a naive null would over-flag.

    s_0(sigma)^2 = s_GP(sigma)^2 + s_MC(sigma)^2

ell_min is REGISTERED. With the lengthscale floored, a GP cannot mimic a sharp gate,
so the fit residual at a gated anchor is the signal rather than something the null
absorbs. Report a sensitivity sweep over it.
"""
import numpy as np

from ..gp_utils import rff_gp_path

__all__ = ["q_analytic", "build_grid", "s_mc", "studentize_anchor"]


def q_analytic(sigmas, rho, ell, d):
    sig2 = np.asarray(sigmas, float) ** 2
    inner = 1.0 + rho**2 * (1.0 - (1.0 + 2.0 * sig2 / ell**2) ** (-d / 2.0)) / sig2
    return 0.5 * np.log(np.maximum(inner, 1e-300))


def build_grid(sigmas, d, m, rho_grid, ell_grid, n_rep, rng, n_feat=128):
    """Simulate GP-realization spread of q on the (rho, ell) grid, noise off."""
    sigmas = np.asarray(sigmas, float)
    T = len(sigmas)
    mu = np.zeros((len(rho_grid), len(ell_grid), T))
    sd = np.zeros_like(mu)
    for i, rho in enumerate(rho_grid):
        for j, ell in enumerate(ell_grid):
            Q = np.empty((n_rep, T))
            for b in range(n_rep):
                g = rff_gp_path(d, rho, ell, n_feat, rng)
                u = rng.normal(size=d); u /= np.linalg.norm(u) + 1e-12
                for t, s in enumerate(sigmas):
                    delta = rng.normal(0.0, s, size=(m, d))
                    y = delta @ u + g(delta)          # beta = 1, tau = 1, no noise
                    Q[b, t] = 0.5 * np.log(max(np.var(y, ddof=1), 1e-300)) - np.log(s)
            mu[i, j] = Q.mean(0)
            sd[i, j] = Q.std(0, ddof=1)
    return {"mu": mu, "sd": sd, "rho": np.asarray(rho_grid, float),
            "ell": np.asarray(ell_grid, float), "sigmas": sigmas, "d": d, "m": m}


def _bilinear(grid, rho, ell):
    lr, le = np.log(grid["rho"]), np.log(grid["ell"])
    x = np.clip(np.log(rho), lr[0], lr[-1])
    y = np.clip(np.log(ell), le[0], le[-1])
    i = int(np.clip(np.searchsorted(lr, x) - 1, 0, len(lr) - 2))
    j = int(np.clip(np.searchsorted(le, y) - 1, 0, len(le) - 2))
    tx = (x - lr[i]) / (lr[i + 1] - lr[i])
    ty = (y - le[j]) / (le[j + 1] - le[j])

    def bl(A):
        return ((1 - tx) * (1 - ty) * A[i, j] + tx * (1 - ty) * A[i + 1, j]
                + (1 - tx) * ty * A[i, j + 1] + tx * ty * A[i + 1, j + 1])

    return bl(grid["mu"]), bl(grid["sd"])


def s_mc(lam, m):
    """Monte-Carlo component of the spread of q, from the chi-square sampling law."""
    lam = np.asarray(lam, float)
    excess = np.maximum(lam - 1.0, 1e-12)
    return 0.5 * lam * np.sqrt(2.0 / (m - 1)) / excess


def studentize_anchor(r_obs, lam, valid, sigmas, tau_obs, hyper, grid, m,
                      shrink_to=None, shrink_w=0.0):
    """T_scale at one anchor against its own fitted null, plus a p-value.

    shrink_to / shrink_w implement the hierarchical regularization: the anchor's own
    (log rho, log ell) is pulled toward the population mean, which stabilizes noisy
    per-anchor fits without letting the population replace them.
    """
    if hyper is None or valid.sum() < 4:
        return np.nan, np.nan, np.nan
    beta, s, ell = hyper["beta"], hyper["s"], hyper["ell"]
    if beta <= 0 or s <= 0:
        return np.nan, np.nan, np.nan
    rho = s / beta
    if shrink_to is not None and shrink_w > 0:
        lg = np.array([np.log(rho), np.log(ell)])
        lg = (1 - shrink_w) * lg + shrink_w * np.asarray(shrink_to, float)
        rho, ell = np.exp(lg)

    mu, sd_gp = _bilinear(grid, rho, ell)
    q_obs = np.log(np.asarray(r_obs, float) * tau_obs / beta + 1e-300)
    sd = np.sqrt(sd_gp**2 + s_mc(lam, m) ** 2)

    v = valid & np.isfinite(q_obs) & np.isfinite(mu) & (sd > 0)
    if v.sum() < 4:
        return np.nan, np.nan, np.nan
    z = (q_obs[v] - mu[v]) / sd[v]
    t = float(np.max(z))
    return t, float(np.argmax(z)), rho
