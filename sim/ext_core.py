"""
Shared primitives for the future-work extension experiments (E1-E4).

Adds to sim_core:
  gmmk_fit / bic_select_k   general-K 1-D Gaussian mixture EM + BIC selection
  ols_residualize           local-linear residualization of responses on the
                            probe displacement (the operator that both removes
                            off-axis smooth spread in high-D probing, E1, and
                            restores A12 validity on clumpy manifolds, E4)
  dip_bonferroni            dip a list of 1-D samples, Bonferroni-corrected
  KnnProbe                  on-manifold probe over an arbitrary point cloud
"""
import numpy as np
from dip import dip_pvalue
from sim_core import gmm2_fit, _norm_logpdf, _chi2_sf

ALPHA = 0.05


# --------------------------------------------------------------- general-K EM
def gmmk_fit(y, K, rng, n_init=3, max_iter=300, tol=1e-7):
    """K-component 1-D Gaussian mixture by EM; components ordered by mean."""
    y = np.asarray(y, float)
    n = y.size
    if K == 1:
        mu, sd = np.array([y.mean()]), np.array([max(y.std(), 1e-9)])
        ll = float(_norm_logpdf(y, mu[0], sd[0]).sum())
        return dict(mu=mu, sd=sd, w=np.array([1.0]), loglik=ll,
                    resp=np.ones((n, 1)))
    best = None
    qs = np.quantile(y, np.linspace(0.1, 0.9, K))
    inits = [qs] + [np.sort(rng.choice(y, K, replace=False))
                    for _ in range(n_init - 1)]
    for mu0 in inits:
        mu = np.asarray(mu0, float).copy()
        sd = np.full(K, max(y.std() / K, 1e-9))
        w = np.full(K, 1.0 / K)
        ll_old = -np.inf
        for _ in range(max_iter):
            logp = np.log(w)[None, :] + _norm_logpdf(y[:, None], mu[None, :], sd[None, :])
            m = logp.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))
            r = np.exp(logp - lse[:, None])
            ll = float(lse.sum())
            nk = r.sum(axis=0) + 1e-12
            w = nk / n
            mu = (r * y[:, None]).sum(axis=0) / nk
            sd = np.maximum(np.sqrt((r * (y[:, None] - mu[None, :]) ** 2)
                                    .sum(axis=0) / nk), 1e-9)
            if ll - ll_old < tol * max(1.0, abs(ll)):
                break
            ll_old = ll
        if best is None or ll > best["loglik"]:
            best = dict(mu=mu, sd=sd, w=w, loglik=ll)
    order = np.argsort(best["mu"])
    mu, sd, w = best["mu"][order], best["sd"][order], best["w"][order]
    logp = np.log(w)[None, :] + _norm_logpdf(y[:, None], mu[None, :], sd[None, :])
    m = logp.max(axis=1, keepdims=True)
    r = np.exp(logp - m); r /= r.sum(axis=1, keepdims=True)
    return dict(mu=mu, sd=sd, w=w, resp=r,
                loglik=best["loglik"])


def bic_select_k(y, rng, k_max=4, min_mass=20):
    """BIC-selected K; components below the soft-mass floor disqualify a fit."""
    n = len(y)
    best_k, best_fit, best_bic = 1, None, np.inf
    for K in range(1, k_max + 1):
        fit = gmmk_fit(y, K, rng)
        if K > 1 and (fit["w"].min() * n) < min_mass:
            continue                        # minimum-mass rule (v2 B4)
        bic = -2.0 * fit["loglik"] + (3 * K - 1) * np.log(n)
        if bic < best_bic:
            best_k, best_fit, best_bic = K, fit, bic
    return best_k, best_fit


# ---------------------------------------------------------- residualization
def ols_residualize(y, X, x0):
    """Residuals of y on [1, delta]: removes the locally linear part of the
    smooth response over the probe cloud. A jump survives (a step is not
    linear); off-axis smooth spread and manifold-induced trend do not."""
    d = X - np.atleast_2d(x0)
    Z = np.column_stack([np.ones(len(X)), d])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


def trimmed_residualize(y, X, x0, trim=0.25, iters=2):
    """Trimmed OLS residualization: fit, drop the largest-|residual| fraction,
    refit on the rest, residualize ALL points. Keeps the fitted line on the
    majority branch, so a minority-branch jump survives at ~full size
    (plain OLS absorbs ~64% of the gap at balanced mixing -- E1 finding)."""
    d = X - np.atleast_2d(x0)
    Z = np.column_stack([np.ones(len(X)), d])
    keep = np.ones(len(y), bool)
    for _ in range(iters):
        beta, *_ = np.linalg.lstsq(Z[keep], y[keep], rcond=None)
        r = y - Z @ beta
        thr = np.quantile(np.abs(r), 1.0 - trim)
        keep = np.abs(r) <= thr
    beta, *_ = np.linalg.lstsq(Z[keep], y[keep], rcond=None)
    return y - Z @ beta


# ------------------------------------------------------------- dip utilities
def min_signal_ok(y, tau_obs, m=None):
    m = m or len(y)
    lam = np.var(y, ddof=1) / tau_obs ** 2
    return _chi2_sf((m - 1) * lam, m - 1) < 0.05


def dip_bonferroni(samples, alpha=ALPHA):
    """Dip each 1-D sample; flag if min p < alpha / (#samples). Returns
    (flag, p_min, index of the most significant sample)."""
    ps = [dip_pvalue(np.asarray(s, float))[1] for s in samples]
    if not ps:
        return False, np.nan, -1
    k = int(np.argmin(ps))
    return bool(ps[k] < alpha / len(ps)), float(ps[k]), k


# ------------------------------------------------------- on-manifold probing
class KnnProbe:
    """Kernel-weighted resampling of a reference point cloud (v2 sec 2.2)."""
    def __init__(self, pts, min_neigh=12):
        self.pts = np.asarray(pts, float)          # brute-force; no KD-tree dep
        self.min_neigh = min_neigh

    def draw(self, x0, sigma, n, rng):
        d2 = ((self.pts - np.atleast_2d(x0)) ** 2).sum(1)
        near = d2 <= (4.0 * sigma) ** 2
        if near.sum() < self.min_neigh:
            return None
        idx = np.where(near)[0]
        w = np.exp(-0.5 * d2[idx] / sigma ** 2)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return None
        pick = rng.choice(idx, size=n, replace=True, p=w / s)
        return self.pts[pick]
