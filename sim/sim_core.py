"""
Shared primitives for the known-regimes simulation suite.

Implements the SIMPLIFIED audit protocol (RESULTS_2026-07-28 sections 6c-6e):
  - naive multi-scale dip scan: dip at each of a few log-spaced scales,
    Bonferroni over the scales actually dipped (no dispersion curve, no sigma*)
  - per-scale minimum-signal rule: a scale is dipped only when the response
    variance clears the observation-noise floor (chi-square test)
  - K=2 Gaussian-mixture recovery on a FRESH draw per scale: recovered gap
    Delta_hat, soft composition pi_hat, anchor orientation z_hat
  - v2 minimum-mass rule: recovery reported only when the minor component
    holds >= NMIN_SOFT points of soft mass
  - abstention when no scale has signal or support

Ground truth follows the corrected criterion: pi_true(sigma) = minority
fraction of probe points across the gate; an anchor is DETECTABLE when
max_sigma pi_true >= PI_DETECTABLE. Truth is the mixing fraction, not distance.

No sklearn dependency: 1-D K=2 GMM via EM (deterministic given rng).
"""
import numpy as np
from math import erf, sqrt
from dip import dip_pvalue

try:                                   # scipy if present, else exact-enough
    from scipy import stats as _st     # closed forms (Wilson-Hilferty etc.)
    def _chi2_sf(x, k): return _st.chi2.sf(x, k)
except ImportError:
    def _chi2_sf(x, k):
        # Wilson-Hilferty cube-root normal approximation; excellent for k >> 30
        z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / sqrt(2.0 / (9.0 * k))
        return 0.5 * (1.0 - erf(z / sqrt(2.0)))

_LOG2PI = np.log(2.0 * np.pi)

def _norm_logpdf(y, mu, sd):
    return -0.5 * ((y - mu) / sd) ** 2 - np.log(sd) - 0.5 * _LOG2PI

ALPHA = 0.05
NMIN_SOFT = 20
PI_DETECTABLE = 0.05


# ----------------------------------------------------------------- 1-D K=2 EM
def gmm2_fit(y, rng, n_init=3, max_iter=200, tol=1e-7):
    """K=2 1-D Gaussian mixture by EM. Returns dict(mu, sd, w, resp, loglik),
    components ordered by mean (index 0 = lower)."""
    y = np.asarray(y, float)
    n = y.size
    best = None
    q = np.quantile(y, [0.25, 0.75])
    inits = [np.array(q)] + [np.sort(rng.choice(y, 2, replace=False))
                             for _ in range(n_init - 1)]
    for mu0 in inits:
        mu = mu0.astype(float).copy()
        sd = np.full(2, max(y.std(), 1e-9))
        w = np.array([0.5, 0.5])
        ll_old = -np.inf
        for _ in range(max_iter):
            logp = (np.log(w)[None, :]
                    + _norm_logpdf(y[:, None], mu[None, :], sd[None, :]))
            m = logp.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))
            r = np.exp(logp - lse[:, None])
            ll = float(lse.sum())
            nk = r.sum(axis=0) + 1e-12
            w = nk / n
            mu = (r * y[:, None]).sum(axis=0) / nk
            sd = np.sqrt((r * (y[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk)
            sd = np.maximum(sd, 1e-9)
            if ll - ll_old < tol * max(1.0, abs(ll)):
                break
            ll_old = ll
        if best is None or ll > best["loglik"]:
            best = dict(mu=mu, sd=sd, w=w, loglik=ll)
    order = np.argsort(best["mu"])
    mu, sd, w = best["mu"][order], best["sd"][order], best["w"][order]
    logp = np.log(w)[None, :] + _norm_logpdf(y[:, None], mu[None, :], sd[None, :])
    m = logp.max(axis=1, keepdims=True)
    r = np.exp(logp - m)
    r /= r.sum(axis=1, keepdims=True)
    return dict(mu=mu, sd=sd, w=w, resp=r, loglik=best["loglik"])


def gmm2_posterior(fit, y0):
    """Posterior component probabilities of a scalar response under a fitted GMM."""
    logp = (np.log(fit["w"])
            + _norm_logpdf(y0, fit["mu"], fit["sd"]))
    p = np.exp(logp - logp.max())
    return p / p.sum()


# ------------------------------------------------------------- audit primitive
def audit_anchor(query, gate_ind, x0, scales, rng,
                 m_dip=1000, m_rec=1000, tau_obs=0.02, alpha=ALPHA,
                 probe=None):
    """Naive-scan audit of one anchor.

    query(X)     -> noiseless model responses at inputs X (n, d)
    gate_ind(X)  -> bool array, true regime indicator (GROUND TRUTH ONLY)
    x0           -> anchor location, shape (d,)
    probe(x0, sigma, n, rng) -> probe points; default ambient Gaussian

    Returns dict with flag, p_min, n_scales, delta_med, delta_cv, pi_hat,
    z_hat, pi_true_max, detectable, abstain.
    """
    x0 = np.atleast_1d(np.asarray(x0, float))
    d = x0.size
    if probe is None:
        def probe(x0_, s_, n_, rng_):
            return x0_[None, :] + rng_.normal(0.0, s_, size=(n_, d))

    dp, dl, pt, pih = [], [], [], []
    z_votes = []
    dl_by_scale = {}          # scale index -> Delta_hat, for delta_at_pmin
    dp_by_scale = {}
    for s in scales:
        X = probe(x0, s, m_dip, rng)
        if X is None:
            continue
        # ground truth is recorded for every attempted scale, independent of
        # whether the audit itself abstains at this scale
        f = float(np.mean(gate_ind(X)))
        pt.append(min(f, 1.0 - f))
        y = query(X) + rng.normal(0.0, tau_obs, len(X))
        # per-scale minimum-signal rule: variance must clear the noise floor
        lam = np.var(y, ddof=1) / tau_obs ** 2
        if _chi2_sf((len(y) - 1) * lam, len(y) - 1) >= 0.05:
            continue
        _, p = dip_pvalue(y)
        dp.append(p)
        dp_by_scale[len(dp) - 1] = p

        Xr = probe(x0, s, m_rec, rng)              # fresh draw for recovery
        if Xr is None:
            continue
        y2 = query(Xr) + rng.normal(0.0, tau_obs, len(Xr))
        fit = gmm2_fit(y2, rng)
        if min(fit["w"]) * m_rec >= NMIN_SOFT:     # v2 minimum-mass rule
            dl.append(float(fit["mu"][1] - fit["mu"][0]))
            dl_by_scale[len(dp) - 1] = dl[-1]
            pih.append(float(fit["resp"][:, 0].mean()))
            y0 = query(x0[None, :])[0] + rng.normal(0.0, tau_obs)
            z_votes.append(int(np.argmax(gmm2_posterior(fit, y0))))

    n_dipped = len(dp)
    ptm = float(np.max(pt)) if pt else np.nan
    detectable = bool(np.isfinite(ptm) and ptm >= PI_DETECTABLE)
    out = dict(n_scales=n_dipped)
    if n_dipped == 0:
        out.update(flag=False, p_min=np.nan, delta_med=np.nan, delta_cv=np.nan,
                   pi_hat=np.nan, z_hat=-1, pi_true_max=ptm,
                   detectable=detectable, abstain=True)
        return out
    dp = np.asarray(dp)
    out["p_min"] = float(dp.min())
    out["flag"] = bool(dp.min() < alpha / n_dipped)      # Bonferroni over scales
    fin = np.asarray(dl, float)
    out["delta_med"] = float(np.median(fin)) if fin.size else np.nan
    out["delta_cv"] = (float(np.std(fin) / (np.mean(fin) + 1e-12))
                       if fin.size >= 2 else np.nan)
    out["pi_hat"] = float(np.median(pih)) if pih else np.nan
    out["z_hat"] = (int(np.round(np.mean(z_votes))) if z_votes else -1)
    # Delta_hat at the most significant scale: the mixture and the dip read the
    # same bimodality, so the scale that minimizes the dip p is where the gap
    # is best resolved; medianing over non-crossing scales dilutes Delta_hat.
    k_min = int(np.argmin(dp))
    out["delta_at_pmin"] = float(dl_by_scale.get(k_min, np.nan))
    out["pi_true_max"] = ptm
    out["detectable"] = detectable
    out["abstain"] = False
    return out


# ------------------------------------------------------------------ GP nulls
def rff_gp_path(d, amp, ell, n_feat, rng):
    """Random-Fourier-feature draw of a stationary GP path (RBF kernel)."""
    W = rng.normal(0.0, 1.0 / ell, size=(n_feat, d))
    b = rng.uniform(0.0, 2 * np.pi, size=n_feat)
    a = rng.normal(0.0, 1.0, size=n_feat)
    c = amp * np.sqrt(2.0 / n_feat)

    def g(X):
        return c * (np.cos(X @ W.T + b[None, :]) @ a)
    return g


# ------------------------------------------------------------------- metrics
def summarize(cellrows):
    """Aggregate a list of audit_anchor outputs (+ 'model' key: gated/honest)."""
    import pandas as pd
    df = pd.DataFrame(cellrows)
    g = df[df.model == "gated"]
    h = df[df.model == "honest"]
    gd = g[g.detectable]
    res = dict(
        n_gated=len(g), n_honest=len(h), n_detectable=len(gd),
        power=float(gd.flag.mean()) if len(gd) else np.nan,
        fp_honest=float(h.flag.mean()) if len(h) else np.nan,
        fp_no_crossing=float(g[~g.detectable].flag.mean())
        if len(g[~g.detectable]) else np.nan,
        abstain_gated=float(g.abstain.mean()) if len(g) else np.nan,
        abstain_honest=float(h.abstain.mean()) if len(h) else np.nan,
    )
    fl = g[g.flag & np.isfinite(g.delta_med)]
    if len(fl):
        res["delta_med"] = float(fl.delta_med.median())
        res["delta_q25"] = float(fl.delta_med.quantile(0.25))
        res["delta_q75"] = float(fl.delta_med.quantile(0.75))
    else:
        res["delta_med"] = res["delta_q25"] = res["delta_q75"] = np.nan
    fp = g[g.flag & np.isfinite(g.get("delta_at_pmin", np.nan))]         if "delta_at_pmin" in g else fl
    if len(fp):
        res["delta_pmin_med"] = float(fp.delta_at_pmin.median())
        res["delta_pmin_q25"] = float(fp.delta_at_pmin.quantile(0.25))
        res["delta_pmin_q75"] = float(fp.delta_at_pmin.quantile(0.75))
    else:
        res["delta_pmin_med"] = res["delta_pmin_q25"] = res["delta_pmin_q75"] = np.nan
    hf = h[np.isfinite(h.delta_med)]
    res["delta_honest"] = float(hf.delta_med.median()) if len(hf) else np.nan
    return res
