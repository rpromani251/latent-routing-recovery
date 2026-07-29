"""
Dispersion-curve statistics.

RESULTS_2026-07-28.md's headline finding is that these do NOT discriminate a routing
gate from smooth curvature (S1, S4, S7: "dispersion does not discriminate, modality
does"). They are kept here — not in legacy/ — because they still play the two roles the
results doc confirms actually hold:

  * a toy-validation baseline that motivates why the dip test is the primary statistic
    (see exp_dip_is_the_discriminator equivalent under scripts/run_dip_discriminator_toy.py)
  * a corroborating signal at flagged anchors (precision 0.52 -> 0.81 when required
    alongside the dip, RESULTS S6c Role 3) — weaker and less independent than the
    recovered effect size Delta_hat, but real.

They are NOT used for scale selection or anchor screening in the current pipeline
(RESULTS S6c Roles 1-2 both fail) and `benjamini_hochberg` is currently unused —
the production pipeline uses Bonferroni over the scales actually dipped instead
(see audit.py). The full per-anchor "Stage A" pipeline that used to orchestrate these
into a single decision (dispersion -> sigma* -> single dip) is superseded; see
legacy/stage_a_full.py.
"""
import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

MIN_VALID_SCALES = 4


def dispersion_curve(f, x, sigmas, m, tau_obs, rng, alpha=0.05):
    """r_i(sigma_t) at one anchor under an ambient isotropic Gaussian probe.

    Returns dict with r, lam, per-scale p-values, and the validity mask.
    A scale is valid when its whitened variance significantly exceeds the noise floor.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    d = x.shape[0]
    sigmas = np.asarray(sigmas, dtype=float)
    r = np.zeros(len(sigmas))
    lam = np.zeros(len(sigmas))
    for t, s in enumerate(sigmas):
        delta = rng.normal(0.0, s, size=(m, d))
        y = f(x[None, :] + delta) + rng.normal(0.0, tau_obs, size=m)
        lam[t] = np.var(y, ddof=1) / tau_obs**2
        r[t] = np.sqrt(max(lam[t] - 1.0, 0.0)) / s

    # per-scale one-sided test: under H0, (m-1)*lam ~ chi2(m-1)
    p_scale = stats.chi2.sf((m - 1) * lam, df=m - 1)
    valid = p_scale < alpha
    return {"r": r, "lam": lam, "p_scale": p_scale, "valid": valid, "sigmas": sigmas}


def r_log(r, valid, eps=1e-12):
    """Log-range over VALID scales only."""
    v = r[valid]
    if v.size < MIN_VALID_SCALES:
        return np.nan
    lr = np.log(v + eps)
    return float(lr.max() - lr.min())


def t_iso(r, sigmas, valid, eps=1e-12, increasing=False):
    """Max positive residual from the best monotone fit to log r over valid scales.

    Direction matters, and the correct direction is NON-INCREASING. Under the
    registered null (linear mean beta.delta plus a STATIONARY Matern GP of amplitude s
    and lengthscale ell), the response variance over a Gaussian probe of scale sigma in
    d dims is

        Var(sigma) = sigma^2 ||beta||^2 + s^2 ( 1 - (1 + 2 sigma^2/ell^2)^(-d/2) )

    so, dividing by sigma^2 tau^2,

        r(sigma)^2 tau^2 = ||beta||^2 + s^2 (1 - (1+2 sigma^2/ell^2)^(-d/2)) / sigma^2

    and the second term is monotone DECREASING in sigma (it tends to s^2 d/ell^2 as
    sigma -> 0 and to 0 as sigma -> infinity). So every member of the null class has a
    non-increasing dispersion curve: flat for a linear branch, decaying for a curved one.

    A routing boundary is the only thing in the model class that makes r RISE: the
    crossing fraction p(sigma) = Phi(-s_i/sigma) is exponentially small at small sigma,
    so the jump contributes nothing there, then switches on. Hence

        deviation above a monotone non-increasing envelope = evidence of a boundary.

    NOTE: an unbounded nonlinearity (e.g. a pure quadratic, whose amplitude grows without
    bound) is OUTSIDE this null class and produces a rising curve. That is a
    misspecification case, not a counterexample.
    """
    v = r[valid]
    s = np.asarray(sigmas)[valid]
    if v.size < MIN_VALID_SCALES:
        return np.nan
    lr = np.log(v + eps)
    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    fit = iso.fit_transform(np.log(s), lr)
    return float(np.max(lr - fit))


def sigma_star(r, sigmas, valid):
    """Scale at which the dispersion curve peaks. RESULTS S6c Role 1: this is a WORSE
    scale-selection rule than picking a scale at random (0.715 vs 0.798 power) — kept
    for the toy validation and as an ablation baseline, not used operationally."""
    v = np.where(valid, r, -np.inf)
    return int(np.argmax(v))


def benjamini_hochberg(pvals, q=0.10):
    """BH screening. Currently unused by the production pipeline (RESULTS S8): spatial
    randomization (spatial_randomization.py) is used for the alignment claim instead,
    since anchors are strongly spatially dependent and BH assumes independence/PRDS."""
    p = np.asarray(pvals, dtype=float)
    idx = np.where(np.isfinite(p))[0]
    flagged = np.zeros_like(p, dtype=bool)
    if idx.size == 0:
        return flagged
    order = idx[np.argsort(p[idx])]
    n = order.size
    passed = p[order] <= q * np.arange(1, n + 1) / n
    if passed.any():
        kmax = int(np.max(np.where(passed)[0]))
        flagged[order[: kmax + 1]] = True
    return flagged
