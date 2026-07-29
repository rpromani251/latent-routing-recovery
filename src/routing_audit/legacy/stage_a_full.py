"""
Stage A detection, scalar-output (V=1) specialization of routing_audit_v2.
SUPERSEDED — see legacy/README.md. Kept for provenance; not used by run_all.sh.

For a scalar target the machinery collapses:
  - Sigma_obs is a scalar variance from repeated unperturbed queries
  - whitening is division by tau_obs
  - lambda_max(S_tilde) is just the whitened response variance
  - the dip projection (v2 3.5 P1) is the identity: dip the whitened responses directly

IMPLEMENTATION NOTE (matters, and is not spelled out in v2):
the minimum-signal rule must be applied PER SCALE, not only per anchor. At scales
where sigma*|slope| < tau_obs the whitened variance sits below the noise floor, the
(lambda-1)_+ clamp sends r to exactly 0, and log(r + eps) hits the epsilon floor.
A range statistic then measures the floor rather than any structure -- R_log values
around 18 = log(1) - log(1e-6) + noise. Screening scales individually fixes it.
"""
import numpy as np
import diptest

from ..dispersion import dispersion_curve, r_log, t_iso, sigma_star, MIN_VALID_SCALES

__all__ = ["audit_anchor", "MIN_VALID_SCALES"]


def audit_anchor(f, x, sigmas, m, tau_obs, rng, m_confirm=None, do_dip=True):
    """Full Stage A pass at one anchor: dispersion curve -> sigma* -> single dip at sigma*."""
    dc = dispersion_curve(f, x, sigmas, m, tau_obs, rng)
    r, valid, sig = dc["r"], dc["valid"], dc["sigmas"]

    out = {
        "r": r, "lam": dc["lam"], "valid": valid,
        "n_valid": int(valid.sum()),
        "insufficient_signal": bool(valid.sum() < MIN_VALID_SCALES),
        "R_log": np.nan, "T_iso": np.nan, "sigma_star": np.nan,
        "dip": np.nan, "dip_p": np.nan,
    }
    if out["insufficient_signal"]:
        return out

    out["R_log"] = r_log(r, valid)
    out["T_iso"] = t_iso(r, sig, valid)
    ts = sigma_star(r, sig, valid)
    out["sigma_star"] = float(sig[ts])

    if do_dip:
        # v2 A6: fresh confirmation sample (S2) at sigma*, never the S1 draw
        mc = m_confirm or m
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        delta = rng.normal(0.0, sig[ts], size=(mc, xa.shape[0]))
        y = f(xa[None, :] + delta) + rng.normal(0.0, tau_obs, size=mc)
        d, p = diptest.diptest(np.ascontiguousarray(y / tau_obs))
        out["dip"], out["dip_p"] = float(d), float(p)
    return out
