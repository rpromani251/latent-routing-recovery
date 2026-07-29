"""
The current production audit protocol (RESULTS_2026-07-28.md S6d, "simplified protocol").

CHANGES vs the original per-anchor Stage A pipeline (legacy/stage_a_full.py)
  probe        on-manifold (kNN resample of real building locations) as primary
  scales       naive scan over 3-6 log-spaced scales; NO dispersion curve, NO sigma*
               selection (RESULTS S6c: sigma* is a WORSE scale-selection rule than a
               random scale, 0.715 vs 0.798 power -- dispersion is dropped as a gate)
  test         dip at each scale, Bonferroni over the scales actually dipped
  corroborate  recovered effect size Delta_hat from a K=2 mixture, and its
               CONSISTENCY across scales -- both free under a multi-scale protocol,
               and far sharper than dispersion (honest model Delta ~ 0.013 vs
               gated ~ 0.30, no overlap)
  truth        the honest ground truth is not distance to a boundary but whether the
               probe neighbourhood ACTUALLY CONTAINS A CROSSING. pi_true = fraction of
               probe points on the far side. A building 600 m out probed at 800 m does
               cross, so flagging it is correct.

Budget is a first-class design parameter (RESULTS S6e): fewer scales with more probes
each beats more scales with fewer probes, once you're in a low-mixing-fraction band.
`AuditConfig` exposes scales/m_dip/m_rec so configs/*.yaml can select either regime
without touching this module -- see configs/main_audit.yaml (6 x 500, the headline run)
vs configs/query_reallocation.yaml (3 x 1000, RESULTS S6e's reallocation experiment).
"""
from dataclasses import dataclass, field

import numpy as np
import diptest
from sklearn.mixture import GaussianMixture

from .model import b_lookup, TAU_B, PENALTY
from .probes import respond


@dataclass
class AuditConfig:
    scales_m: np.ndarray = field(default_factory=lambda: np.geomspace(25.0, 1200.0, 6))
    m_dip: int = 500
    m_rec: int = 500
    tau_obs: float = 0.010
    tau_b: float = TAU_B
    penalty: float = PENALTY
    alpha: float = 0.05
    min_component_mass: int = 20      # v2 minimum-mass rule for the K=2 recovery


def audit_building(k, loc, ras, probe, rng, cfg: AuditConfig, gated=True):
    """Dip at every scale in cfg.scales_m, Bonferroni over the scales actually dipped,
    plus K=2 recovery at every scale. Returns a dict of per-building summary stats."""
    lat0, lon0, hb = loc["lat0"][k], loc["lon0"][k], loc["h_base"][k]
    scales = cfg.scales_m
    dp = np.full(len(scales), np.nan)
    dl = np.full(len(scales), np.nan)
    pt = np.full(len(scales), np.nan)          # TRUE mixing fraction

    for t, sm in enumerate(scales):
        la, lo = probe.draw(lat0, lon0, sm, cfg.m_dip, rng)
        if la is None:
            continue
        y = respond(la, lo, hb, loc, ras, rng, gated,
                    tau_obs=cfg.tau_obs, tau_b=cfg.tau_b, penalty=cfg.penalty)
        dp[t] = diptest.diptest(np.ascontiguousarray(y))[1]
        f = float(np.mean(b_lookup(ras, la, lo) >= cfg.tau_b))
        pt[t] = min(f, 1 - f)                  # 0 = no crossing reachable

        la, lo = probe.draw(lat0, lon0, sm, cfg.m_rec, rng)   # fresh draw for recovery
        if la is None:
            continue
        y2 = respond(la, lo, hb, loc, ras, rng, gated,
                     tau_obs=cfg.tau_obs, tau_b=cfg.tau_b, penalty=cfg.penalty)
        gm = GaussianMixture(2, n_init=2, random_state=0).fit(y2.reshape(-1, 1))
        mu = np.sort(gm.means_.ravel())
        w = gm.weights_[np.argsort(gm.means_.ravel())]
        if min(w) * cfg.m_rec >= cfg.min_component_mass:
            dl[t] = float(mu[1] - mu[0])

    n_dipped = int(np.isfinite(dp).sum())
    out = {"n_scales": n_dipped}
    if n_dipped:
        out["p_min"] = float(np.nanmin(dp))
        out["flag"] = bool(np.nanmin(dp) < cfg.alpha / n_dipped)   # Bonferroni over scales
        out["delta_med"] = float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan
        fin = dl[np.isfinite(dl)]
        out["delta_cv"] = float(np.std(fin) / (np.mean(fin) + 1e-12)) if fin.size >= 2 else np.nan
        out["pi_true_max"] = float(np.nanmax(pt)) if np.isfinite(pt).any() else np.nan
    else:
        out.update(p_min=np.nan, flag=False, delta_med=np.nan,
                   delta_cv=np.nan, pi_true_max=np.nan)
    return out


def run_audit_batch(indices, loc, ras, probe, rng, cfg: AuditConfig):
    """Audit both the gated (deployed) and honest (no-gate control) model at each
    building index. Returns a list of per-building dict rows."""
    rows = []
    for k in indices:
        rec = {"idx": int(k), "lat": float(loc["lat0"][k]), "lon": float(loc["lon0"][k])}
        for gated, pre in ((True, "g"), (False, "h")):
            rec.update({f"{pre}_{a}": v
                        for a, v in audit_building(k, loc, ras, probe, rng, cfg, gated).items()})
        rows.append(rec)
    return rows
