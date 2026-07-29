"""
Probe families and the gated/honest response function.

PROBE FAMILIES (routing_audit_v2 2.2)
  ambient       delta ~ N(0, sigma^2 I) on (lat, lon). Lands anywhere, including water.
  on-manifold   probe points RESAMPLED from real building locations with weights
                proportional to exp(-d^2 / 2 sigma^2). Support realism exact by
                construction. Note the support is discrete: at small sigma only a few
                real buildings are nearby, which sets a minimum resolvable scale from
                data density. Scales with too few distinct neighbours are abstained from.

On-manifold is primary (RESULTS_2026-07-28.md S7): the spatial-randomization alignment
claim is only significant under on-manifold probing (p = 0.013-0.033 vs 0.083-0.208
ambient, across every statistic tried) — but see docs/results_2026-07-28.md S7b for the
precondition this relies on (the probe distribution itself must be locally smooth in the
probe metric, or the data's own clustering reads as model multimodality).
"""
import numpy as np

from .model import b_lookup, TAU_B, PENALTY

MIN_NEIGH = 12          # distinct real buildings needed for an on-manifold scale


class Probe:
    def __init__(self, tree, pts_ll, kind, m_lat, m_lon):
        self.tree, self.pts, self.kind = tree, pts_ll, kind
        self.m_lat, self.m_lon = m_lat, m_lon
        self.lat0_ref = None
        self.lon0_ref = None

    def draw(self, lat0, lon0, sig_m, n, rng):
        """Return (lat, lon) arrays, or (None, None) if support is insufficient."""
        if self.kind == "ambient":
            return (lat0 + rng.normal(0, sig_m / self.m_lat, n),
                    lon0 + rng.normal(0, sig_m / self.m_lon, n))
        # on-manifold: kernel-weighted resample of real building locations
        p = np.array([(lon0 - self.lon0_ref) * self.m_lon, (lat0 - self.lat0_ref) * self.m_lat])
        idx = self.tree.query_ball_point(p, r=4.0 * sig_m)
        if len(idx) < MIN_NEIGH:
            return None, None
        idx = np.asarray(idx)
        d2 = ((self.tree.data[idx] - p) ** 2).sum(1)
        w = np.exp(-0.5 * d2 / sig_m**2)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return None, None
        pick = rng.choice(idx, size=n, replace=True, p=w / s)
        return self.pts[pick, 0], self.pts[pick, 1]


def respond(lat, lon, hb, loc, ras, rng, gated=True, tau_obs=0.010, tau_b=TAU_B, penalty=PENALTY):
    y = hb + np.interp(lat, loc["lat_g"], loc["g_lat"]) \
           + np.interp(lon, loc["lon_g"], loc["g_lon"])
    if gated:
        y = y - penalty * (b_lookup(ras, lat, lon) >= tau_b)
    return y + rng.normal(0, tau_obs, len(lat))
