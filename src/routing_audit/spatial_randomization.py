"""
Spatial randomization test for the alignment claim.

Rather than controlling FDR over thousands of dependent per-anchor tests, test the claim
that actually matters: do flags concentrate near gate boundaries more than chance?

  statistic   fraction of dip-flagged buildings within R metres of a gate boundary
  null        permute the b values ACROSS TRACTS, keeping tract geometry and building
              locations fixed. This breaks the association between demographics and
              space while preserving both spatial structures exactly, so the resulting
              p-value is valid under arbitrary spatial dependence -- no independence
              assumption, no null generator.

RESULTS_2026-07-28.md S7c: the radius must match the method's detection radius, measured
independently from the power-vs-distance curve (800 m for the current naive-scan
protocol), not chosen for significance. Report the full radius sweep for auditability.

Implementation: rasterize TRACT INDEX once; under a permutation the gate mask is an
array lookup, and distance to the nearest boundary comes from a Euclidean distance
transform. So each replicate costs a distance transform, not a re-rasterization.
"""
import json
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt

from .location_terms import M_LAT, M_LON


def tract_index_raster(tracts_path, lat_range, lon_range, n=900):
    gj = json.load(open(tracts_path))
    lons = np.linspace(lon_range[0], lon_range[1], n)
    lats = np.linspace(lat_range[0], lat_range[1], n)
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    tid = np.full(pts.shape[0], -1, int)
    bvals = []
    for j, ft in enumerate(gj["features"]):
        v = ft["properties"].get("pct_black_latino")
        if v is None:
            continue
        k = len(bvals); bvals.append(float(v))
        geom = ft["geometry"]
        rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else \
                [p[0] for p in geom["coordinates"]]
        for ring in rings:
            a = np.asarray(ring, float)[:, :2]
            if a[:, 0].max() < lon_range[0] or a[:, 0].min() > lon_range[1]: continue
            if a[:, 1].max() < lat_range[0] or a[:, 1].min() > lat_range[1]: continue
            tid[MplPath(a).contains_points(pts)] = k
    return tid.reshape(n, n), np.array(bvals), lats, lons


def dist_to_boundary(gate, lats, lons):
    """Metres from every cell to the nearest gate-indicator flip."""
    edge = np.zeros_like(gate, bool)
    edge[:, :-1] |= gate[:, :-1] != gate[:, 1:]
    edge[:-1, :] |= gate[:-1, :] != gate[1:, :]
    if not edge.any():
        return np.full(gate.shape, np.inf)
    dy = (lats[1] - lats[0]) * M_LAT
    dx = (lons[1] - lons[0]) * M_LON
    return distance_transform_edt(~edge, sampling=[dy, dx])


def sample(arr, lats, lons, lat, lon):
    i = np.clip(((lat - lats[0]) / (lats[-1] - lats[0]) * (len(lats) - 1)).astype(int),
                0, len(lats) - 1)
    j = np.clip(((lon - lons[0]) / (lons[-1] - lons[0]) * (len(lons) - 1)).astype(int),
                0, len(lons) - 1)
    return arr[i, j]


def alignment_test(flag, lat, lon, tid, bvals, lats, lons, tau_b, radius_m, n_perm, rng):
    """Fraction of flagged buildings within `radius_m` of a gate boundary, against a
    permutation null over tract demographics. Returns (observed, null_draws, p)."""
    def stat(bv):
        gate = np.zeros_like(tid, bool)
        m = tid >= 0
        gate[m] = bv[tid[m]] >= tau_b
        dist = dist_to_boundary(gate, lats, lons)
        db = sample(dist, lats, lons, lat, lon)
        return float(np.mean(db[flag] <= radius_m))

    obs = stat(bvals)
    null = np.array([stat(rng.permutation(bvals)) for _ in range(n_perm)])
    p = (1.0 + np.sum(null >= obs)) / (n_perm + 1.0)
    return obs, null, p
