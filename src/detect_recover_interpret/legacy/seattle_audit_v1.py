"""
Stage A audit of the gated Seattle EUI model — the FIRST Seattle run.
SUPERSEDED — see legacy/README.md. Kept for provenance; not used by run_all.sh.
Superseded by audit.py's naive multi-scale scan (RESULTS S6d), which also corrects the
ground truth from "distance to boundary" to "does the probe neighbourhood actually
contain a crossing."

Probes perturb LOCATION in metres. b(x) is recomputed at each perturbed point from the
tract raster, so probes near a demographic boundary cross the gate.

Reported per building:
  R_log       uncalibrated log-range of the dispersion curve  (v2 3.2 S1)
  sigma*      scale at which dispersion peaks
  dip, dip_p  Hartigan dip on a fresh confirmation sample at sigma*  (v2 3.5)
  d_bound     true metres to the nearest gate boundary, for scoring only --
              never an estimator input

Run:  python -m src.detect_recover_interpret.legacy.seattle_audit_v1 <i0> <i1> <tag>
(expects seattle_buildings.csv / seattle_prep.npz / seattle_model.pkl in data/processed/,
 see scripts/build_model.py)
"""
import sys

import numpy as np, pandas as pd, pickle
import diptest
from scipy import stats

from ..model import b_lookup, TAU_B, PENALTY
from ..location_terms import build_location_terms, M_LAT, M_LON
from .. import paths

SIG_M = np.geomspace(25.0, 1200.0, 12)      # probe scales in metres
M_PROBE = 200
M_CONFIRM = 600
TAU_OBS = 0.010
MIN_VALID = 4


def audit_building(k, loc, ras, rng, gated=True):
    lat0 = loc["lat0"][k]; lon0 = loc["lon0"][k]; hb = loc["h_base"][k]
    r = np.zeros(len(SIG_M)); lam = np.zeros(len(SIG_M))
    for t, sm in enumerate(SIG_M):
        dlat = rng.normal(0, sm / M_LAT, M_PROBE)
        dlon = rng.normal(0, sm / M_LON, M_PROBE)
        la, lo = lat0 + dlat, lon0 + dlon
        y = hb + np.interp(la, loc["lat_g"], loc["g_lat"]) \
               + np.interp(lo, loc["lon_g"], loc["g_lon"])
        if gated:
            y = y - PENALTY * (b_lookup(ras, la, lo) >= TAU_B)
        y = y + rng.normal(0, TAU_OBS, M_PROBE)
        lam[t] = np.var(y, ddof=1) / TAU_OBS**2
        r[t] = np.sqrt(max(lam[t] - 1.0, 0.0)) / sm
    valid = stats.chi2.sf((M_PROBE - 1) * lam, df=M_PROBE - 1) < 0.05
    if valid.sum() < MIN_VALID:
        return dict(R_log=np.nan, sstar=np.nan, dip=np.nan, dip_p=np.nan, nvalid=int(valid.sum()))
    lr = np.log(r[valid] + 1e-12)
    t_star = int(np.argmax(np.where(valid, r, -np.inf)))
    sm = SIG_M[t_star]
    dlat = rng.normal(0, sm / M_LAT, M_CONFIRM); dlon = rng.normal(0, sm / M_LON, M_CONFIRM)
    la, lo = lat0 + dlat, lon0 + dlon
    y = hb + np.interp(la, loc["lat_g"], loc["g_lat"]) + np.interp(lo, loc["lon_g"], loc["g_lon"])
    if gated:
        y = y - PENALTY * (b_lookup(ras, la, lo) >= TAU_B)
    y = y + rng.normal(0, TAU_OBS, M_CONFIRM)
    d_, p_ = diptest.diptest(np.ascontiguousarray(y))
    return dict(R_log=float(lr.max() - lr.min()), sstar=float(sm),
                dip=float(d_), dip_p=float(p_), nvalid=int(valid.sum()))


def distance_to_boundary(ras, lat, lon, max_m=1500.0, n_ring=64):
    """Metres to the nearest point where the gate indicator flips. Scoring only."""
    gate0 = b_lookup(ras, np.array([lat]), np.array([lon]))[0] >= TAU_B
    lo_, hi_ = 0.0, max_m
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    def flips(rad):
        la = lat + rad * np.sin(th) / M_LAT
        lo2 = lon + rad * np.cos(th) / M_LON
        return np.any((b_lookup(ras, la, lo2) >= TAU_B) != gate0)
    if not flips(hi_):
        return np.nan
    for _ in range(14):
        mid = 0.5 * (lo_ + hi_)
        if flips(mid): hi_ = mid
        else: lo_ = mid
    return 0.5 * (lo_ + hi_)


def main(i0, i1, tag):
    df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
    prep = np.load(paths.SEATTLE_PREP_NPZ)
    ras = {"B": prep["B"], "lats": prep["lats"], "lons": prep["lons"]}
    model, num = pickle.load(open(paths.SEATTLE_MODEL_PKL, "rb"))
    loc = build_location_terms(model, num, df)
    loc["lat0"] = df["Latitude"].to_numpy(); loc["lon0"] = df["Longitude"].to_numpy()

    rng = np.random.default_rng(1234 + i0)
    rows = []
    for k in range(i0, min(i1, len(df))):
        g = audit_building(k, loc, ras, rng, gated=True)
        h = audit_building(k, loc, ras, rng, gated=False)   # honest-model control
        rows.append(dict(idx=k,
                         lat=loc["lat0"][k], lon=loc["lon0"][k],
                         b=prep["b_bldg"][k],
                         d_bound=distance_to_boundary(ras, loc["lat0"][k], loc["lon0"][k]),
                         **{f"g_{a}": b for a, b in g.items()},
                         **{f"h_{a}": b for a, b in h.items()}))
    out = paths.RESULTS / f"legacy_audit_{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}  rows={len(rows)}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
