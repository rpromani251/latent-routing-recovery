"""
On-manifold (kNN) probes, ambient/on-manifold comparison, and K=2 regime recovery
— under the σ*-based single-scale v1 protocol.
SUPERSEDED — see legacy/README.md. Kept for provenance; not used by run_all.sh.

The FINDING here (on-manifold should be the primary probe family) still holds and is
cited in docs/results_2026-07-28.md S6b/S7. What's superseded is the protocol it was
measured under: sigma*-based single-scale dip+recovery, same as seattle_audit_v1.py.
The current pipeline (audit.py) is on-manifold only and hasn't been re-run with an
ambient variant under the naive multi-scale scan.

BUDGET SPLIT (v2 2.1) -- three independent draws, never reused:
  S1  dispersion curve and sigma* selection
  S2  dip confirmation at sigma*
  S3  K=2 mixture recovery at sigma*

Run:  python -m src.routing_audit.legacy.probe_family_comparison <i0> <i1> <tag>
"""
import sys

import numpy as np, pandas as pd, pickle
import diptest
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture

from ..model import b_lookup, TAU_B, PENALTY
from ..location_terms import build_location_terms, M_LAT, M_LON
from ..probes import Probe, respond
from .. import paths

SIG_M = np.geomspace(25.0, 1200.0, 12)
M_PROBE, M_CONFIRM, M_RECOVER = 200, 600, 600
TAU_OBS = 0.010
MIN_VALID = 4


def audit(k, loc, ras, probe, rng, gated=True):
    lat0, lon0, hb = loc["lat0"][k], loc["lon0"][k], loc["h_base"][k]
    r = np.zeros(len(SIG_M)); lam = np.zeros(len(SIG_M)); avail = np.ones(len(SIG_M), bool)
    for t, sm in enumerate(SIG_M):
        la, lo = probe.draw(lat0, lon0, sm, M_PROBE, rng)          # S1
        if la is None:
            avail[t] = False; continue
        y = respond(la, lo, hb, loc, ras, rng, gated, tau_obs=TAU_OBS)
        lam[t] = np.var(y, ddof=1) / TAU_OBS**2
        r[t] = np.sqrt(max(lam[t] - 1.0, 0.0)) / sm
    valid = avail & (stats.chi2.sf((M_PROBE - 1) * lam, df=M_PROBE - 1) < 0.05)
    out = dict(R_log=np.nan, sstar=np.nan, dip=np.nan, dip_p=np.nan,
               pi=np.nan, delta=np.nan, zhat=np.nan, nvalid=int(valid.sum()))
    if valid.sum() < MIN_VALID:
        return out
    lr = np.log(r[valid] + 1e-12)
    out["R_log"] = float(lr.max() - lr.min())
    t_star = int(np.argmax(np.where(valid, r, -np.inf)))
    sm = SIG_M[t_star]; out["sstar"] = float(sm)

    la, lo = probe.draw(lat0, lon0, sm, M_CONFIRM, rng)            # S2
    if la is None:
        return out
    y2 = respond(la, lo, hb, loc, ras, rng, gated, tau_obs=TAU_OBS)
    out["dip"], out["dip_p"] = [float(v) for v in
                                diptest.diptest(np.ascontiguousarray(y2))]

    la, lo = probe.draw(lat0, lon0, sm, M_RECOVER, rng)            # S3
    if la is None:
        return out
    y3 = respond(la, lo, hb, loc, ras, rng, gated, tau_obs=TAU_OBS)
    gm = GaussianMixture(2, covariance_type="full", n_init=3,
                         random_state=0).fit(y3.reshape(-1, 1))
    mu = gm.means_.ravel()
    order = np.argsort(mu)                       # consistent global ordering by mean
    resp = gm.predict_proba(y3.reshape(-1, 1))[:, order]
    out["pi"] = float(resp[:, 0].mean())         # mass on the LOWER component
    out["delta"] = float(abs(mu[order[1]] - mu[order[0]]))
    y0 = respond(np.array([lat0]), np.array([lon0]), hb, loc, ras, rng, gated, tau_obs=TAU_OBS)
    out["zhat"] = int(np.argmax(gm.predict_proba(y0.reshape(-1, 1))[:, order][0]))
    return out


def main(i0, i1, tag):
    df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
    prep = np.load(paths.SEATTLE_PREP_NPZ)
    ras = {"B": prep["B"], "lats": prep["lats"], "lons": prep["lons"]}
    model, num = pickle.load(open(paths.SEATTLE_MODEL_PKL, "rb"))
    loc = build_location_terms(model, num, df)
    loc["lat0"] = df["Latitude"].to_numpy(); loc["lon0"] = df["Longitude"].to_numpy()

    lat_ref, lon_ref = loc["lat0"].min(), loc["lon0"].min()
    XY = np.column_stack([(loc["lon0"] - lon_ref) * M_LON, (loc["lat0"] - lat_ref) * M_LAT])
    tree = cKDTree(XY)
    pts_ll = np.column_stack([loc["lat0"], loc["lon0"]])

    probes = {}
    for kind in ("ambient", "onmanifold"):
        p = Probe(tree, pts_ll, kind, M_LAT, M_LON)
        p.lat0_ref, p.lon0_ref = lat_ref, lon_ref
        probes[kind] = p

    rng = np.random.default_rng(555 + i0)
    rows = []
    for k in range(i0, min(i1, len(df))):
        rec = dict(idx=k, lat=loc["lat0"][k], lon=loc["lon0"][k], b=prep["b_bldg"][k])
        for kind, p in probes.items():
            g = audit(k, loc, ras, p, rng, gated=True)
            h = audit(k, loc, ras, p, rng, gated=False)
            rec.update({f"{kind}_g_{a}": v for a, v in g.items()})
            rec.update({f"{kind}_h_{a}": v for a, v in h.items()})
        rows.append(rec)
    out = paths.RESULTS / f"legacy_onman_{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}  rows={len(rows)}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
