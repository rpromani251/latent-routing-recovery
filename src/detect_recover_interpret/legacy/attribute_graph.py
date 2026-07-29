"""
The anchor graph need not be geographic — NEGATIVE RESULT, with a diagnosis worth keeping.
Kept for provenance; not used by run_all.sh. See legacy/README.md and
docs/results_2026-07-28.md S7b for the full write-up.

routing_audit_v2 SYNC1 already allows "geographic, manifold, or task-metric" neighborhoods,
but only as a Stage B synchronization device. If the probe/pooling graph is arbitrary, then
geography is one instantiation and the method is black-box auditing with an anchor graph --
spatial auditing being a special case. This tests that directly.

SETUP -- nothing spatial anywhere in it
  graph      kNN over standardized BUILDING ATTRIBUTES
             (log GFA, YearBuilt, floors, log #buildings)
  probe      kernel-weighted resample of real buildings in ATTRIBUTE space
  gate       f(x) = h(x) - 0.30 * 1[ YearBuilt < TAU_YEAR ]
             a vintage gate: pre-TAU_YEAR stock is systematically underpredicted, so it
             looks more efficient than it is and is screened out of retrofit programs.
             Energy-relevant, and invisible to any geographic probe.

TESTS
  dip    Hartigan, null = unimodal (nonparametric, assumption-light)
  GLRT   K=1 vs K=2 Gaussian mixture, null = a single Gaussian (narrower, more powerful,
         breaks under non-Gaussian single-branch responses).
         The LR statistic is location/scale invariant, so its null law depends only on the
         sample size -- simulate once, reuse for every anchor, no per-anchor bootstrap.

CONCLUSION (RESULTS S7b): does not work on this dataset. The data itself is multimodal
in the probe metric (construction booms, standard building heights), so a neighbourhood
contains buildings from distinct clusters and their responses inherit that clustering
under a perfectly smooth model with NO gate at all. On-manifold probing requires the
probe distribution itself to be locally smooth in the probe metric -- geography happened
to satisfy this by luck, not by principle.

Run:  python -m src.detect_recover_interpret.legacy.attribute_graph <i0> <i1> <tag>
"""
import sys

import numpy as np, pandas as pd, pickle
import diptest
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture

from .. import paths

TAU_YEAR = 1960
PENALTY = 0.30
TAU_OBS = 0.010
SCALES = np.geomspace(0.10, 3.0, 3)      # standardized attribute-space units
M_DIP = M_REC = 1000
ALPHA = 0.05
MIN_NEIGH = 12
ATTRS = ["logGFA", "YearBuilt", "NumberofFloors", "logNB"]


# ------------------------------------------------------------------ GLRT null
def glrt_stat(y):
    """2 * (loglik of 2-component GMM - loglik of a single Gaussian)."""
    y = np.asarray(y, float).reshape(-1, 1)
    n = len(y)
    s = y.std(ddof=1)
    if not np.isfinite(s) or s <= 0:
        return np.nan
    l1 = -0.5 * n * (np.log(2 * np.pi * s**2) + 1.0)
    try:
        gm = GaussianMixture(2, n_init=2, random_state=0).fit(y)
        l2 = gm.score(y) * n
    except Exception:
        return np.nan
    return float(2.0 * (l2 - l1))


def simulate_glrt_null(m, B, rng):
    """LR is location/scale invariant, so one N(0,1) simulation calibrates every anchor."""
    return np.sort([glrt_stat(rng.normal(size=m)) for _ in range(B)])


def glrt_p(stat, null):
    if not np.isfinite(stat):
        return np.nan
    return float((1.0 + np.sum(null >= stat)) / (len(null) + 1.0))


# ------------------------------------------------------------------ probe
class AttrProbe:
    """kNN in standardized attribute space, RESTRICTED to matching `group`.

    The group restriction is not cosmetic. A first pass without it flagged the
    honest model everywhere: resampling across use types crosses a genuine
    discontinuity (an office and a hospital have very different EUI), so the
    honest response is multimodal over the neighbourhood with no gate present.
    That is the ell > sigma_T condition generalised -- the anchor graph must be
    fine enough that the honest model is smooth over it IN THE PROBE'S OWN METRIC.
    Any strong predictor omitted from the graph reappears as a false positive.
    """

    def __init__(self, Z, tree, group):
        self.Z, self.tree, self.group = Z, tree, group

    def draw(self, k, sig, n, rng):
        idx = self.tree.query_ball_point(self.Z[k], r=4.0 * sig)
        if len(idx) < MIN_NEIGH:
            return None
        idx = np.asarray(idx)
        idx = idx[self.group[idx] == self.group[k]]
        if idx.size < MIN_NEIGH:
            return None
        d2 = ((self.Z[idx] - self.Z[k]) ** 2).sum(1)
        w = np.exp(-0.5 * d2 / sig**2)
        t = w.sum()
        if not np.isfinite(t) or t <= 0:
            return None
        return rng.choice(idx, size=n, replace=True, p=w / t)


def main(i0, i1, tag):
    df = pd.read_csv(paths.SEATTLE_BUILDINGS_CSV)
    model, num = pickle.load(open(paths.SEATTLE_MODEL_PKL, "rb"))

    X = df[num + ["LargestPropertyUseType"]].copy()
    X["PropertyGFATotal"] = np.log(X["PropertyGFATotal"].clip(lower=1))
    h_all = np.asarray(model.predict(X), float)          # honest prediction per building
    old = (df["YearBuilt"].to_numpy() < TAU_YEAR)        # the gate indicator

    # The graph must SPAN THE MODEL'S INPUT SPACE. A first pass over building
    # attributes alone flagged the honest model on 55% of anchors: buildings matching
    # on those attributes sit anywhere in the city, so the omitted lat/lon variation
    # appeared as unexplained multimodality. Location enters here as one coordinate
    # among six -- the gate is still on a NON-SPATIAL feature (vintage), which is the
    # point of the experiment.
    A = pd.DataFrame({
        "Latitude": df["Latitude"],
        "Longitude": df["Longitude"],
        "logGFA": np.log(df["PropertyGFATotal"].clip(lower=1)),
        "YearBuilt": df["YearBuilt"],
        "NumberofFloors": df["NumberofFloors"],
        "logNB": np.log(df["NumberofBuildings"].clip(lower=1) + 1),
    })[["Latitude", "Longitude"] + ATTRS].to_numpy(float)
    Z = (A - A.mean(0)) / (A.std(0) + 1e-9)
    group = pd.factorize(df["LargestPropertyUseType"])[0]
    probe = AttrProbe(Z, cKDTree(Z), group)

    rng = np.random.default_rng(31337 + i0)
    null = simulate_glrt_null(M_DIP, 400, np.random.default_rng(5))

    rows = []
    for k in range(i0, min(i1, len(df))):
        rec = {"idx": k, "old": bool(old[k]), "year": float(df["YearBuilt"].iloc[k])}
        for gated, pre in ((True, "g"), (False, "h")):
            dp = np.full(len(SCALES), np.nan)
            gp = np.full(len(SCALES), np.nan)
            dl = np.full(len(SCALES), np.nan)
            pt = np.full(len(SCALES), np.nan)
            for t, sig in enumerate(SCALES):
                sel = probe.draw(k, sig, M_DIP, rng)
                if sel is None:
                    continue
                y = h_all[sel] - (PENALTY * old[sel] if gated else 0.0)
                y = y + rng.normal(0, TAU_OBS, len(y))
                dp[t] = diptest.diptest(np.ascontiguousarray(y))[1]
                gp[t] = glrt_p(glrt_stat(y), null)
                f = float(np.mean(old[sel])); pt[t] = min(f, 1 - f)

                sel2 = probe.draw(k, sig, M_REC, rng)
                if sel2 is None:
                    continue
                y2 = h_all[sel2] - (PENALTY * old[sel2] if gated else 0.0)
                y2 = y2 + rng.normal(0, TAU_OBS, len(y2))
                gm = GaussianMixture(2, n_init=2, random_state=0).fit(y2.reshape(-1, 1))
                mu = np.sort(gm.means_.ravel())
                w = gm.weights_[np.argsort(gm.means_.ravel())]
                if min(w) * M_REC >= 20:
                    dl[t] = float(mu[1] - mu[0])
            nd = int(np.isfinite(dp).sum())
            rec[f"{pre}_n"] = nd
            rec[f"{pre}_dip_flag"] = bool(nd and np.nanmin(dp) < ALPHA / nd)
            rec[f"{pre}_glrt_flag"] = bool(np.isfinite(gp).any()
                                           and np.nanmin(gp) < ALPHA / max(nd, 1))
            rec[f"{pre}_delta"] = float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan
            rec[f"{pre}_pi_true"] = float(np.nanmax(pt)) if np.isfinite(pt).any() else np.nan
        rows.append(rec)
    out = paths.RESULTS / f"legacy_attr_{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
