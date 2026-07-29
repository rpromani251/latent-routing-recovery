"""
Is the honest model inside the smooth null class?
SUPERSEDED — see legacy/README.md. Kept for provenance; not used by run_all.sh.

routing_audit_v2 3.3 defines the null as a population of SMOOTH single-branch local
response surfaces (Matern GP, curvature lengthscale ell). A gradient-boosted tree
ensemble is piecewise constant: every anchor sits near some split boundary, so its
local response surface is a staircase, not a smooth surface.

If the honest model is outside the null class, Stage A flags it everywhere and the
audit has no usable null -- regardless of whether any gate is present.

This compares dispersion curves for the same data under:
  (a) XGBoost      -- the current pipeline's honest model
  (b) a smooth fit -- ridge on a spline basis

DATA DEPENDENCY: this diagnostic runs against the *original* Phase 1/2 housing dataset
from the sibling `geospatial-xai-attacks` repo
(`data/processed/seattle_housing_with_demographics.csv`), not this repo's Seattle EUI
data. It is not reproducible from this repo alone — pass --geo-xai-repo to point at a
checkout of that repo, or set ROUTING_AUDIT_GEOXAI_REPO.

Run:  python -m src.routing_audit.legacy.diag_model_class [--geo-xai-repo PATH]
"""
import argparse
import os
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from ..dispersion import dispersion_curve
from ..paths import FIGURES

RNG = np.random.default_rng(7)

FEATS = ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
         "waterfront", "view", "age", "UTM_X", "UTM_Y"]

SIGMAS = np.geomspace(0.005, 1.0, 14)
TAU_OBS = 0.01
M = 400
N_ANCHOR = 40


def peakedness(r, valid):
    """How far the max sits from the endpoints of the valid ladder, in log units.
    ~0 for monotone curves; large for interior peaks."""
    v = r[valid]
    if v.size < 4:
        return np.nan
    lr = np.log(v + 1e-12)
    return float(lr.max() - max(lr[0], lr[-1]))


def main(geo_xai_repo: Path):
    csv_path = geo_xai_repo / "data" / "processed" / "seattle_housing_with_demographics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. This diagnostic needs a checkout of the "
            "geospatial-xai-attacks repo with data/processed/ populated "
            "(see that repo's scripts/process_census.py).")

    df = pd.read_csv(csv_path)
    X = df[FEATS].to_numpy(float)
    y = df["log_price"].to_numpy(float)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # (a) boosted trees, structurally identical to that pipeline's XGBoost honest model
    gbm = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.1,
                                        random_state=0).fit(Xs, y)

    # (b) smooth: cubic spline basis + ridge
    spl = make_pipeline(SplineTransformer(n_knots=6, degree=3), Ridge(alpha=1.0)).fit(Xs, y)

    print(f"train R2   xgboost={gbm.score(Xs,y):.3f}   spline={spl.score(Xs,y):.3f}")

    idx = RNG.choice(len(Xs), N_ANCHOR, replace=False)

    rows = {}
    for name, model in (("xgboost", gbm), ("spline", spl)):
        f = lambda Z, m=model: m.predict(Z)
        pk, nv = [], []
        curves = []
        for i in idx:
            dc = dispersion_curve(f, Xs[i], SIGMAS, M, TAU_OBS, RNG)
            pk.append(peakedness(dc["r"], dc["valid"]))
            nv.append(int(dc["valid"].sum()))
            curves.append((dc["r"], dc["valid"]))
        pk = np.array(pk, float)
        rows[name] = (pk, curves)
        print(f"{name:>8}: peakedness  median={np.nanmedian(pk):.3f}  "
              f"p90={np.nanquantile(pk,0.9):.3f}  frac>0.2={np.nanmean(pk>0.2):.2f}   "
              f"mean valid scales={np.mean(nv):.1f}/{len(SIGMAS)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, name in zip(ax, ("xgboost", "spline")):
        _, curves = rows[name]
        for r, v in curves[:40]:
            if v.sum() >= 4:
                a.plot(SIGMAS[v], r[v], color="k", alpha=0.18, lw=1)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel(r"probe scale $\sigma$")
        a.set_title(f"{name}: honest model, NO gate", fontsize=11)
    ax[0].set_ylabel(r"$r_i(\sigma)$")
    fig.suptitle("Dispersion curves under the honest model alone "
                 "(smooth null must look flat/monotone)", fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "legacy_fig_model_class.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo-xai-repo", type=Path,
                     default=Path(os.environ.get("ROUTING_AUDIT_GEOXAI_REPO",
                                                  "../geospatial-xai-attacks")))
    args = ap.parse_args()
    main(args.geo_xai_repo)
