"""
Experiment S, measurements S5 / S1 / S3 / S4 -- characterising the real scaffold.

PRE-REGISTERED (docs/experiment_designs, sec. 13). Predictions stated before running:
  S-1  the OOD detector classifies >80% of our tangent-frame probes as REAL
       -> trichotomy regime (i), the scaffold defeated by evasion not detection.
  S-2  the scaffold's Delta/sigma_resid exceeds 2.5, so detection power is not
       the binding constraint.
  S-3  retention rho at boundary-crossing anchors falls below rho_min = 0.5, so
       the audit abstains where the signal is. EXPLICIT FALSIFICATION CONDITION.

WHAT IS MEASURED
  S5  fraction of the AUDIT's own probes the OOD detector calls "real",
      against the fraction for LIME-style probes as the reference contrast.
      The detector was trained to recognise LIME's perturbation (ambient,
      all features, N(0, 0.3)); ours is isotropic inside a d-hat dimensional
      tangent frame at the local geometric scale. Different distribution.
  S1  Delta / sigma_resid of the routed output: the jump between probes routed
      to f and probes routed to psi, over the LTS residual scale.
  S3  density-filter retention rho at anchors whose ball crosses the boundary.
  S4  fraction of REAL held-out points the detector misclassifies as perturbed.

TARGETS
  housing            our geospatial scaffold, continuous response (log price)
  compas/german/cc   Slack et al. (AIES 2020), via their own adversarial_models

Geometry: Stage 0 is deferred, so the frame and radius are SUPPLIED -- local PCA
on the k nearest real points in standardised space, radius = median distance to
the k/2-th neighbour. Probes are generated in standardised space and mapped back
to raw units, because that is the space the detector sees.
"""
import json
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
SEED = 20260813
K_NEIGH = 60
VAR_KEEP = 0.90
D_MAX = 30
M_PROBE = 400
N_ANCHOR = 150
RHO_MIN = 0.5
C_FILTER = 1.5
K_DENS = 5


# --------------------------------------------------------------- probe geometry
def local_frame(Zs, nbr_idx, x_std):
    """Local PCA frame + radius from the k nearest real points, standardised space."""
    N = Zs[nbr_idx] - x_std
    C = np.cov(N.T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    frac = np.cumsum(w) / max(w.sum(), 1e-12)
    d_hat = int(np.clip(np.searchsorted(frac, VAR_KEEP) + 1, 1, D_MAX))
    U = V[:, :d_hat]
    r = float(np.median(np.linalg.norm(N[: max(2, len(N) // 2)], axis=1)))
    return U, d_hat, r


def audit_probes(x_std, U, d_hat, r, m, rng):
    sigma = r / np.sqrt(d_hat)
    z = rng.normal(0.0, sigma, size=(m, d_hat))
    return x_std[None, :] + z @ U.T


def retention(P_std, Zs, nn_dens, x_std):
    """Density filter: d_k(p) <= c * s(x), all in standardised space."""
    dk, _ = nn_dens.kneighbors(P_std, n_neighbors=K_DENS)
    dk = dk[:, -1]
    d_anchor, _ = nn_dens.kneighbors(x_std[None, :], n_neighbors=K_NEIGH)
    s_x = float(np.median(d_anchor[0]))
    return dk <= C_FILTER * s_x


def lts_resid_sd(P_std, y, x_std, h=0.75):
    """Residual scale after a trimmed local-linear fit in probe coordinates."""
    A = np.column_stack([np.ones(len(P_std)), P_std - x_std])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    keep = np.argsort(np.abs(r))[: int(h * len(y))]
    c2, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    r2 = y - A @ c2
    return float(np.std(r2[keep], ddof=1)), r2


# --------------------------------------------------------------- targets
def build_housing():
    df = pd.read_csv("/mnt/user-data/uploads/geospatial-xai-attacks/data/processed/"
                     "seattle_housing_with_demographics.csv")
    feats = ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
             "waterfront", "view", "age", "pct_black_latino", "UTM_X", "UTM_Y"]
    safe = ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
            "waterfront", "view", "age"]
    df = df.dropna(subset=feats + ["log_price"])
    X = df[feats].values.astype(float)
    y = df["log_price"].values.astype(float)
    pct = feats.index("pct_black_latino")
    safe_idx = [feats.index(c) for c in safe]

    import xgboost as xgb
    honest = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1,
                              random_state=42).fit(X, y)
    psi = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1,
                           random_state=42).fit(X[:, safe_idx], y)

    def f_biased(Z):
        return honest.predict(Z) - 0.3 * (Z[:, pct] > 0.25)

    def psi_innocuous(Z):
        return psi.predict(Z[:, safe_idx])

    # LIME-style OOD detector, exactly the published recipe
    rng = np.random.default_rng(SEED)
    ax, ay = [], []
    for _ in range(30):
        ax.append(np.vstack([X, X + rng.normal(0, 0.3, size=X.shape)]))
        ay.append(np.concatenate([np.ones(len(X)), np.zeros(len(X))]))
    ax, ay = np.vstack(ax), np.concatenate(ay)
    xtr, xte, ytr, yte = train_test_split(ax, ay, test_size=0.2, random_state=42)
    det = RandomForestClassifier(n_estimators=100, random_state=42).fit(xtr, ytr)
    return dict(name="housing", X=X, feats=feats, detector=det,
                f=f_biased, psi=psi_innocuous, continuous=True,
                det_acc=float(det.score(xte, yte)))


def build_slack(which):
    sys.path.insert(0, "/tmp/s_exp/flime")
    import os
    os.chdir("/tmp/s_exp/flime")
    from adversarial_models import Adversarial_Lime_Model
    from utils import Params, one_hot_encode
    import get_data as gd

    params = Params("model_configurations/experiment_params.json")
    np.random.seed(params.seed)
    getter = {"compas": gd.get_and_preprocess_compas_data,
              "german": gd.get_and_preprocess_german,
              "cc": gd.get_and_preprocess_cc}[which]
    X, y, cols = getter(params)
    X["unrelated_column_one"] = np.random.choice([0, 1], size=X.shape[0])
    feats = [c for c in X]
    sens = {"compas": "race", "german": "Gender", "cc": "racePctWhite numeric"}[which]
    sens = sens if sens in feats else [c for c in feats if "race" in c.lower()
                                       or "gender" in c.lower()][0]
    si, ui = feats.index(sens), feats.index("unrelated_column_one")
    Xv = X.values.astype(float)

    class F:
        def predict(self, Z):
            return np.array([params.negative_outcome if z[si] > 0
                             else params.positive_outcome for z in Z])
        def predict_proba(self, Z):
            return one_hot_encode(self.predict(Z))

    class Psi:
        def predict_proba(self, Z):
            return one_hot_encode(np.array([params.negative_outcome if z[ui] > 0
                                            else params.positive_outcome for z in Z]))

    adv = Adversarial_Lime_Model(F(), Psi()).train(Xv, y, feature_names=feats)
    det = adv.perturbation_identifier
    ncols = adv.numerical_cols

    def f_biased(Z):
        return F().predict_proba(Z)[:, 1].astype(float)

    def psi_innocuous(Z):
        return Psi().predict_proba(Z)[:, 1].astype(float)

    return dict(name=which, X=Xv, feats=feats, detector=det, ncols=ncols,
                f=f_biased, psi=psi_innocuous, continuous=False, det_acc=np.nan)


# --------------------------------------------------------------- the experiment
def run(target):
    rng = np.random.default_rng(SEED)
    X, det = target["X"], target["detector"]
    ncols = target.get("ncols")
    sc = StandardScaler().fit(X)
    Zs = sc.transform(X)
    nn = NearestNeighbors().fit(Zs)
    nn_dens = NearestNeighbors().fit(Zs)

    def det_real(P_raw):
        Q = P_raw[:, ncols] if ncols is not None else P_raw
        return det.predict_proba(Q)[:, 1] >= 0.5

    # S4: detector leakage on real held-out points
    s4 = float(1.0 - det_real(X).mean())

    idx = rng.choice(len(X), size=min(N_ANCHOR, len(X)), replace=False)
    rows = []
    for i in idx:
        x_std = Zs[i]
        _, nbr = nn.kneighbors(x_std[None, :], n_neighbors=K_NEIGH)
        U, d_hat, r = local_frame(Zs, nbr[0], x_std)
        if not np.isfinite(r) or r <= 0:
            continue

        P_std = audit_probes(x_std, U, d_hat, r, M_PROBE, rng)
        P_raw = sc.inverse_transform(P_std)
        real_audit = det_real(P_raw)

        # reference: LIME's own perturbation, the distribution the detector saw
        L_raw = X[i][None, :] + rng.normal(0, 0.3, size=(M_PROBE, X.shape[1]))
        real_lime = det_real(L_raw)

        keep = retention(P_std, Zs, nn_dens, x_std)
        frac_real = float(real_audit.mean())
        straddles = 0.02 <= frac_real <= 0.98

        row = dict(anchor=int(i), d_hat=d_hat, r=r,
                   s5_audit_frac_real=frac_real,
                   s5_lime_frac_real=float(real_lime.mean()),
                   s3_rho=float(keep.mean()),
                   straddles=bool(straddles))

        # S1: the routed jump, in LTS residual units
        yb, yp = target["f"](P_raw), target["psi"](P_raw)
        y_routed = np.where(real_audit, yb, yp)
        if straddles and real_audit.sum() > 20 and (~real_audit).sum() > 20:
            sd, _ = lts_resid_sd(P_std, y_routed, x_std)
            gap = abs(y_routed[real_audit].mean() - y_routed[~real_audit].mean())
            row["s1_delta"] = float(gap)
            row["s1_sd"] = sd
            row["s1_ratio"] = float(gap / sd) if sd > 1e-12 else np.nan
        # forced pairs: real point vs its LIME-perturbed twin (always defined)
        row["s1b_forced_gap"] = float(abs(
            np.where(det_real(X[i][None, :]), target["f"](X[i][None, :]),
                     target["psi"](X[i][None, :]))[0]
            - np.where(real_lime, yb[:1].repeat(M_PROBE), yp[:1].repeat(M_PROBE)).mean()))
        row["n_response_values"] = int(len(np.unique(np.round(y_routed, 9))))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["target"] = target["name"]
    df["s4_detector_leakage"] = s4
    df["detector_acc"] = target["det_acc"]
    df["continuous_response"] = target["continuous"]
    return df


if __name__ == "__main__" and False:
    out = []
    for builder, tag in [(build_housing, "housing"),
                         (lambda: build_slack("compas"), "compas"),
                         (lambda: build_slack("german"), "german"),
                         (lambda: build_slack("cc"), "cc")]:
        try:
            t = builder()
            d = run(t)
            out.append(d)
            print(f"{tag:9s} n={len(d):4d}  "
                  f"S5 audit-real={d.s5_audit_frac_real.mean():.3f}  "
                  f"LIME-real={d.s5_lime_frac_real.mean():.3f}  "
                  f"rho={d.s3_rho.mean():.3f}  straddle={d.straddles.mean():.3f}")
        except Exception as e:
            print(f"{tag:9s} FAILED: {type(e).__name__}: {e}")
    if out:
        pd.concat(out).to_csv("/tmp/s_exp/s_experiment_rows_dmax30.csv", index=False)
        print("\nwrote s_experiment_rows.csv")
