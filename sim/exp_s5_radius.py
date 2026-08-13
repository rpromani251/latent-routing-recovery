"""
S5 as a curve: at what probe radius does the audit remain invisible to the
OOD detector, and does the routing boundary live inside that radius?

A single yes/no answer to "does the detector see our probes" is the wrong
object, because it depends entirely on the radius the audit chooses -- which
Stage 0 is supposed to select and which is currently supplied by a k-NN
heuristic. Sweeping the radius gives two curves per target:

  visibility(r)  fraction of audit probes the detector calls REAL
  crossing(r)    fraction of anchors whose ball straddles the OOD boundary

The audit is usable where visibility is high AND crossing is non-zero. If the
two curves do not overlap, the scaffold is not auditable at any radius by this
probe family -- a structural statement, not a tuning failure.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/s_exp")
from exp_s_scaffold import (local_frame, retention, build_housing, build_slack,
                       K_NEIGH, K_DENS, C_FILTER)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

SEED = 20260813
M_PROBE = 300
N_ANCHOR = 60
R_FRACS = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
VAR_KEEP = 0.90
D_MAX = 30


def sweep(target):
    rng = np.random.default_rng(SEED)
    X, det, ncols = target["X"], target["detector"], target.get("ncols")
    sc = StandardScaler().fit(X)
    Zs = sc.transform(X)
    nn = NearestNeighbors().fit(Zs)

    def det_real(P_raw):
        Q = P_raw[:, ncols] if ncols is not None else P_raw
        return det.predict_proba(Q)[:, 1] >= 0.5

    idx = rng.choice(len(X), size=min(N_ANCHOR, len(X)), replace=False)
    rows = []
    for i in idx:
        x_std = Zs[i]
        _, nb = nn.kneighbors(x_std[None, :], n_neighbors=K_NEIGH)
        N = Zs[nb[0]] - x_std
        w, V = np.linalg.eigh(np.cov(N.T))
        o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
        frac = np.cumsum(w) / max(w.sum(), 1e-12)
        d_hat = int(np.clip(np.searchsorted(frac, VAR_KEEP) + 1, 1, D_MAX))
        U = V[:, :d_hat]
        r_knn = float(np.median(np.linalg.norm(N[: K_NEIGH // 2], axis=1)))
        for fr in R_FRACS:
            r = fr * r_knn
            sigma = r / np.sqrt(d_hat)
            z = rng.normal(0.0, sigma, size=(M_PROBE, d_hat))
            P_std = x_std[None, :] + z @ U.T
            P_raw = sc.inverse_transform(P_std)
            real = det_real(P_raw)
            fr_real = float(real.mean())
            yb, yp = target["f"](P_raw), target["psi"](P_raw)
            y_routed = np.where(real, yb, yp)
            rows.append(dict(
                target=target["name"], anchor=int(i), r_frac=fr, d_hat=d_hat,
                sigma_per_coord=sigma, frac_real=fr_real,
                crosses=bool(0.02 <= fr_real <= 0.98),
                n_resp_vals=int(len(np.unique(np.round(y_routed, 9)))),
                gap=(float(abs(y_routed[real].mean() - y_routed[~real].mean()))
                     if real.sum() > 15 and (~real).sum() > 15 else np.nan)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for b, tag in [(build_housing, "housing"), (lambda: build_slack("compas"), "compas"),
                   (lambda: build_slack("german"), "german"), (lambda: build_slack("cc"), "cc")]:
        try:
            d = sweep(b()); out.append(d)
            os.chdir("/tmp/s_exp")
            g = d.groupby("r_frac").agg(vis=("frac_real", "mean"),
                                        cross=("crosses", "mean"))
            print(f"\n{tag}   (d_hat median {int(d.d_hat.median())})")
            print("  r_frac  visibility  crossing")
            for rf, row in g.iterrows():
                print(f"   {rf:<6}  {row.vis:9.3f}  {row.cross:8.3f}")
        except Exception as e:
            print(f"{tag} FAILED: {type(e).__name__}: {e}")
    if out:
        pd.concat(out).to_csv("/tmp/s_exp/s5_radius_sweep.csv", index=False)
        print("\nwrote s5_radius_sweep.csv")
