"""
Step 2: does the separability classifier recover the boundary normal?

SETTING
  Intrinsic d = 2, ambient D = 20. Flat manifold, planted hyperplane boundary.
  The tangent frame U is SUPPLIED (Stage 0 deferred), so the lift
  n_a = U nu / ||U nu|| is exact and contributes no error -- ambient D is
  exercised as a code path but should not affect orientation error.

  Honest surface   h(z) = beta . z          (linear trend in tangent coords)
  Gate             y = h(z) - Delta * 1[ z.n > d ] + eps,   eps ~ N(0, tau^2)

  The true normal n is drawn per anchor (uniform on the circle) so that no
  coordinate is privileged; orientation error is angle(n_hat, n) in degrees,
  taken modulo sign since the normal is only defined up to orientation.

WHY THIS IS THE QUESTION
  Per-anchor orientation error largely pools away as 1/sqrt(N); systematic
  BIAS does not. So the headline is the bias/variance split, not mean error.
  The oracle arm (true branch labels) gives the variance floor; the gap
  between it and the estimated-responsibility arm is the misassignment cost.

PRE-STATED PREDICTIONS
  P1  In the idealised case the direction is unbiased by symmetry: for
      z ~ N(0, sigma^2 I), both class-conditional means of z lie exactly
      along n. Misassignment should therefore ATTENUATE separation without
      ROTATING the estimate -> bias ~ 0, variance up.
  P2  Orientation error is U-shaped in pi: pi -> 0.5 leaves LTS no majority
      branch; pi -> 0 leaves the discriminant almost no minority points.
      Optimum expected around pi ~ 0.15-0.30.
  P3  Error falls with Delta/tau throughout.

LOGGING
  Everything needed to reconstruct ANY downstream arm without a rerun:
  per-probe z, response, true label, fitted responsibility, trim mask,
  filter mask. Arms (oracle / soft / hard labels, trim on-off, filter
  on-off) are post-processing over this log.
"""
import numpy as np
import pandas as pd
from sim_core import gmm2_fit

D_AMBIENT = 20
D_INTRINSIC = 2
M = 1000
SEED = 20260813

DELTA_OVER_TAU = [1.0, 1.5, 2.5, 5.0]
PI_TARGET = [0.05, 0.10, 0.20, 0.35, 0.50]
N_ANCHOR = 200
TAU = 0.02
BETA_NORM = 0.15          # honest-trend gradient magnitude in tangent coords
SIGMA = 0.20              # per-coordinate probe scale
TRIM_FRAC = 0.75          # LTS h = 0.75 m


def lts_residuals(Z, y, h=TRIM_FRAC):
    """Trimmed local-linear residualisation. Returns residuals + inlier mask."""
    A = np.column_stack([np.ones(len(Z)), Z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    k = int(h * len(Z))
    keep = np.argsort(np.abs(r))[:k]
    coef2, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    mask = np.zeros(len(Z), bool)
    mask[keep] = True
    return y - A @ coef2, mask


def normal_from_labels(Z, w, use_lda=True):
    """Fisher/LDA direction from soft or hard weights w in [0,1]."""
    w = np.clip(np.asarray(w, float), 0.0, 1.0)
    n1, n2 = (1.0 - w).sum(), w.sum()
    if min(n1, n2) < 5:
        return None
    mu1 = ((1.0 - w)[:, None] * Z).sum(0) / n1
    mu2 = (w[:, None] * Z).sum(0) / n2
    dmu = mu2 - mu1
    if not use_lda:
        nrm = np.linalg.norm(dmu)
        return dmu / nrm if nrm > 1e-12 else None
    C1 = ((1.0 - w)[:, None] * (Z - mu1)).T @ (Z - mu1) / max(n1, 1e-9)
    C2 = (w[:, None] * (Z - mu2)).T @ (Z - mu2) / max(n2, 1e-9)
    S = (n1 * C1 + n2 * C2) / (n1 + n2) + 1e-9 * np.eye(Z.shape[1])
    v = np.linalg.solve(S, dmu)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-12 else None


def angle_deg(a, b):
    """Angle between directions, modulo sign (a normal has no preferred side)."""
    if a is None or b is None:
        return np.nan
    c = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def run_anchor(rng, dt_ratio, pi_target, U):
    """One anchor. Returns (summary row, per-probe log frame)."""
    tau = TAU
    delta = dt_ratio * tau
    n_true = rng.normal(size=D_INTRINSIC)
    n_true /= np.linalg.norm(n_true)
    # distance placing the crossing fraction at pi_target:  pi = Phi(-d/sigma)
    from scipy.stats import norm as _n
    d_true = -SIGMA * _n.ppf(pi_target)

    beta = rng.normal(size=D_INTRINSIC)
    beta = BETA_NORM * beta / np.linalg.norm(beta)

    Z = rng.normal(0.0, SIGMA, size=(M, D_INTRINSIC))
    proj = Z @ n_true
    crossed = proj > d_true                       # GROUND TRUTH ONLY
    y = Z @ beta - delta * crossed + rng.normal(0.0, tau, M)

    # density filter: flat uniform manifold here, so retention is isotropic and
    # the mask is all-true. Kept as a logged column so the arm exists later.
    keep_filter = np.ones(M, bool)

    resid, inlier = lts_residuals(Z[keep_filter], y[keep_filter])
    fit = gmm2_fit(resid, rng)
    gamma = fit["resp"][:, 1]                     # responsibility for component 1
    # orient responsibilities so that "1" is the lower-mean (penalised) branch
    if fit["mu"][1] > fit["mu"][0]:
        gamma = 1.0 - gamma

    Zf = Z[keep_filter]
    n_oracle = normal_from_labels(Zf, crossed[keep_filter].astype(float))
    n_soft = normal_from_labels(Zf, gamma)
    n_hard = normal_from_labels(Zf, (gamma > 0.5).astype(float))
    n_soft_in = normal_from_labels(Zf[inlier], gamma[inlier])

    row = dict(
        dt_ratio=dt_ratio, pi_target=pi_target, delta=delta, tau=tau,
        d_true=d_true, pi_emp=float(crossed.mean()),
        pi_hat=float(gamma.mean()),
        dhat_gap=float(abs(fit["mu"][1] - fit["mu"][0])),
        w_min=float(min(fit["w"])),
        err_oracle=angle_deg(n_oracle, n_true),
        err_soft=angle_deg(n_soft, n_true),
        err_hard=angle_deg(n_hard, n_true),
        err_soft_inlier=angle_deg(n_soft_in, n_true),
    )
    # signed rotation in the 2-D tangent plane, for the bias/variance split
    perp = np.array([-n_true[1], n_true[0]])
    for tag, nh in (("oracle", n_oracle), ("soft", n_soft)):
        if nh is None:
            row[f"signed_{tag}"] = np.nan
            continue
        s = np.sign(np.dot(nh, n_true)) or 1.0
        row[f"signed_{tag}"] = float(np.degrees(np.arctan2(np.dot(s * nh, perp),
                                                          np.dot(s * nh, n_true))))
    log = pd.DataFrame(dict(
        z1=Z[:, 0], z2=Z[:, 1], y=y, true_cross=crossed.astype(int),
        gamma=np.nan, inlier=0, keep_filter=keep_filter.astype(int)))
    log.loc[keep_filter, "gamma"] = gamma
    log.loc[np.flatnonzero(keep_filter)[inlier], "inlier"] = 1
    return row, log


def main():
    rng = np.random.default_rng(SEED)
    U, _ = np.linalg.qr(rng.normal(size=(D_AMBIENT, D_INTRINSIC)))   # supplied frame
    rows, logs = [], []
    for dt in DELTA_OVER_TAU:
        for pt in PI_TARGET:
            for a in range(N_ANCHOR):
                r, lg = run_anchor(rng, dt, pt, U)
                r["anchor"] = a
                rows.append(r)
                if a < 2:                       # keep a sample of full logs
                    lg = lg.assign(dt_ratio=dt, pi_target=pt, anchor=a)
                    logs.append(lg)
    df = pd.DataFrame(rows)
    df.to_csv("normal_estimator_rows.csv", index=False)
    pd.concat(logs).to_csv("normal_estimator_probelog.csv", index=False)
    print(f"anchors: {len(df)}   cells: {df.groupby(['dt_ratio','pi_target']).ngroups}")
    return df


if __name__ == "__main__":
    main()
