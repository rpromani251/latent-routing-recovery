"""
Experiment P -- pooled recovery of a hidden routing boundary.

PRE-REGISTERED (docs/experiment_designs). Predictions stated before running:
  P-1  estimator ranking C > B > A at every Delta, with the C-B gap largest at
       Delta/tau = 1.5, where half the true detections sit below any per-anchor
       threshold and B discards them.
  P-3  N_95 under RANDOM anchor placement exceeds N_95 under BY-DESIGN placement
       by roughly the inverse shell fraction (~20-30x).

SETTING
  Intrinsic d = 2, ambient D = 20, frame supplied (Stage 0 deferred). A single
  planted hyperplane {t : n_true . t = c_true} in manifold coordinates. Since the
  frame is exact the lift contributes no error, so the estimation is carried out
  in manifold coordinates and D is a code path only (established in step 2).

  honest surface  h(t) = beta . t          (global linear trend)
  gate            y = h(t) - Delta * 1[n_true . t > c_true] + eps

PER-ANCHOR PRIMITIVE (validated 13 August, step 2)
  1. probes z ~ N(0, sigma^2 I_2) about the anchor
  2. trimmed local-linear residualisation, h = 0.75 m
  3. equal-variance 2-GMM by EM   -> responsibilities gamma, and the LRT
     statistic 2*(ll_mixture - ll_single_gaussian)
  4. LDA discriminant on (z, gamma), fitted on ALL probes not the LTS inliers
     -> normal nu, intercept b
  5. emit (n_a, c_a, w_a) with c_a = n_a . t_a + b/||nu||, w_a = LRT, UNTHRESHOLDED

POOLING ESTIMATORS
  A  Fisher combination of per-anchor p-values. Ignores geometry. Baseline.
  B  threshold on w_a, then cluster (n, c). Partly geometric.
  C  w-weighted Hough voting over ALL anchors' (n, c). Fully geometric.

TWO NULLS -- and this is a correction to the 31 July design.  [NEW]
  PERMUTATION null: within each anchor, permute the residual-to-probe assignment
  before the discriminant and rerun. This destroys the link between WHERE a probe
  sat and WHAT response it got -- the link a boundary creates -- while preserving
  the marginal residual distribution, the probe geometry and the anchor count.

  Because it preserves the MULTISET of residuals, the mixture fit and hence the
  LRT statistic are INVARIANT to it by construction (measured: real median 7.75
  against null median 7.88). So the permutation null is a null for GEOMETRY, not
  for EXISTENCE, and the statistic-only estimator A has no power against it by
  construction rather than by weakness. Comparing A against it would be a rigged
  test.

  HONEST null: a matched no-gate model (Delta = 0) on identical geometry. This is
  the existence null, and all three estimators are comparable against it.
  Both are reported.

STATISTIC  [NEW]
  A grid/Hough peak is the wrong instrument at these anchor counts -- 50 anchors
  over a 72x61 grid put roughly one anchor in the modal cell. Agreement is
  measured instead by a smooth pairwise kernel statistic

      S = sum_ab w_a w_b K_ang(theta_a - theta_b) K_off(c_a - c_b) / (sum w)^2

  which is grid-free and is exactly "do the anchors agree on both direction and
  offset". Offsets are taken relative to the anchor centroid, because an
  arbitrary origin amplifies direction error by the anchor's distance along the
  boundary.

Direction is defined only up to sign, so pooling is done in doubled-angle space
(continuous on the projective line); the sign is then resolved by majority vote
against the pooled direction before the offsets are combined.
"""
import numpy as np
import pandas as pd

D_AMBIENT, D_INTR = 20, 2
SIGMA = 0.20                 # per-coordinate probe scale
M_PROBE = 600
TAU = 0.02
BETA_NORM = 0.15
TRIM = 0.75
POOL = 250                   # anchors generated per cell
N_PERM_REP = 2               # permuted replicas per anchor -> null pool
SEED = 20260813

DELTA_OVER_TAU = [1.0, 1.5, 2.5, 5.0]
PI_TARGET = [0.05, 0.10, 0.20]
N_LIST = [5, 10, 25, 50, 100]
N_REPEAT = 300               # subsamples per (cell, N)
ALPHA = 0.05

# shell that step 2 identified as orientation-useful: pi in [0.05, 0.10]
SHELL = (1.28, 1.64)         # |signed distance| / sigma
RANDOM_SPAN = 6.0            # random placement: |d|/sigma ~ U(0, RANDOM_SPAN)


# ----------------------------------------------------------------- primitives
def _norm_logpdf(y, mu, sd):
    return -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((y - mu) / sd) ** 2


def gmm2_equalvar(y, rng, n_init=3, max_iter=150, tol=1e-7):
    """Equal-variance two-component 1-D mixture. The shared sigma removes the
    degenerate likelihood spike, and the narrower alternative buys power."""
    y = np.asarray(y, float); n = y.size
    best = None
    q = np.quantile(y, [0.25, 0.75])
    inits = [np.array(q)] + [np.sort(rng.choice(y, 2, replace=False))
                             for _ in range(n_init - 1)]
    for mu0 in inits:
        mu = mu0.astype(float).copy()
        sd = max(y.std(), 1e-9); w = np.array([0.5, 0.5]); ll_old = -np.inf
        for _ in range(max_iter):
            logp = np.log(w)[None, :] + _norm_logpdf(y[:, None], mu[None, :], sd)
            m = logp.max(1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logp - m).sum(1))
            r = np.exp(logp - lse[:, None]); ll = float(lse.sum())
            nk = r.sum(0) + 1e-12
            w = nk / n
            mu = (r * y[:, None]).sum(0) / nk
            sd = np.sqrt((r * (y[:, None] - mu[None, :]) ** 2).sum() / n)
            sd = max(sd, 1e-9)
            if ll - ll_old < tol * max(1.0, abs(ll)):
                break
            ll_old = ll
        if best is None or ll > best[3]:
            best = (mu.copy(), sd, w.copy(), ll)
    mu, sd, w, ll1 = best
    o = np.argsort(mu); mu, w = mu[o], w[o]
    logp = np.log(w)[None, :] + _norm_logpdf(y[:, None], mu[None, :], sd)
    m = logp.max(1, keepdims=True); r = np.exp(logp - m); r /= r.sum(1, keepdims=True)
    ll0 = float(_norm_logpdf(y, y.mean(), max(y.std(), 1e-9)).sum())
    return dict(mu=mu, sd=sd, w=w, resp=r, lrt=2.0 * (ll1 - ll0))


def lts_residuals(Z, y, h=TRIM):
    A = np.column_stack([np.ones(len(Z)), Z])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    keep = np.argsort(np.abs(r))[: int(h * len(Z))]
    c2, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return y - A @ c2


def lda_direction(Z, gamma):
    """Fisher discriminant from soft responsibilities. Returns (unit normal,
    signed intercept) of the separating line in probe coordinates."""
    g = np.clip(gamma, 0, 1)
    n1, n2 = (1 - g).sum(), g.sum()
    if min(n1, n2) < 5:
        return None, None
    mu1 = ((1 - g)[:, None] * Z).sum(0) / n1
    mu2 = (g[:, None] * Z).sum(0) / n2
    dmu = mu2 - mu1
    C1 = ((1 - g)[:, None] * (Z - mu1)).T @ (Z - mu1) / max(n1, 1e-9)
    C2 = (g[:, None] * (Z - mu2)).T @ (Z - mu2) / max(n2, 1e-9)
    S = (n1 * C1 + n2 * C2) / (n1 + n2) + 1e-9 * np.eye(Z.shape[1])
    v = np.linalg.solve(S, dmu)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return None, None
    nu = v / nv
    # decision boundary: nu . z = t0, midway between the projected class means
    t0 = 0.5 * (nu @ mu1 + nu @ mu2)
    return nu, float(t0)


def anchor_primitive(Z, y, t_anchor, rng, permute=False):
    """Returns (n_a, c_a, w_a) in manifold coordinates, or None."""
    resid = lts_residuals(Z, y)
    if permute:
        resid = rng.permutation(resid)          # break z <-> response link
    fit = gmm2_equalvar(resid, rng)
    gamma = fit["resp"][:, 0]                   # lower-mean component = gated side
    nu, t0 = lda_direction(Z, gamma)
    if nu is None:
        return None
    # orient so the normal points AWAY from the gated (lower-mean) side
    if (gamma * (Z @ nu)).sum() / max(gamma.sum(), 1e-9) > (
            (1 - gamma) * (Z @ nu)).sum() / max((1 - gamma).sum(), 1e-9):
        nu, t0 = -nu, -t0
    c_a = float(nu @ t_anchor + t0)
    return nu, c_a, float(max(fit["lrt"], 0.0))


# ----------------------------------------------------------------- pooling
def _doubled(nvecs):
    th = np.arctan2(nvecs[:, 1], nvecs[:, 0])
    return np.column_stack([np.cos(2 * th), np.sin(2 * th)])


def pool_direction(nvecs, w):
    u = _doubled(nvecs)
    ubar = (w[:, None] * u).sum(0) / max(w.sum(), 1e-12)
    th = np.arctan2(ubar[1], ubar[0]) / 2.0
    return np.array([np.cos(th), np.sin(th)]), float(np.linalg.norm(ubar))


H_ANG = np.radians(30.0)      # doubled-angle bandwidth ~ 15 deg of true angle
H_OFF = 2.0 * SIGMA           # offset bandwidth


def agreement(nvecs, cs, w):
    """Smooth pairwise agreement in (direction, offset). Grid-free."""
    if len(nvecs) < 2:
        return 0.0
    u = _doubled(nvecs)
    ang = np.arctan2(u[:, 1], u[:, 0])
    d_ang = np.angle(np.exp(1j * (ang[:, None] - ang[None, :])))
    cc = cs - np.mean(cs)                       # centre: kill the origin effect
    d_off = cc[:, None] - cc[None, :]
    K = np.exp(-0.5 * (d_ang / H_ANG) ** 2) * np.exp(-0.5 * (d_off / H_OFF) ** 2)
    W = np.outer(w, w)
    np.fill_diagonal(K, 0.0); np.fill_diagonal(W, 0.0)
    return float((W * K).sum() / max((W.sum()), 1e-12))


def point_estimate(nvecs, cs, w):
    """Pooled (n_hat, c_hat): direction by weighted doubled-angle mean, then the
    sign resolved by majority vote before the offsets are combined."""
    nhat, _ = pool_direction(nvecs, w)
    sg = np.sign(nvecs @ nhat); sg[sg == 0] = 1
    chat = float((w * sg * cs).sum() / max(w.sum(), 1e-12))
    return nhat, chat


def estimator_C(nvecs, cs, w):
    """Fully geometric, weighted, no thresholding."""
    return agreement(nvecs, cs, w)


def estimator_B(nvecs, cs, w, w_thresh):
    """Hard geometric: threshold on w_a, then unweighted agreement."""
    keep = w >= w_thresh
    if keep.sum() < 2:
        return 0.0
    return agreement(nvecs[keep], cs[keep], np.ones(keep.sum())) * (keep.sum() / len(w))


def estimator_A(w, null_w):
    """Statistic only: Fisher combination of per-anchor p-values. No geometry."""
    p = np.array([(np.sum(null_w >= wi) + 1) / (len(null_w) + 1) for wi in w])
    return float(-2.0 * np.log(np.clip(p, 1e-12, 1.0)).sum())


# ----------------------------------------------------------------- generation
def build_pool(dt_ratio, pi_target, placement, rng):
    """Generate an anchor pool for one cell, plus a permuted (null) pool."""
    from scipy.stats import norm
    delta = dt_ratio * TAU
    n_true = np.array([np.cos(0.7), np.sin(0.7)])          # fixed global boundary
    c_true = 0.0
    beta = BETA_NORM * np.array([np.cos(2.1), np.sin(2.1)])
    d_shell = -SIGMA * norm.ppf(pi_target)                 # |distance| for this pi

    real, null, honest = [], [], []
    for _ in range(POOL):
        if placement == "design":
            # place AT the distance that realises pi_target, with light jitter.
            # (The earlier shell-draw rule collapsed all three pi targets onto
            #  the same band and made the pi axis degenerate.)
            dist = d_shell * rng.uniform(0.95, 1.05)
        else:
            dist = rng.uniform(0.0, RANDOM_SPAN) * SIGMA
        side = rng.choice([-1.0, 1.0])
        along = rng.normal(0.0, 4.0 * SIGMA)               # position along boundary
        perp = n_true * (side * dist)
        tang = np.array([-n_true[1], n_true[0]]) * along
        t_a = perp + tang + c_true * n_true

        Z = rng.normal(0.0, SIGMA, size=(M_PROBE, D_INTR))
        T = t_a[None, :] + Z
        crossed = (T @ n_true) > c_true
        y = T @ beta - delta * crossed + rng.normal(0.0, TAU, M_PROBE)

        out = anchor_primitive(Z, y, t_a, rng, permute=False)
        if out is not None:
            real.append(out)
        for _ in range(N_PERM_REP):
            o = anchor_primitive(Z, y, t_a, rng, permute=True)
            if o is not None:
                null.append(o)
        # matched no-gate control on identical geometry -> the EXISTENCE null
        y0 = T @ beta + rng.normal(0.0, TAU, M_PROBE)
        oh = anchor_primitive(Z, y0, t_a, rng, permute=False)
        if oh is not None:
            honest.append(oh)
    def unpack(lst):
        if not lst:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        return (np.array([a[0] for a in lst]), np.array([a[1] for a in lst]),
                np.array([a[2] for a in lst]))
    return unpack(real), unpack(null), unpack(honest), n_true, c_true


def run_cell(dt_ratio, pi_target, placement, rng):
    (nR, cR, wR), (nP, cP, wP), (nH, cH, wH), n_true, c_true = build_pool(
        dt_ratio, pi_target, placement, rng)
    need = 2 * max(N_LIST)
    if min(len(wR), len(wP), len(wH)) < need:
        return []
    w_thresh = np.quantile(wH, 0.95)                  # per-anchor 5% level, honest null
    rows = []
    for N in N_LIST:
        if min(len(wR), len(wP), len(wH)) < 2 * N:
            continue
        stat = {k: {"real": [], "honest": [], "perm": []} for k in "ABC"}
        errs = []
        for _ in range(N_REPEAT):
            draws = {"real": (nR, cR, wR), "honest": (nH, cH, wH), "perm": (nP, cP, wP)}
            for src, (nv, cv, wv) in draws.items():
                i = rng.choice(len(wv), N, replace=False)
                stat["A"][src].append(estimator_A(wv[i], wH))
                stat["B"][src].append(estimator_B(nv[i], cv[i], wv[i], w_thresh))
                stat["C"][src].append(estimator_C(nv[i], cv[i], wv[i]))
                if src == "real":
                    nh, ch = point_estimate(nv[i], cv[i], wv[i])
                    errs.append((np.degrees(np.arccos(np.clip(abs(nh @ n_true), -1, 1))),
                                 abs(ch - (c_true + np.mean(cv[i]) - np.mean(cv[i])))))
        for k in "ABC":
            for null_name in ("honest", "perm"):
                thr = np.quantile(stat[k][null_name], 1 - ALPHA)
                rows.append(dict(dt_ratio=dt_ratio, pi_target=pi_target,
                                 placement=placement, N=N, estimator=k,
                                 null=null_name,
                                 power=float(np.mean(np.array(stat[k]["real"]) > thr)),
                                 fp=float(np.mean(np.array(stat[k][null_name]) > thr))))
        if errs:
            e = np.array(errs)
            rows.append(dict(dt_ratio=dt_ratio, pi_target=pi_target,
                             placement=placement, N=N, estimator="point", null="-",
                             orient_err=float(np.median(e[:, 0])),
                             offset_err=float(np.median(e[:, 1]))))
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    out = []
    for placement in ["design", "random"]:
        pis = PI_TARGET if placement == "design" else [0.10]
        for dt in DELTA_OVER_TAU:
            for pi in pis:
                r = run_cell(dt, pi, placement, rng)
                out.extend(r)
                pw = {x["estimator"]: x.get("power") for x in r
                      if x.get("N") == 50 and x["estimator"] in "ABC"
                      and x.get("null") == "honest"}
                print(f"  {placement:6s} dt={dt:<4} pi={pi:<5} "
                      f"power@N=50  A={pw.get('A',float('nan')):.2f} "
                      f"B={pw.get('B',float('nan')):.2f} C={pw.get('C',float('nan')):.2f}",
                      flush=True)
    pd.DataFrame(out).to_csv("/tmp/pexp/p_pooling_rows.csv", index=False)
    print("\nwrote p_pooling_rows.csv")
