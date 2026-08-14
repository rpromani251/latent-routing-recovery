"""
Experiment B -- the full-pipeline bootstrap.  B = 300.

WHAT WAS REGISTERED. boundary_recovery_v5 sec.9 (Stage 3) specifies, and sec.17 lists
as an open item ("no full-pipeline bootstrap yet"), an anchor-conditional calibration:

    "Resample inlier residuals, regenerate responses, rerun filter -> trim -> refit ->
     residualise -> both tests, B = 300. Trimming is a nonlinear, data-dependent
     operator, so anything calibrated against untrimmed theory is miscalibrated. In
     particular, do not reuse a threshold derived from clean Gaussian draws: LTS
     truncates the tails before the LRT sees them, which deflates the null. Costs zero
     model queries."

WHY IT IS NOT WHAT EXPERIMENT A ALREADY DID. Commit 125555a recalibrated the LRT per
rung on trimmed residuals from honest-smooth anchors (5.50 / 5.35 / 4.60). That is a
GLOBAL threshold read off a surface known to be honest -- an oracle no auditor has.
The calibration here is generated from the anchor's own probes and no others, so it is
deployable; A's arm is kept as a reference to show what the oracle was worth.

THREE STAGES, run in this order.

  nulls    High-B pipeline nulls under three null-generating arms, plus the invariance
           conditions. Answers: what IS the pipeline's null, and does it depend on the
           rung, the noise scale, or the trend?
  anchors  Observed statistics only, on the full surface grid at high anchor count.
           Cheap (one EM per anchor-rung), so size and power get tight error bars.
  boot     The registered per-anchor B = 300 bootstrap on a subset of anchors, all
           three arms. Answers: does conditioning on the anchor buy anything, and how
           much Monte-Carlo noise does B = 300 itself carry?

THE THREE ARMS. All run the identical downstream pipeline and differ only in the
distribution of the regenerated noise.
    emp_inlier    resample the LTS inlier residuals with replacement   <- v5's wording
    emp_all       resample ALL residuals with replacement
    param         Gaussian draws at the robust scale

Scale is irrelevant to all three: LTS keeps the same points under a common rescaling,
so residuals are scale-equivariant, and both the equal-variance LRT and the dip are
scale-invariant. SHAPE is not irrelevant, and that is where the registered recipe runs
into trouble -- the inlier set is the truncated middle 75% of the residuals, and
resampling it feeds an already-truncated shape back through a pipeline that truncates
again. Both facts are checked in verify_bootstrap.py rather than assumed here.

THE ESTIMABILITY QUESTION (carried from commit 125555a, recorded only there). The
minimum-mass rule does NOT enforce estimability: EM splits noise and returns pi_hat
above the 0.05 floor at rungs the gate never reaches, so the "deepest estimable rung"
convention does not self-enforce as written. Every anchor row records pi_hat, the
min-mass verdict, the TRUE crossing count, and the calibrated p-value, so the two
candidate gates can be compared exactly where the truth is "no crossers at all".

SCOPE. The density filter is a no-op in this flat generative setting (every probe is
retained), so the chain calibrated here is trim -> refit -> residualise -> {dip, LRT}.
Intrinsic d = 2, frame supplied, Stage 0 deferred, as in Experiments P and A.

    python3 exp_bootstrap_calibration.py nulls
    python3 exp_bootstrap_calibration.py anchors
    python3 exp_bootstrap_calibration.py boot
"""
import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import diptest
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("p", "exp_p_pooling.py")          # lts_residuals, gmm2_equalvar
FE = _load("fe", "fast_em.py")              # batched LRT for the null replicates

# ------------------------------------------------------------------ settings
# Matched to exp_a_invariance.py so the two compose.
SIGMA, TAU, M = 0.20, 0.02, 800
L_PROBE = 2.0 * SIGMA * np.sqrt(2)          # probe diameter, ~0.566
RUNGS = [1.0, 0.5, 0.25]                    # multiples of sigma_top
PI_MIN = 0.05                               # the minimum-mass rule
TRIM = 0.75
B_BOOT = 300                                # registered
B_REF = 20000                               # the two reference nulls
B_INV = 6000                                # the invariance sweep: enough to resolve
                                            # a q95 to well inside the effect sought
B_DIAG = 3000                               # diagnostic arms: the effect is ~100x,
                                            # so it does not need B_INV to be seen
ALPHA = 0.05
SEED = 20260814

NT = np.array([np.cos(0.7), np.sin(0.7)])   # true gate normal
TG = np.array([-NT[1], NT[0]])
BETA = 0.15 * np.array([np.cos(2.1), np.sin(2.1)])

ARMS = ("param", "emp_inlier", "emp_all")

# sd of a standard normal truncated to its central 75%. Used only to put the param
# arm on a comparable scale; irrelevant to the p-values, and that irrelevance is
# itself one of the things checked.
_Q = norm.ppf(0.5 + TRIM / 2.0)
TRUNC_SD = float(np.sqrt(1.0 - 2.0 * _Q * norm.pdf(_Q) / TRIM))

CELLS = {
    "honest":           dict(dt=None, amp=0.0,       ell=None,           pi=None),
    "resonant_0.5L":    dict(dt=None, amp=2.5 * TAU, ell=0.5 * L_PROBE,  pi=None),
    "resonant_1.0L":    dict(dt=None, amp=2.5 * TAU, ell=1.0 * L_PROBE,  pi=None),
    "resonant_1.5L":    dict(dt=None, amp=2.5 * TAU, ell=1.5 * L_PROBE,  pi=None),
    "gated_dt1.5":      dict(dt=1.5,  amp=0.0,       ell=None,           pi=0.10),
    "gated_dt1.95":     dict(dt=1.95, amp=0.0,       ell=None,           pi=0.10),
    "gated_dt2.5":      dict(dt=2.5,  amp=0.0,       ell=None,           pi=0.10),
    "gated_dt2.5_pi35": dict(dt=2.5,  amp=0.0,       ell=None,           pi=0.35),
    "gated_dt5.0":      dict(dt=5.0,  amp=0.0,       ell=None,           pi=0.10),
}
N_ANCHOR = 400                              # for the observed-statistics grid
N_BOOT_ANCHOR = 100                         # for the per-anchor bootstrap (param)
N_BOOT_ARMS = 60                            # for the three-arm comparison
BOOT_CELLS = ["honest", "gated_dt1.95"]


# ------------------------------------------------------------------ generative
def respond(T, cell, rng):
    """Model response at probe locations T, plus the TRUE crossing mask."""
    y = T @ BETA + rng.normal(0.0, TAU, len(T))
    crossed = (T @ NT) > 0.0
    if cell["dt"] is not None:
        y = y - cell["dt"] * TAU * crossed
    if cell["amp"] > 0:
        y = y + cell["amp"] * np.sin(2 * np.pi * (T @ NT) / cell["ell"])
    return y, crossed


def place_anchor(cell, rng):
    if cell["dt"] is None:
        return NT * rng.normal(0, 0.8) + TG * rng.normal(0, 0.8)
    d0 = -SIGMA * norm.ppf(cell["pi"])
    return (NT * (d0 * rng.uniform(0.95, 1.05) * rng.choice([-1.0, 1.0]))
            + TG * rng.normal(0, 0.8))


# ------------------------------------------------------------------ pipeline
def pipeline_residuals(Z, y):
    """OLS -> trim 25% -> refit -> residualise ALL points. Returns (resid, keep)."""
    A = np.column_stack([np.ones(len(Z)), Z])
    c0, *_ = np.linalg.lstsq(A, y, rcond=None)
    keep = np.argsort(np.abs(y - A @ c0))[: int(TRIM * len(Z))]
    c2, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return y - A @ c2, keep


def observed_stats(Z, y, rng):
    resid, keep = pipeline_residuals(Z, y)
    fit = P.gmm2_equalvar(resid, rng)
    dip, dip_p_table = diptest.diptest(np.ascontiguousarray(resid))
    return dict(lrt=float(max(fit["lrt"], 0.0)), dip=float(dip),
                dip_p_table=float(dip_p_table), pi_hat=float(min(fit["w"])),
                gap=float(abs(fit["mu"][1] - fit["mu"][0])),
                resid_sd=float(resid.std()), inlier_sd=float(resid[keep].std())), resid, keep


def null_draws(Z, base, resid, keep, arm, B, rng):
    """B replicates of the FULL pipeline under the chosen null-generating arm."""
    if arm == "param":
        E = rng.normal(0.0, resid[keep].std() / TRUNC_SD, size=(B, len(Z)))
    elif arm == "emp_inlier":
        E = rng.choice(resid[keep], size=(B, len(Z)), replace=True)
    elif arm == "emp_all":
        E = rng.choice(resid, size=(B, len(Z)), replace=True)
    else:
        raise ValueError(arm)
    R = np.empty((B, len(Z)))
    for b in range(B):
        R[b] = pipeline_residuals(Z, base + E[b])[0]
    lrt = np.maximum(FE.batched_lrt(R, FE.make_inits(R, rng)), 0.0)
    dip = np.array([diptest.dipstat(np.ascontiguousarray(R[b])) for b in range(B)])
    return lrt, dip


# ======================================================== stage 1: the nulls
def _null_unit(args):
    i, c, arm = args
    path = os.path.join(HERE, "_parts_nulls", f"{c['tag']}__{c['rep']}__{arm}.csv")
    if os.path.exists(path):
        return pd.read_csv(path).iloc[0].to_dict()
    rng = np.random.default_rng(SEED + 991 * i + 17 * ARMS.index(arm))
    Z = rng.normal(0.0, c["sig"], size=(M, 2))
    base = c["beta"] * (Z @ BETA)
    resid, keep = pipeline_residuals(Z, base + rng.normal(0.0, c["tau"], M))
    B = (B_INV if arm == "param" else B_DIAG)
    if not c["tag"].startswith("rung"):
        B = B // 4
    lrt, dip = null_draws(Z, base, resid, keep, arm, B, rng)
    row = dict(condition=c["tag"], rep=c["rep"], arm=arm, B=B,
                lrt_med=float(np.median(lrt)), lrt_q90=float(np.quantile(lrt, 0.90)),
                lrt_q95=float(np.quantile(lrt, 0.95)),
                lrt_q99=float(np.quantile(lrt, 0.99)),
                dip_med=float(np.median(dip)), dip_q95=float(np.quantile(dip, 0.95)),
                resid_kurt=float(((resid - resid.mean()) ** 4).mean() / resid.var() ** 2),
                inlier_kurt=float(((resid[keep] - resid[keep].mean()) ** 4).mean()
                                  / resid[keep].var() ** 2))
    pd.DataFrame([row]).to_csv(path, index=False)
    return row


def stage_nulls():
    os.makedirs(os.path.join(HERE, "_parts_nulls"), exist_ok=True)
    """High-B pipeline nulls, and the invariance conditions.

    Conditions vary the rung, the noise scale, the trend magnitude and the anchor's
    probe draw. If the null is invariant to all of them, the calibration is a single
    number per pipeline rather than a per-anchor, per-rung object -- which changes
    both what B = 300 is worth and what it costs to deploy.
    """
    conds = ([dict(tag=f"rung_{k}", sig=k * SIGMA, tau=TAU, beta=1.0, rep=r)
              for k in RUNGS for r in range(3)]
             + [dict(tag="tau_x10", sig=SIGMA, tau=10 * TAU, beta=1.0, rep=0),
                dict(tag="tau_d100", sig=SIGMA, tau=TAU / 100, beta=1.0, rep=0),
                dict(tag="trend_x100", sig=SIGMA, tau=TAU, beta=100.0, rep=0),
                dict(tag="trend_zero", sig=SIGMA, tau=TAU, beta=0.0, rep=0)])
    units = [(i, c, arm) for i, c in enumerate(conds) for arm in ARMS]
    import multiprocessing as mp
    nproc = min(int(os.environ.get("NPROC", "2")), max(1, mp.cpu_count()))
    with mp.Pool(nproc) as pool:
        rows = []
        for r in pool.imap_unordered(_null_unit, units):
            rows.append(r)
            print(f"  {r['condition']:12s} rep{r['rep']} {r['arm']:11s} B={r['B']:<6} "
                  f"med {r['lrt_med']:7.2f}  q95 {r['lrt_q95']:7.2f}", flush=True)

    # the thing v5 says not to reuse: LRT on UNTRIMMED clean Gaussian draws
    rng = np.random.default_rng(SEED + 4242)
    Yc = rng.normal(0.0, 1.0, size=(B_REF, M))
    lrt_c = np.maximum(FE.batched_lrt(Yc, FE.make_inits(Yc, rng)), 0.0)
    dip_c = np.array([diptest.dipstat(np.ascontiguousarray(Yc[b])) for b in range(B_REF)])
    rows.append(dict(condition="clean_gauss_untrimmed", rep=0, arm="none", B=B_REF,
                     lrt_med=float(np.median(lrt_c)), lrt_q90=float(np.quantile(lrt_c, .90)),
                     lrt_q95=float(np.quantile(lrt_c, .95)),
                     lrt_q99=float(np.quantile(lrt_c, .99)),
                     dip_med=float(np.median(dip_c)), dip_q95=float(np.quantile(dip_c, .95)),
                     resid_kurt=np.nan, inlier_kurt=np.nan))
    print(f"  clean_gauss (untrimmed)      med {np.median(lrt_c):7.2f}  "
          f"q95 {np.quantile(lrt_c, .95):7.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "bootstrap_nulls.csv"), index=False)

    # the reference pipeline null itself, kept in full so thresholds at any level can
    # be read off later without a rerun
    rng = np.random.default_rng(SEED + 31337)
    Z = rng.normal(0.0, SIGMA, size=(M, 2))
    base = Z @ BETA
    resid, keep = pipeline_residuals(Z, base + rng.normal(0.0, TAU, M))
    lrt_ref, dip_ref = null_draws(Z, base, resid, keep, "param", B_REF, rng)
    np.savez_compressed(os.path.join(HERE, "bootstrap_null_reference.npz"),
                        lrt_pipeline=lrt_ref, dip_pipeline=dip_ref,
                        lrt_clean=lrt_c, dip_clean=dip_c)
    # .npz is gitignored, so the quantiles that actually get used are also written
    # to a committed CSV; the npz is a regenerable cache of the full draws.
    qs = [0.50, 0.90, 0.95, 0.975, 0.99, 0.995]
    pd.DataFrame([dict(null=nm, stat=st, B=len(v),
                       **{f"q{int(q*1000):03d}": float(np.quantile(v, q)) for q in qs})
                  for nm, st, v in [("pipeline", "lrt", lrt_ref), ("pipeline", "dip", dip_ref),
                                    ("clean_untrimmed", "lrt", lrt_c),
                                    ("clean_untrimmed", "dip", dip_c)]]
                 ).to_csv(os.path.join(HERE, "bootstrap_reference_thresholds.csv"), index=False)
    print(f"\nwrote bootstrap_nulls.csv ({len(df)} rows) + bootstrap_null_reference.npz")
    print(f"  reference pipeline null (B={B_REF}): "
          f"LRT q95 {np.quantile(lrt_ref, .95):.2f}, dip q95 {np.quantile(dip_ref, .95):.5f}")


# =================================================== stage 2: observed stats
def _anchor_unit(args):
    cell_name, chunk, n = args
    path = os.path.join(HERE, "_parts_anchors", f"{cell_name}__{chunk:03d}.csv")
    if os.path.exists(path):
        return path
    cell = CELLS[cell_name]
    rng = np.random.default_rng(SEED + 13 * (abs(hash(cell_name)) % 9973) + 7919 * chunk)
    rows = []
    for a in range(n):
        t_a = place_anchor(cell, rng)
        d_true = float(abs(t_a @ NT))
        for k in RUNGS:
            s = k * SIGMA
            Z = rng.normal(0.0, s, size=(M, 2))
            y, crossed = respond(t_a[None, :] + Z, cell, rng)
            st, _, _ = observed_stats(Z, y, rng)
            n_cross = int(min(crossed.sum(), M - crossed.sum())) if cell["dt"] else 0
            st.update(cell=cell_name, chunk=chunk, anchor=a, rung=k, sigma_s=s,
                      d_true=d_true, d_over_sigma=d_true / s,
                      pi_true=n_cross / M, n_cross=n_cross,
                      minmass_pass=bool(st["pi_hat"] >= PI_MIN))
            rows.append(st)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def stage_anchors():
    os.makedirs(os.path.join(HERE, "_parts_anchors"), exist_ok=True)
    units = [(c, i, 50) for c in CELLS for i in range(N_ANCHOR // 50)]
    _run_pool(_anchor_unit, units, "_parts_anchors")
    frames = [pd.read_csv(os.path.join(HERE, "_parts_anchors", f"{u[0]}__{u[1]:03d}.csv"))
              for u in units]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(HERE, "bootstrap_anchor_stats.csv"), index=False)
    print(f"\nwrote bootstrap_anchor_stats.csv ({len(df)} rows)")


# ================================================ stage 3: per-anchor bootstrap
def _boot_unit(args):
    cell_name, chunk, n, arms = args
    tag = "+".join(a[:3] for a in arms)
    path = os.path.join(HERE, "_parts_boot", f"{cell_name}__{tag}__{chunk:03d}.csv")
    if os.path.exists(path):
        return path
    cell = CELLS[cell_name]
    rng = np.random.default_rng(SEED + 29 * (abs(hash(cell_name + tag)) % 9973) + 7919 * chunk)
    rows = []
    for a in range(n):
        t_a = place_anchor(cell, rng)
        for k in RUNGS:
            s = k * SIGMA
            Z = rng.normal(0.0, s, size=(M, 2))
            y, crossed = respond(t_a[None, :] + Z, cell, rng)
            st, resid, keep = observed_stats(Z, y, rng)
            A = np.column_stack([np.ones(M), Z])
            base = y - resid                      # the fitted majority-branch plane
            for arm in arms:
                lrt_n, dip_n = null_draws(Z, base, resid, keep, arm, B_BOOT, rng)
                st[f"p_lrt_{arm}"] = float((1 + np.sum(lrt_n >= st["lrt"])) / (B_BOOT + 1))
                st[f"p_dip_{arm}"] = float((1 + np.sum(dip_n >= st["dip"])) / (B_BOOT + 1))
                st[f"q95_lrt_{arm}"] = float(np.quantile(lrt_n, 0.95))
                st[f"med_lrt_{arm}"] = float(np.median(lrt_n))
            n_cross = int(min(crossed.sum(), M - crossed.sum())) if cell["dt"] else 0
            st.update(cell=cell_name, chunk=chunk, anchor=a, rung=k, sigma_s=s,
                      pi_true=n_cross / M, n_cross=n_cross,
                      minmass_pass=bool(st["pi_hat"] >= PI_MIN))
            rows.append(st)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def stage_boot():
    os.makedirs(os.path.join(HERE, "_parts_boot"), exist_ok=True)
    units = []
    for c in BOOT_CELLS:
        units += [(c, i, 10, ("param",)) for i in range(N_BOOT_ANCHOR // 10)]
        units += [(c, 100 + i, 10, ("emp_inlier", "emp_all")) for i in range(N_BOOT_ARMS // 10)]
    _run_pool(_boot_unit, units, "_parts_boot",
              namer=lambda u: f"{u[0]}__{'+'.join(a[:3] for a in u[3])}__{u[1]:03d}.csv")
    frames = [pd.read_csv(os.path.join(HERE, "_parts_boot",
                                       f"{u[0]}__{'+'.join(a[:3] for a in u[3])}__{u[1]:03d}.csv"))
              for u in units]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(HERE, "bootstrap_anchor_pvalues.csv"), index=False)
    print(f"\nwrote bootstrap_anchor_pvalues.csv ({len(df)} rows)")


def _run_pool(fn, units, partdir, namer=None):
    import multiprocessing as mp
    namer = namer or (lambda u: f"{u[0]}__{u[1]:03d}.csv")
    todo = [u for u in units if not os.path.exists(os.path.join(HERE, partdir, namer(u)))]
    print(f"{len(units)} units, {len(todo)} to run", flush=True)
    if not todo:
        return
    nproc = min(int(os.environ.get("NPROC", "2")), max(1, mp.cpu_count()))
    with mp.Pool(nproc) as pool:
        for i, p in enumerate(pool.imap_unordered(fn, todo), 1):
            print(f"  [{i}/{len(todo)}] {os.path.basename(p)}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("nulls", "all"):
        stage_nulls()
    if stage in ("anchors", "all"):
        stage_anchors()
    if stage in ("boot", "all"):
        stage_boot()
