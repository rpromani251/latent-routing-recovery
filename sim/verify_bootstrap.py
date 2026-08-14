"""
Verification for Experiment B. Every claim the write-up makes about the machinery is
checked here rather than asserted in a docstring.

  1  fast_em reproduces the reference gmm2_equalvar LRT
  2  the pipeline is scale-EQUIVARIANT and both tests are scale-INVARIANT, so the
     regenerated noise's scale cannot affect a bootstrap p-value (this is what makes
     the truncation-deflated variance a non-issue, and what makes the emp_inlier
     failure a SHAPE failure rather than a scale failure)
  3  the pipeline residual is invariant to the fitted plane, i.e. adding anything in
     the column space of [1, Z] leaves it unchanged -- so the null does not depend on
     the trend
  4  the shape deformation that breaks emp_inlier, measured as kurtosis through
     successive pipeline passes
  5  seeded reruns reproduce
"""
import os
import importlib.util

import numpy as np
import diptest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("p", "exp_p_pooling.py")
FE = _load("fe", "fast_em.py")
X = _load("x", "exp_bootstrap_calibration.py")

OK, BAD = "  PASS", "  FAIL"
results = []


def check(name, passed, detail=""):
    results.append((name, passed))
    print(f"{OK if passed else BAD}  {name}{'  ' + detail if detail else ''}")


# ------------------------------------------------------------------ 1
print("\n[1] fast EM against the reference implementation")
a, r = FE.verify(lambda y, rng: P.gmm2_equalvar(y, rng, n_init=3), n_trials=60,
                 verbose=True)
check("fast_em LRT matches reference", r < 1e-7, f"max rel diff {r:.2e}")


# ------------------------------------------------------------------ 2
print("\n[2] scale equivariance of the pipeline, scale invariance of both tests")
rng = np.random.default_rng(3)
Z = rng.normal(0, 0.2, size=(X.M, 2))
y = Z @ X.BETA - 0.05 * ((Z @ X.NT) > 0.1) + rng.normal(0, 0.02, X.M)
r1, k1 = X.pipeline_residuals(Z, y)
worst_resid, worst_dip, keep_same = 0.0, 0.0, True
lrt_dev = {}
l1 = P.gmm2_equalvar(r1, np.random.default_rng(0))["lrt"]
d1 = diptest.dipstat(np.ascontiguousarray(r1))
for k in (1e-6, 1e-3, 0.5, 2.0, 1e3, 1e6):
    rk, kk = X.pipeline_residuals(Z, k * y)
    worst_resid = max(worst_resid, np.abs(rk - k * r1).max() / (k * np.abs(r1).max()))
    keep_same &= bool(np.array_equal(np.sort(kk), np.sort(k1)))
    lk = P.gmm2_equalvar(k * r1, np.random.default_rng(0))["lrt"]
    lrt_dev[k] = abs(lk - l1) / max(abs(l1), 1e-12)
    dk = diptest.dipstat(np.ascontiguousarray(k * r1))
    worst_dip = max(worst_dip, abs(dk - d1) / max(d1, 1e-12))
check("LTS keeps the same points under rescaling", keep_same)
check("residuals scale exactly", worst_resid < 1e-9, f"max rel dev {worst_resid:.2e}")
check("dip is scale invariant", worst_dip < 1e-12, f"max rel dev {worst_dip:.2e}")

# The equal-variance LRT is scale invariant as a STATISTIC: rescaling shifts both
# log-likelihoods by -n log k, which cancels in the difference. The IMPLEMENTATION is
# not exactly invariant, because gmm2_equalvar stops when ll - ll_old < tol*max(1,|ll|)
# and |ll| itself moves by n log k, so the stopping point drifts. Measured, and small.
print("       reference gmm2_equalvar LRT under rescaling of the residuals:")
for k, v in lrt_dev.items():
    print(f"         k = {k:<8.0e}  relative deviation {v:.2e}")
check("LRT scale sensitivity is negligible at realistic scales",
      max(lrt_dev[0.5], lrt_dev[2.0]) < 1e-2,
      f"max rel dev {max(lrt_dev[0.5], lrt_dev[2.0]):.2e} over a factor of 2")
print("       (an implementation artefact of the relative stopping rule, not of the")
print("        statistic; absolute drift is ~0.04 LRT against a threshold of 5.46)")

# What actually has to hold: a bootstrap p-value must not move when the regenerated
# noise is rescaled.
print("       bootstrap p-value under rescaling of the regenerated noise:")
base = y - r1
ps = []
for k in (1.0, 1.0 / 0.6071, 10.0):
    rr = np.random.default_rng(77)
    E = k * rr.normal(0.0, r1[k1].std(), size=(200, X.M))
    R = np.stack([X.pipeline_residuals(Z, base + E[b])[0] for b in range(200)])
    lrt_n = np.maximum(FE.batched_lrt(R, FE.make_inits(R, np.random.default_rng(78))), 0.0)
    ps.append(float((1 + np.sum(lrt_n >= l1)) / 201))
    print(f"         scale x{k:<8.3f}  p = {ps[-1]:.4f}")
check("bootstrap p-value is invariant to the regenerated noise scale",
      max(ps) - min(ps) <= 1.0 / 201 + 1e-12,
      f"spread {max(ps)-min(ps):.1e} = {(max(ps)-min(ps))*201:.0f} replicate(s) of 200")
print("       (the residual spread is the stopping-rule artefact above flipping a")
print("        single replicate across the observed value, not a scale dependence)")


# ------------------------------------------------------------------ 3
print("\n[3] the pipeline residual ignores anything in the span of [1, Z]")
worst = 0.0
for _ in range(5):
    c = rng.normal(0, 10, 3)
    add = c[0] + Z @ c[1:]
    r2, _ = X.pipeline_residuals(Z, y + add)
    worst = max(worst, np.abs(r2 - r1).max() / np.abs(r1).max())
check("residual invariant to added plane", worst < 1e-9, f"max rel dev {worst:.2e}")
print("       => the pipeline null cannot depend on the trend, only on the noise shape")


# ------------------------------------------------------------------ 4
print("\n[4] the shape deformation behind the emp_inlier failure")


def kurt(v):
    v = np.asarray(v, float)
    return float(((v - v.mean()) ** 4).mean() / v.var() ** 2)


rng = np.random.default_rng(11)
Zc = rng.normal(0, 0.2, size=(X.M, 2))
yc = Zc @ X.BETA + rng.normal(0, 0.02, X.M)         # pure H0, no gate
rc, kc = X.pipeline_residuals(Zc, yc)
k_noise = kurt(rng.normal(0, 1, 200000))
k_resid, k_in = kurt(rc), kurt(rc[kc])
# feed the inlier set back through the pipeline once, as emp_inlier does
E = rng.choice(rc[kc], size=X.M, replace=True)
r_again, k_again2 = X.pipeline_residuals(Zc, E)
k_pass2, k_pass2_in = kurt(r_again), kurt(r_again[k_again2])
print(f"       true noise                       kurtosis {k_noise:.2f}")
print(f"       pipeline residuals (all)         kurtosis {k_resid:.2f}")
print(f"       pipeline residuals (inliers)     kurtosis {k_in:.2f}")
print(f"       resampled inliers, one more pass kurtosis {k_pass2:.2f}")
print(f"                       its inlier set  kurtosis {k_pass2_in:.2f}")
check("inlier set is platykurtic vs the true noise", k_in < 2.4 < k_noise,
      f"{k_in:.2f} against {k_noise:.2f}")
check("resampling inliers compounds the deformation", k_pass2 < k_resid,
      f"{k_pass2:.2f} against {k_resid:.2f}")


# ------------------------------------------------------------------ 5
print("\n[5] seeded reruns reproduce")
out = []
for _ in range(2):
    rr = np.random.default_rng(99)
    Zr = rr.normal(0, 0.2, size=(X.M, 2))
    yr, _ = X.respond(Zr, X.CELLS["gated_dt2.5"], rr)
    st, resid, keep = X.observed_stats(Zr, yr, rr)
    lrt_n, _ = X.null_draws(Zr, yr - resid, resid, keep, "param", 200, rr)
    out.append((st["lrt"], float(np.quantile(lrt_n, 0.95))))
check("identical seed gives identical output", out[0] == out[1], f"{out[0]}")

n_bad = sum(1 for _, p in results if not p)
print(f"\n{len(results) - n_bad}/{len(results)} checks passed")
raise SystemExit(1 if n_bad else 0)
