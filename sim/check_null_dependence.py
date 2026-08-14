"""
Does the Stage-3 null depend on the pipeline, or only on the sample size?

boundary_recovery_v5 sec.10 justifies the full-pipeline bootstrap by asserting that
"LTS truncates the tails before the LRT sees them, which deflates the null". But
Stage 2 residualises ALL points against the refit plane -- the trim chooses the FIT,
not the test sample -- so the LRT may see an essentially untouched Gaussian sample.
This checks that directly, at matched B and matched sample size, and then sweeps the
one thing that plausibly does move the null: m.
"""
import importlib.util, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def L(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
FE = L("fe", "fast_em.py"); X = L("x", "exp_bootstrap_calibration.py")

B = 6000
print(f"\nPipeline null vs clean-Gaussian null, matched B={B} and matched m")
print(f"  {'m':>6}  {'pipeline med':>13} {'q95':>7} {'q99':>7} | "
      f"{'clean med':>10} {'q95':>7} {'q99':>7}")
for m in (400, 800, 1000, 2000):
    rng = np.random.default_rng(20260814 + m)
    Z = rng.normal(0.0, X.SIGMA, size=(m, 2))
    base = Z @ X.BETA
    resid, keep = X.pipeline_residuals(Z, base + rng.normal(0.0, X.TAU, m))
    lp, _ = X.null_draws(Z, base, resid, keep, "param", B, rng)
    Yc = rng.normal(0.0, 1.0, size=(B, m))
    lc = np.maximum(FE.batched_lrt(Yc, FE.make_inits(Yc, rng)), 0.0)
    print(f"  {m:>6}  {np.median(lp):>13.2f} {np.quantile(lp,.95):>7.2f} "
          f"{np.quantile(lp,.99):>7.2f} | {np.median(lc):>10.2f} "
          f"{np.quantile(lc,.95):>7.2f} {np.quantile(lc,.99):>7.2f}", flush=True)
print("\n  If the two halves agree at every m, the trim does not deflate the null and")
print("  the stated reason for a full-pipeline bootstrap does not hold; any threshold")
print("  mismatch is a sample-size mismatch instead.")
