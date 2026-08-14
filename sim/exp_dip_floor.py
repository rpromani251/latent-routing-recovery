"""
The dip's floor is a function of the mixing fraction, not a constant.

boundary_recovery_v5 sec.11 states the floor as a property of the mixture: "an
equal-variance two-component Gaussian mixture is literally unimodal until separation
exceeds about 2 standard deviations." That is the EQUAL-WEIGHT floor. The method's own
anchor placement targets pi <= 0.10 (step 2: orientation error is monotone increasing
in pi), and at pi = 0.10 a minority bump has to clear the majority's tail, which takes
far more separation.

This measures the floor as a surface over (pi, separation), for both the dip's tabled
p-value and a threshold calibrated on the pipeline's own null.
"""
import numpy as np, pandas as pd, diptest

N, REPS, SEED = 800, 400, 20260814
DIP_CALIB_Q95 = 0.01198          # pipeline null q95, from bootstrap_null_reference.npz
rows = []
rng = np.random.default_rng(SEED)
for pi in (0.50, 0.35, 0.25, 0.10, 0.05):
    for mu in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0):
        d, p = [], []
        for _ in range(REPS):
            k = rng.binomial(N, pi)
            x = np.concatenate([rng.normal(mu, 1, k), rng.normal(0, 1, N - k)])
            dd, pp = diptest.diptest(np.ascontiguousarray(x))
            d.append(dd); p.append(pp)
        d, p = np.array(d), np.array(p)
        rows.append(dict(pi=pi, sep=mu, dip_med=float(np.median(d)),
                         fire_calibrated=float(np.mean(d > DIP_CALIB_Q95)),
                         fire_table=float(np.mean(p < 0.05))))
        print(f"  pi={pi:<5} sep={mu:<4} dip {np.median(d):.5f}  "
              f"calibrated {rows[-1]['fire_calibrated']:.3f}  "
              f"table {rows[-1]['fire_table']:.3f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv("dip_floor_rows.csv", index=False)
print("\n  Separation at which the CALIBRATED dip first reaches power 0.5:")
for pi, g in df.groupby("pi"):
    g = g.sort_values("sep")
    hit = g[g.fire_calibrated >= 0.5]
    print(f"    pi={pi:<5} {hit.sep.iloc[0] if len(hit) else '>7'}")
print("\nwrote dip_floor_rows.csv")
