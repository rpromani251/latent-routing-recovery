"""
Analysis for Experiment B -- the full-pipeline bootstrap.

Reads bootstrap_nulls.csv, bootstrap_null_reference.npz, bootstrap_anchor_stats.csv
and bootstrap_anchor_pvalues.csv, and prints the tables the write-up quotes. Nothing
here recomputes a statistic; it only aggregates, so a disagreement between this and
the run is a reporting bug, not a physics one.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import kstest, beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
RUNGS = [1.0, 0.5, 0.25]
# Experiment A's per-rung LRT thresholds, from commit 125555a
A_THRESHOLDS = {1.0: 5.50, 0.5: 5.35, 0.25: 4.60}


def ci(k, n, conf=0.95):
    """Clopper-Pearson interval for a rate."""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta_dist.ppf((1 - conf) / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - (1 - conf) / 2, k + 1, n - k)
    return float(lo), float(hi)


def rate(mask):
    k, n = int(np.sum(mask)), int(len(mask))
    lo, hi = ci(k, n)
    return k / n if n else np.nan, lo, hi, n


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def load(name):
    p = os.path.join(HERE, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    nulls = load("bootstrap_nulls.csv")
    stats = load("bootstrap_anchor_stats.csv")
    pv = load("bootstrap_anchor_pvalues.csv")
    refp = os.path.join(HERE, "bootstrap_null_reference.npz")
    ref = np.load(refp) if os.path.exists(refp) else None

    # ------------------------------------------------ 1. what the null actually is
    if nulls is not None:
        hdr("1. THE PIPELINE NULL, AND WHETHER IT DEPENDS ON ANYTHING")
        rung = nulls[nulls.condition.str.startswith("rung")].copy()
        rung["k"] = rung.condition.str.replace("rung_", "").astype(float)
        print("\n  LRT null by rung and replicate, param arm (Gaussian regeneration):")
        print(f"    {'rung':>6}  {'rep':>3}  {'B':>6}  {'median':>8}  {'q95':>8}  {'q99':>8}")
        p = rung[rung.arm == "param"].sort_values(["k", "rep"], ascending=[False, True])
        for _, r in p.iterrows():
            print(f"    {r.k:>6}  {int(r.rep):>3}  {int(r.B):>6}  "
                  f"{r.lrt_med:>8.2f}  {r.lrt_q95:>8.2f}  {r.lrt_q99:>8.2f}")
        by_rung = p.groupby("k").lrt_q95.mean()
        within = p.groupby("k").lrt_q95.std().mean()
        between = by_rung.std()
        print(f"\n    between-rung sd of q95 {between:.3f}   "
              f"within-rung (replicate) sd {within:.3f}")
        print("    => rung structure is not distinguishable from replicate scatter"
              if between <= 1.5 * within else
              "    => rung structure EXCEEDS replicate scatter")
        print(f"\n    For comparison, Experiment A reported per-rung thresholds "
              f"{A_THRESHOLDS[1.0]} / {A_THRESHOLDS[0.5]} / {A_THRESHOLDS[0.25]}")
        print(f"    (spread {max(A_THRESHOLDS.values()) - min(A_THRESHOLDS.values()):.2f}, "
              f"from B=300 calibration anchors each).")

        print("\n  Invariance conditions, param arm:")
        other = nulls[(~nulls.condition.str.startswith("rung")) & (nulls.arm == "param")]
        for _, r in other.iterrows():
            print(f"    {r.condition:22s} B={int(r.B):<6} median {r.lrt_med:7.2f}  "
                  f"q95 {r.lrt_q95:7.2f}")

        print("\n  The three null-generating arms (honest surface, no gate):")
        print(f"    {'arm':12s} {'median':>9} {'q95':>9}   {'resid kurt':>10} {'inlier kurt':>11}")
        for arm in ("param", "emp_all", "emp_inlier"):
            s = rung[rung.arm == arm]
            if len(s):
                print(f"    {arm:12s} {s.lrt_med.median():>9.2f} {s.lrt_q95.median():>9.2f}"
                      f"   {s.resid_kurt.median():>10.2f} {s.inlier_kurt.median():>11.2f}")
        cg = nulls[nulls.condition == "clean_gauss_untrimmed"]
        if len(cg):
            r = cg.iloc[0]
            print(f"    {'clean (untrimmed)':12s} {r.lrt_med:>9.2f} {r.lrt_q95:>9.2f}"
                  "   <- the threshold v5 says not to reuse")

    if ref is not None:
        hdr("2. THE REFERENCE THRESHOLDS")
        lp, lc = ref["lrt_pipeline"], ref["lrt_clean"]
        dp, dc = ref["dip_pipeline"], ref["dip_clean"]
        print(f"\n  pipeline null   B={len(lp):<6} LRT q95 {np.quantile(lp,.95):7.3f}   "
              f"q99 {np.quantile(lp,.99):7.3f}   median {np.median(lp):6.3f}")
        print(f"  clean Gaussian  B={len(lc):<6} LRT q95 {np.quantile(lc,.95):7.3f}   "
              f"q99 {np.quantile(lc,.99):7.3f}   median {np.median(lc):6.3f}")
        print(f"\n  Using the clean-Gaussian q95 as a threshold on the pipeline's own null "
              f"fires at {np.mean(lp >= np.quantile(lc,.95)):.4f} rather than {ALPHA}")
        print(f"  Using Experiment A's rung-1 threshold ({A_THRESHOLDS[1.0]}) fires at "
              f"{np.mean(lp >= A_THRESHOLDS[1.0]):.4f}")
        print(f"\n  dip: pipeline q95 {np.quantile(dp,.95):.5f}  "
              f"clean q95 {np.quantile(dc,.95):.5f}")

    # ------------------------------------------------ 3. size, power, confounders
    if stats is not None:
        thr = float(np.quantile(ref["lrt_pipeline"], 1 - ALPHA)) if ref is not None else np.nan
        thr_clean = float(np.quantile(ref["lrt_clean"], 1 - ALPHA)) if ref is not None else np.nan
        dthr = float(np.quantile(ref["dip_pipeline"], 1 - ALPHA)) if ref is not None else np.nan

        hdr("3. SIZE, POWER AND CONFOUNDER FALSE POSITIVES (top rung)")
        print(f"\n  Rules compared, all at nominal {ALPHA}:")
        print(f"    calibrated   LRT > {thr:.2f}      (pipeline null, this experiment)")
        print(f"    clean        LRT > {thr_clean:.2f}      (untrimmed Gaussian -- v5 warns against)")
        print(f"    expA         LRT > per-rung {A_THRESHOLDS}")
        print(f"    dip table    diptest p < {ALPHA}   dip calibrated  dip > {dthr:.5f}")
        print(f"\n  {'cell':18s} {'rung':>5} {'calibrated':>18} {'clean':>10} {'expA':>10}"
              f" {'dipTable':>10} {'dipCalib':>10}")
        for cell in stats.cell.unique():
            for k in RUNGS:
                s = stats[(stats.cell == cell) & (stats.rung == k)]
                if not len(s):
                    continue
                f1, lo, hi, n = rate(s.lrt.values > thr)
                f2 = np.mean(s.lrt.values > thr_clean)
                f3 = np.mean(s.lrt.values > A_THRESHOLDS[k])
                f4 = np.mean(s.dip_p_table.values < ALPHA)
                f5 = np.mean(s.dip.values > dthr)
                print(f"  {cell:18s} {k:>5} {f1:>8.3f} [{lo:.3f},{hi:.3f}] "
                      f"{f2:>10.3f} {f3:>10.3f} {f4:>10.3f} {f5:>10.3f}")

        # ------------------------------------------- 4. estimability
        hdr("4. DOES THE CALIBRATED TEST SUPPLY THE ESTIMABILITY GATE?")
        print("\n  The minimum-mass rule asks pi_hat >= 0.05. The question is whether it")
        print("  fires where the gate is NOT reached -- where the truth is 'no crossers'.")
        print(f"\n  {'cell':18s} {'rung':>5} {'d/sigma':>8} {'true pi':>9} "
              f"{'minmass':>9} {'calibrated':>11} {'both':>7}")
        for cell in stats.cell.unique():
            for k in RUNGS:
                s = stats[(stats.cell == cell) & (stats.rung == k)]
                if not len(s):
                    continue
                fire = s.lrt.values > thr
                mm = s.minmass_pass.values.astype(bool)
                print(f"  {cell:18s} {k:>5} {s.d_over_sigma.median():>8.2f} "
                      f"{s.pi_true.mean():>9.5f} {mm.mean():>9.3f} {fire.mean():>11.3f} "
                      f"{np.mean(mm & fire):>7.3f}")
        print("\n  Rows where true pi is ~0 are the diagnostic ones: the min-mass column")
        print("  is the rate at which EM splits noise and passes the floor anyway.")

    # ------------------------------------------------ 5. per-anchor bootstrap
    if pv is not None:
        hdr("5. WHAT THE PER-ANCHOR B=300 BOOTSTRAP ADDS")
        for cell in pv.cell.unique():
            for arm in ("param", "emp_inlier", "emp_all"):
                col = f"p_lrt_{arm}"
                if col not in pv.columns:
                    continue
                s = pv[(pv.cell == cell) & pv[col].notna()]
                if not len(s):
                    continue
                print(f"\n  {cell} / {arm}   (n = {len(s)} anchor-rungs)")
                for k in RUNGS:
                    ss = s[s.rung == k]
                    if not len(ss):
                        continue
                    f, lo, hi, n = rate(ss[col].values <= ALPHA)
                    q = ss[f"q95_lrt_{arm}"].values
                    line = (f"    rung {k:<5} fire {f:.3f} [{lo:.3f},{hi:.3f}] n={n:<4} "
                            f"per-anchor q95: median {np.median(q):7.2f} "
                            f"sd {np.std(q):6.2f}")
                    if cell == "honest":
                        ks = kstest(ss[col].values, "uniform")
                        line += f"   KS p={ks.pvalue:.3f}"
                    print(line)
        if ref is not None:
            hdr("6. HOW MUCH OF THE PER-ANCHOR SPREAD IS JUST B=300 MONTE CARLO")
            lp = ref["lrt_pipeline"]
            sub = np.array([np.quantile(np.random.default_rng(i).choice(lp, 300), 0.95)
                            for i in range(400)])
            print(f"\n  Resampling B=300 from the B={len(lp)} reference null gives")
            print(f"  q95 estimates with sd {sub.std():.2f} "
                  f"(2.5-97.5 pct: {np.quantile(sub,.025):.2f} to {np.quantile(sub,.975):.2f}).")
            s = pv[(pv.cell == "honest") & pv.get("q95_lrt_param", pd.Series(dtype=float)).notna()]
            if len(s):
                print(f"  Observed per-anchor q95 sd on honest anchors: "
                      f"{s.q95_lrt_param.std():.2f}")
                print("  If these match, the per-anchor variation is Monte-Carlo noise, not")
                print("  anchor structure -- and the bootstrap can be precomputed once.")


if __name__ == "__main__":
    main()
