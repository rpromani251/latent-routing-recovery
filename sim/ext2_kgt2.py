"""
E2 — K > 2 regimes: model selection and multi-penalty recovery.

Model: f(x) = beta x - 0.25*1[x >= 1/3] - 0.50*1[x >= 2/3]; three regimes at
levels {0, -0.25, -0.75}; adjacent gaps 0.25 and 0.50.

PROPOSITION P2. With adjacent level gaps exceeding the local resolvable width
(A11 + noise), (a) the BIC-selected K at the most significant scale recovers
the number of regimes the probe actually overlaps (1 in interiors, 2 near one
boundary, 3 when the probe spans both); (b) adjacent-gap estimates are
consistent for the gaps the probe resolves; (c) in the scalar case, matching
local components to global regimes reduces to clustering trend-adjusted
component levels, because within-regime drift over a probe width is small
against the gaps (B5 + margin) -- a mini-synchronization with no assignment
machinery.

Scoring: true overlap count per scale = #regions holding >= 5% of probe mass;
per-anchor K_true = overlap count at the anchor's most significant scale.
Global step: adjusted level = component mean - beta_hat * x, with beta_hat the
median local slope estimated from smallest-scale probes (no oracle trend);
k-means(3) over adjusted levels; label accuracy up to permutation.
"""
import os
import sys
import numpy as np
import pandas as pd
from ext_core import bic_select_k, min_signal_ok, ALPHA
from dip import dip_pvalue

BETA = 0.15
T1, T2 = 1.0 / 3.0, 2.0 / 3.0
D1, D2 = 0.25, 0.50            # levels 0, -0.25, -0.75
TAU = 0.02
SCALES = [0.02, 0.09, 0.40]    # top scale spans both boundaries from mid-region
M = 1000
N_ANCHOR = 150
SEED = 20260731

LEVELS = np.array([0.0, -D1, -(D1 + D2)])


def region(x):  return (x >= T1).astype(int) + (x >= T2).astype(int)
def f_gated(x): return BETA * x + LEVELS[region(x)]


def run_block(bi):
    out = f"e2_b{bi}.csv"
    if os.path.exists(out):
        return False
    rng = np.random.default_rng(SEED + bi)
    anchors = np.array_split(np.linspace(0.02, 0.98, N_ANCHOR), 6)[bi]
    rows = []
    for a in anchors:
        # local slope estimate from smallest scale (for the global step)
        Xs = a + rng.normal(0, SCALES[0], M)
        ys = f_gated(Xs) + rng.normal(0, TAU, M)
        Z = np.column_stack([np.ones(M), Xs - a])
        beta_loc = float(np.linalg.lstsq(Z, ys, rcond=None)[0][1])

        best = dict(p=np.inf)
        for s in SCALES:
            X = a + rng.normal(0, s, M)
            reg = region(X)
            frac = np.bincount(reg, minlength=3) / M
            k_true = int((frac >= 0.05).sum())
            y = f_gated(X) + rng.normal(0, TAU, M)
            if not min_signal_ok(y, TAU):
                continue
            _, p = dip_pvalue(y)
            Xr = a + rng.normal(0, s, M)
            yr = f_gated(Xr) + rng.normal(0, TAU, M)
            k_hat, fit = bic_select_k(yr, rng, k_max=4)
            if p < best["p"]:
                best = dict(p=p, s=s, k_true=k_true, k_hat=k_hat,
                            mus=fit["mu"] if fit is not None else np.array([yr.mean()]),
                            frac=frac)
        if not np.isfinite(best["p"]):
            continue
        gaps = np.diff(np.sort(best["mus"]))
        rows.append(dict(
            anchor=a, region=int(region(np.array([a]))[0]),
            p_min=best["p"], sstar=best["s"],
            k_true=best["k_true"], k_hat=best["k_hat"],
            n_scales_signal=1,
            gap1=float(gaps[0]) if len(gaps) >= 1 else np.nan,
            gap2=float(gaps[1]) if len(gaps) >= 2 else np.nan,
            beta_loc=beta_loc,
            levels=";".join(f"{m - beta_loc * a:.4f}" for m in best["mus"])))
    pd.DataFrame(rows).to_csv(out, index=False)
    return True


MERGE_GAP = 0.15          # the registered effect-size floor doubles as the
                          # component-merge rule: gaps below it are not regimes


def merged_mus(mus, gap=MERGE_GAP):
    mus = np.sort(np.asarray(mus, float))
    out = [[mus[0]]]
    for m in mus[1:]:
        if m - out[-1][-1] < gap:
            out[-1].append(m)
        else:
            out.append([m])
    return np.array([np.mean(g) for g in out])


def do_summarize():
    df = pd.concat([pd.read_csv(f"e2_b{b}.csv") for b in range(6)])
    df.to_csv("e2_anchors.csv", index=False)

    # recover component means: stored level = mu - beta_loc * anchor
    mus_all = [np.array([float(v) for v in str(r.levels).split(";")])
               + r.beta_loc * r.anchor for _, r in df.iterrows()]
    df["k_merged"] = [len(merged_mus(m)) for m in mus_all]

    # (a) K_hat vs K_true confusion
    conf = pd.crosstab(df.k_true, df.k_hat)
    conf.to_csv("e2_confusion.csv")
    print("=== raw BIC K_hat vs true overlap count ===\n", conf.to_string())
    confm = pd.crosstab(df.k_true, df.k_merged)
    confm.to_csv("e2_confusion_merged.csv")
    print("\n=== K after the effect-size merge rule (gap >= 0.15) ===\n",
          confm.to_string())
    print(f"\nK accuracy: raw BIC {float((df.k_hat == df.k_true).mean()):.3f}"
          f"  -> merged {float((df.k_merged == df.k_true).mean()):.3f}")

    # (b) gap recovery
    k2 = df[(df.k_hat == 2) & (df.k_true == 2)]
    lo = k2[k2.region <= 1]   # boundary 1: true gap 0.25
    hi = k2[k2.region >= 1]   # boundary 2: true gap 0.50 (region 1 or 2 anchors)
    b1 = lo[abs(lo.anchor - T1) < abs(lo.anchor - T2)]
    b2 = hi[abs(hi.anchor - T2) < abs(hi.anchor - T1)]
    print(f"\ngap at boundary 1 (true 0.25): median {b1.gap1.median():.3f} (n={len(b1)})")
    print(f"gap at boundary 2 (true 0.50): median {b2.gap1.median():.3f} (n={len(b2)})")
    g1, g2 = [], []
    for (_, r), mus in zip(df.iterrows(), mus_all):
        mm = merged_mus(mus)
        if len(mm) == 3:
            g1.append(mm[1] - mm[0]); g2.append(mm[2] - mm[1])
    if g1:
        print(f"K=3 anchors, MERGED means: gaps median ({np.median(g1):.3f}, "
              f"{np.median(g2):.3f})  true (0.25, 0.50)  n={len(g1)}")

    # (c) global matching by trend-adjusted levels. Per-anchor slopes near a
    # boundary absorb the step (the E1 lesson), so use the GLOBAL median slope
    # -- robust because most anchors are interior -- not each anchor's own.
    beta_hat = float(df.beta_loc.median())
    lv, tr = [], []
    for (_, r), mus in zip(df.iterrows(), mus_all):
        for m in merged_mus(mus):
            lam = float(m - beta_hat * r.anchor)
            lv.append(lam)
            tr.append(int(np.argmin(np.abs(LEVELS - lam))))
    lv = np.asarray(lv); tr = np.asarray(tr)
    c = np.quantile(lv, [0.15, 0.5, 0.85])
    for _ in range(100):
        z = np.argmin(np.abs(lv[:, None] - c[None, :]), axis=1)
        c = np.array([lv[z == k].mean() if (z == k).any() else c[k] for k in range(3)])
    order = np.argsort(c)[::-1]      # descending -> matches LEVELS order
    zz = np.empty_like(z); [zz.__setitem__(z == order[k], k) for k in range(3)]
    c_sorted = c[order]
    lab_acc = float((zz == tr).mean())
    print(f"\nglobal level centers, shifted to top=0 "
          f"(true 0 / -0.25 / -0.75): {np.round(c_sorted - c_sorted[0], 3)}")
    print(f"component -> global-regime label accuracy: {lab_acc:.3f} "
          f"(n components = {len(lv)})")
    print(f"beta_hat (median local slope, true {BETA}): {beta_hat:.3f}")
    pd.DataFrame(dict(center=c_sorted - c_sorted[0])).to_csv("e2_levels.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        do_summarize()
    else:
        import time
        t0 = time.time()
        for b in range(6):
            if run_block(b):
                print(f"e2 block {b} ({time.time()-t0:.0f}s)", flush=True)
                if time.time() - t0 > 25:
                    print("CHUNK LIMIT"); sys.exit(0)
        print("E2 ALL DONE")
