"""Figure: Experiment P — estimator ranking, recovery, and the placement effect."""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
mpl.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 8.8, "axes.labelsize": 9.2, "axes.titlesize": 10.2,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})
df = pd.read_csv("p_pooling_final.csv")
est = df[df.estimator.isin(list("ABC"))]
Ns = [5, 10, 25, 50, 100]

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.6, 4.3))

# --- A: estimator ranking at the discriminating cell -------------------------
h = est[(est.null == "honest") & (est.placement == "design") & (est.pi_target <= 0.10)
        & (est.dt_ratio == 1.5)]
lbl = {"A": "A  statistic only", "B": "B  threshold, then geometry",
       "C": "C  weighted geometry, no threshold"}
for (e, sub), col, mk in zip(h.groupby("estimator"), S, ["o", "s", "^"]):
    g = sub.groupby("N").power.mean()
    a1.plot(g.index, g.values, "-", color=col, lw=2.0, marker=mk, ms=6.5,
            mec="#fcfcfb", mew=1.0, zorder=3, label=lbl[e])
a1.axhline(0.05, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
a1.text(5.3, 0.075, "nominal", color=INK2, fontsize=8)
a1.set_xscale("log"); a1.set_xticks(Ns); a1.set_xticklabels(map(str, Ns))
a1.set_ylim(0, 1.03); a1.set_xlabel("anchors pooled,  $N$")
a1.set_ylabel("power vs the honest (no-gate) null")
a1.set_title("Δ/τ = 1.5:  geometry wins, thresholding costs", loc="left", pad=9)
a1.grid(True, color=MUTED, lw=0.4, alpha=0.45, zorder=0)
a1.legend(loc="upper left", fontsize=8.1, handletextpad=0.6, labelspacing=0.3)

# --- B: recovery -------------------------------------------------------------
pt = df[(df.estimator == "point") & (df.placement == "design") & (df.pi_target <= 0.10)]
for (dt, sub), col, mk in zip(pt.groupby("dt_ratio"), S, ["o", "s", "^", "D"]):
    g = sub.groupby("N").orient_err.mean()
    a2.plot(g.index, g.values, "-", color=col, lw=2.0, marker=mk, ms=6.5,
            mec="#fcfcfb", mew=1.0, zorder=3, label=f"Δ/τ = {dt}")
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xticks(Ns); a2.set_xticklabels(map(str, Ns))
a2.set_xlabel("anchors pooled,  $N$")
a2.set_ylabel("orientation error of $\\hat n$  (degrees)")
a2.set_title("Recovery converges — except at Δ/τ = 1", loc="left", pad=9)
a2.grid(True, which="major", color=MUTED, lw=0.4, alpha=0.45, zorder=0)
a2.legend(loc="lower left", fontsize=8.1, handletextpad=0.6, labelspacing=0.3)
a2.annotate("no-information floor", xy=(25, 38.9), xytext=(6.5, 15),
            fontsize=8.1, color=INK2,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

# --- C: placement ------------------------------------------------------------
r = est[(est.null == "honest") & (est.pi_target == 0.10) & (est.dt_ratio == 1.5)]
for (pl, sub), col, mk, ls in zip(r.groupby("placement"), [S[2], S[1]], ["^", "v"],
                                  ["-", (0, (5, 2))]):
    for e, style in (("C", 2.2), ("A", 1.3)):
        g = sub[sub.estimator == e].groupby("N").power.mean()
        a3.plot(g.index, g.values, color=col, linestyle=ls, lw=style,
                marker=(mk if e == "C" else "None"), ms=6.5, mec="#fcfcfb", mew=1.0,
                alpha=1.0 if e == "C" else 0.55, zorder=3,
                label=f"{'by-design' if pl=='design' else 'random'} — {e}")
a3.set_xscale("log"); a3.set_xticks(Ns); a3.set_xticklabels(map(str, Ns))
a3.set_ylim(0, 1.03); a3.set_xlabel("anchors pooled,  $N$")
a3.set_ylabel("power vs the honest null")
a3.set_title("Geometry needs placement; the statistic doesn't", loc="left", pad=9)
a3.grid(True, color=MUTED, lw=0.4, alpha=0.45, zorder=0)
a3.legend(loc="upper left", fontsize=8.0, handletextpad=0.6, labelspacing=0.3)

fig.text(0.006, 0.005,
         "Intrinsic d = 2, ambient D = 20, frame supplied. 250 anchors per cell, 600 probes per anchor, "
         "300 subsamples per (cell, N), α = 0.05.  Panels A and C use the honest (no-gate) null;\n"
         "the permutation null cannot test existence, because it preserves the residual multiset and the LRT "
         "statistic is invariant to it.  Panels A–B pool π ≤ 0.10; panel C is π = 0.10, Δ/τ = 1.5.",
         fontsize=7.4, color=INK2)
fig.tight_layout(rect=[0, 0.075, 1, 1])
fig.savefig("fig_p_pooling.png", dpi=190)
print("wrote fig_p_pooling.png")
