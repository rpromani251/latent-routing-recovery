"""Figure: does the separability classifier recover the boundary normal?"""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical 1-4
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
mpl.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

df = pd.read_csv("normal_estimator_rows.csv")
rng = np.random.default_rng(11)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

# ---- Panel A: per-anchor information, sd of signed rotation vs pi -------------
ax1.axhline(52.0, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
ax1.text(0.055, 53.5, "uninformative  (uniform on ±90°)", color=INK2, fontsize=8.3,
         ha="left", va="bottom")
for (dt, sub), col, mk in zip(df.groupby("dt_ratio"), S, ["o", "s", "^", "D"]):
    g = sub.groupby("pi_target").signed_soft.std()
    ax1.plot(g.index, g.values, "-", color=col, lw=2.0, marker=mk, ms=6.5,
             mec="#fcfcfb", mew=1.0, zorder=3, label=f"Δ/τ = {dt}")
ax1.set_xlabel(r"mixing fraction  $\pi$   (share of probes crossing)")
ax1.set_ylabel("sd of per-anchor rotation  (degrees)")
ax1.set_title("Per-anchor signal lives at LOW $\\pi$", loc="left", pad=10)
ax1.set_ylim(0, 60); ax1.set_xlim(0.02, 0.53)
ax1.grid(True, color=MUTED, lw=0.4, alpha=0.45, zorder=0)
ax1.legend(loc="lower right", fontsize=8.4, handletextpad=0.6, labelspacing=0.3)

# ---- Panel B: pooled orientation error vs N ----------------------------------
def pooled(theta, N, B=800):
    t = np.radians(theta.dropna().values) * 2.0
    if len(t) < 2 * N: return np.nan
    o = [abs(np.degrees(np.arctan2(np.sin(s).mean(), np.cos(s).mean())) / 2.0)
         for s in (rng.choice(t, N, replace=False) for _ in range(B))]
    return float(np.median(o))

Ns = [1, 5, 10, 25, 50, 100]
cells = [(5.0, 0.05, S[3], "D"), (2.5, 0.05, S[2], "^"),
         (1.5, 0.05, S[1], "s"), (1.0, 0.05, S[0], "o")]
for dt, pi, col, mk in cells:
    s = df[(df.dt_ratio == dt) & (df.pi_target == pi)]
    ys = [pooled(s.signed_soft, n) for n in Ns]
    ax2.plot(Ns, ys, "-", color=col, lw=2.0, marker=mk, ms=6.5,
             mec="#fcfcfb", mew=1.0, zorder=3, label=f"Δ/τ = {dt}")
ax2.axhline(5.0, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
ax2.text(100, 5.7, "5° — enough for axis-dominance", color=INK2, fontsize=8.3, ha="right")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xticks(Ns); ax2.set_xticklabels([str(n) for n in Ns])
ax2.set_xlabel("anchors pooled,  $N$")
ax2.set_ylabel("pooled orientation error  (degrees)")
ax2.set_title("Variance pools away; there is no bias to survive", loc="left", pad=10)
ax2.grid(True, which="major", color=MUTED, lw=0.4, alpha=0.45, zorder=0)
ax2.legend(loc="lower left", fontsize=8.4, handletextpad=0.6, labelspacing=0.3)
ax2.text(24, 33, r"$\pi = 0.05$", color=INK2, fontsize=9)

fig.text(0.007, 0.005,
         "Intrinsic d = 2, ambient D = 20, frame supplied. m = 1000 probes/anchor, 200 anchors/cell. "
         "Normal from an LDA discriminant on soft responsibilities, fitted on all probes.\n"
         "Per-cell bias is not significant at 95% in any of the 20 cells (soft arm |bias| ≤ 6.6°); "
         "the oracle arm's bias is ≤ 0.35° throughout. Right panel is the π = 0.05 row.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=[0, 0.075, 1, 1])
fig.savefig("fig_normal_estimator.png", dpi=190)
print("wrote fig_normal_estimator.png")
