"""Figure: Experiment A — the 3-rung rule repairs what pooling cannot, and vice versa."""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

S = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
mpl.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 8.8, "axes.labelsize": 9.2, "axes.titlesize": 10.2,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})
df = pd.read_csv("a_invariance_rows.csv")
d = df[(df.rule != "alpha") & (df.detector == "dip|LRT") & (df.pi_top == 0.10)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

# --- A: FP repair and power cost, by rule ------------------------------------
surf = ["honest", "resonant_0.5L", "resonant_1.0L", "resonant_1.5L", "gated"]
lab = ["honest", "resonant\n0.5 L", "resonant\n1.0 L", "resonant\n1.5 L", "GATED\n(power)"]
x = np.arange(len(surf)); w = 0.26
for k, (rule, col) in enumerate(zip(["naive", "2-rung", "3-rung"], S)):
    v = [float(d[(d.surface == s) & (d.rule == rule)].fire.iloc[0]) for s in surf]
    a1.bar(x + (k - 1) * (w + 0.012), v, w, color=col, zorder=3,
           label={"naive": "naive (top rung only)", "2-rung": "2-rung",
                  "3-rung": "3-rung, log-log α"}[rule])
a1.axhline(0.05, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
a1.text(-0.45, 0.075, "nominal", color=INK2, fontsize=8)
a1.set_xticks(x); a1.set_xticklabels(lab, fontsize=8.2)
a1.set_ylim(0, 1.12); a1.set_ylabel("fire rate")
a1.set_title("The 3-rung rule repairs long-wavelength resonance", loc="left", pad=9)
a1.grid(True, axis="y", color=MUTED, lw=0.4, alpha=0.45, zorder=0)
a1.legend(loc="upper right", fontsize=8.2, handletextpad=0.6)
a1.annotate("not repaired", xy=(1.26, 0.74), xytext=(1.05, 0.42),
            fontsize=8.2, color=INK2,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

# --- B: the complementarity ---------------------------------------------------
L = 2 * 0.20 * np.sqrt(2)
per_x = [0.5, 1.0, 1.5]
per_y = [float(d[(d.surface == f"resonant_{e}L") & (d.rule == "3-rung")].fire.iloc[0])
         for e in ("0.5", "1.0", "1.5")]
pool = pd.read_csv("/tmp/pexp/p_comb_rows.csv")
pool_x = [0.3 / L, 0.5 / L, 0.8 / L]
pool_y = [float(pool[pool.surface == f"resonant ell={e}"].C_fires.iloc[0])
          for e in ("0.3", "0.5", "0.8")]
a2.plot(per_x, per_y, "-o", color=S[2], lw=2.2, ms=7, mec="#fcfcfb", mew=1.0,
        zorder=3, label="per-anchor 3-rung α rule")
a2.plot(pool_x, pool_y, "-s", color=S[1], lw=2.2, ms=7, mec="#fcfcfb", mew=1.0,
        zorder=3, label="geometric pooling (estimator C)")
a2.axhline(0.05, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
a2.axvline(1.0, color=MUTED, ls=":", lw=1.0, zorder=1)
a2.set_xlim(0.4, 1.6); a2.set_ylim(-0.03, 1.06)
a2.set_xlabel("resonance wavelength  ℓ,  in probe diameters")
a2.set_ylabel("false-positive rate on a boundary-free surface")
a2.set_title("Each covers the regime the other fails in", loc="left", pad=9)
a2.grid(True, color=MUTED, lw=0.4, alpha=0.45, zorder=0)
a2.legend(loc="center left", fontsize=8.4, handletextpad=0.6)
a2.text(0.44, 0.60, "short ℓ:\npooling catches it,\nthe ladder does not",
        fontsize=8.2, color=INK2)
a2.text(1.06, 0.60, "long ℓ:\nthe ladder catches it,\npooling does not",
        fontsize=8.2, color=INK2)

fig.text(0.006, 0.006,
         "Left: π_top = 0.10, detector dip∨LRT, 220 anchors per surface. The dip contributes nothing at this "
         "separation (fire rate 0.000 on gated), so dip∨LRT ≡ LRT.\n"
         "LRT thresholds recalibrated per rung on trimmed residuals from honest-smooth pipeline anchors "
         "(5.50 / 5.35 / 4.60; null medians 0.68 / 0.68 / 0.61) — the 6.08 figure from clean Gaussian draws is not reused.",
         fontsize=7.4, color=INK2)
fig.tight_layout(rect=[0, 0.075, 1, 1])
fig.savefig("fig_a_invariance.png", dpi=190)
print("wrote fig_a_invariance.png")
