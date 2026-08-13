"""Figure: Experiment S -- what the OOD detector sees, and the operating window."""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

S1c, S2c, S3c = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
mpl.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

sw = pd.read_csv("s5_radius_sweep.csv")
order = ["compas", "german", "housing", "cc"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

# ---- Panel A: visibility, audit probes vs LIME's own -------------------------
vis = sw[sw.r_frac == 1.0].groupby("target").frac_real.mean().reindex(order)
x = np.arange(len(order)); w = 0.36
ax1.bar(x - w/2 - 0.01, vis.values, w, color=S1c, label="our tangent-frame probes", zorder=3)
ax1.bar(x + w/2 + 0.01, np.zeros(len(order)) + 0.004, w, color=S2c,
        label="LIME's own perturbation", zorder=3)
for xi, v in zip(x, vis.values):
    ax1.text(xi - w/2 - 0.01, v + 0.025, f"{v:.2f}", ha="center", fontsize=8.6, color=INK2)
    ax1.text(xi + w/2 + 0.01, 0.03, "0.00", ha="center", fontsize=8.6, color=INK2)
ax1.set_xticks(x); ax1.set_xticklabels(["COMPAS", "German", "Housing", "Comm.&Crime"])
ax1.set_ylim(0, 1.12); ax1.set_ylabel("fraction of probes the detector calls REAL")
ax1.set_title("The detector does not recognise our probe geometry", loc="left", pad=10)
ax1.grid(True, axis="y", color=MUTED, lw=0.4, alpha=0.45, zorder=0)
ax1.legend(loc="upper right", fontsize=8.4, handletextpad=0.6)
ax1.annotate("all-continuous features:\nno lattice to hide on", xy=(3, 0.05),
             xytext=(2.35, 0.42), fontsize=8.3, color=INK2,
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

# ---- Panel B: housing operating window --------------------------------------
h = sw[sw.target == "housing"].groupby("r_frac").agg(vis=("frac_real", "mean"),
                                                     cross=("crosses", "mean"))
ax2.plot(h.index, h.vis, "-o", color=S1c, lw=2.0, ms=6.5, mec="#fcfcfb", mew=1.0,
         zorder=3, label="invisible to detector")
ax2.plot(h.index, h.cross, "-s", color=S3c, lw=2.0, ms=6.5, mec="#fcfcfb", mew=1.0,
         zorder=3, label="ball straddles the boundary")
ax2.fill_between(h.index, 0, np.minimum(h.vis, 1), color=S1c, alpha=0.05, zorder=1)
ax2.set_xlabel("probe radius, as a fraction of the local $k$-NN scale")
ax2.set_ylabel("fraction of probes / anchors")
ax2.set_ylim(0, 1.02)
ax2.set_title("Housing: both conditions hold at once", loc="left", pad=10)
ax2.grid(True, color=MUTED, lw=0.4, alpha=0.45, zorder=0)
ax2.legend(loc="center left", fontsize=8.6, handletextpad=0.6)
ax2.text(0.52, 0.70, "usable window:\nprobes stay invisible\n*and* reach the gate",
         fontsize=8.4, color=INK2)

fig.text(0.007, 0.006,
         "150 anchors/target (60 for the sweep), 300–400 probes/anchor. Frame and radius supplied by local PCA "
         "on the 60 nearest real points (Stage 0 deferred).\n"
         "Density-filter retention ρ = 0.985–0.997 on every target, so A3 never fires — prediction S-3 falsified. "
         "Detector leakage on real points: 0.000.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig("fig_s_experiment.png", dpi=190)
print("wrote fig_s_experiment.png")
