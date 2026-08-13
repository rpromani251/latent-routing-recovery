"""Figure: the distance-to-boundary estimator d_hat = -sigma * Phi^-1(pi_hat)."""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
mpl.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

df = pd.read_csv("distance_estimator_rows.csv")
df["fires"] = df.dip_p < 0.05 / 3
df["inband"] = (df.pi_cross >= 0.05) & (df.pi_cross <= 0.5) & df.d_hat.notna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))

# ---- Panel A: recovery vs truth, by probe scale -------------------------------
g = df[(df.model == "gated") & (df.tau_obs == 0.02) & df.fires & df.inband]
lim = (0.004, 0.45)
ax1.plot(lim, lim, ls=(0, (4, 3)), lw=1.0, color=MUTED, zorder=1)
ax1.annotate("perfect recovery", xy=(0.30, 0.30), xytext=(0.105, 0.345),
             color=INK2, fontsize=8.2, rotation=0)
for (s, col, mk) in zip(sorted(g.sigma.unique()), [S1, S2, S3], ["o", "s", "^"]):
    sub = g[g.sigma == s]
    ax1.scatter(sub.d_true, sub.d_hat, s=26, marker=mk, facecolor=col,
                edgecolor="#fcfcfb", linewidth=0.9, alpha=0.9, zorder=3,
                label=f"σ = {s:.3f}   (n={len(sub)})")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlim(*lim); ax1.set_ylim(*lim)
ax1.set_xlabel("true distance to boundary  $d$")
ax1.set_ylabel(r"estimated  $\hat d = -\sigma\,\Phi^{-1}(\hat\pi)$")
ax1.set_title("Distance recovers from the mixing fraction", loc="left", pad=10)
ax1.grid(True, which="major", color=MUTED, lw=0.4, alpha=0.45, zorder=0)
ax1.legend(loc="lower right", fontsize=8.2, handletextpad=0.4, labelspacing=0.35)
ax1.text(0.0055, 0.30, "r = 0.996\nmedian rel. error  −0.9%",
         fontsize=8.6, color=INK2, va="top")

# ---- Panel B: across-rung constancy, gate vs smooth confounder ----------------
rows = []
for (m, tau, a), sub in df[df.inband].groupby(["model", "tau_obs", "anchor"]):
    if len(sub) < 2 or not sub.fires.any():
        continue
    v = sub.d_hat.values
    rows.append(dict(model=m, cv=float(np.std(v) / (np.mean(v) + 1e-12))))
cv = pd.DataFrame(rows)
grp = {"gated": ("true gate", S1), "confounder": ("smooth / kink confounder", S2)}
cv["cls"] = np.where(cv.model == "gated", "gated", "confounder")

for cls, (lab, col) in grp.items():
    v = np.sort(cv[cv.cls == cls].cv.values)
    y = np.arange(1, len(v) + 1) / len(v)
    ax2.step(np.concatenate([[0], v]), np.concatenate([[0], y]), where="post",
             lw=2.0, color=col, label=f"{lab}   (n={len(v)})", zorder=3)
ax2.axvline(0.15, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
ax2.text(0.175, 0.965, "CV = 0.15", color=INK2, fontsize=8.2, va="top")
ax2.scatter([0.15, 0.15], [0.557, 0.071], s=34, facecolor="#fcfcfb",
            edgecolor=[S1, S2], linewidth=1.8, zorder=4)
ax2.text(0.175, 0.545, "56% of gates retained", fontsize=8.4, color=INK2)
ax2.text(0.175, 0.115, "93% of confounders rejected", fontsize=8.4, color=INK2)
ax2.set_xlim(0, 1.4); ax2.set_ylim(0, 1.02)
ax2.set_xlabel(r"across-rung coefficient of variation of  $\hat d$")
ax2.set_ylabel("fraction of anchors ≤ CV")
ax2.set_title("A real boundary sits at one distance; curvature does not",
              loc="left", pad=10)
ax2.grid(True, color=MUTED, lw=0.4, alpha=0.45, zorder=0)
ax2.legend(loc="lower right", fontsize=8.4, handletextpad=0.6)

fig.text(0.007, 0.005,
         "1-D known-regimes setting, Δ=0.30, ladder geomspace(0.02, 0.2, 3), "
         "m=1000/rung. Guards: dip fires (Bonferroni) and π̂ ∈ [0.05, 0.50]. "
         "Confounders: GP nulls with ℓ < ladder top, plus the kink control.",
         fontsize=7.6, color=INK2)
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig("fig_distance_estimator.png", dpi=190)
print("wrote fig_distance_estimator.png")
