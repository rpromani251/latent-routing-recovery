"""Four-panel figure for the future-work extension experiments E1-E4."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"iso_total": "#8C8C8C", "iso_coord": "#1f77b4", "iso_resid": "#C44E52",
     "iso_resid_rob": "#2E7D32", "coord_scan": "#9467bd"}
L = {"iso_total": "isotropic, fixed total budget", "iso_coord": "isotropic, fixed per-coord",
     "iso_resid": "+ OLS residualization", "iso_resid_rob": "+ trimmed residualization",
     "coord_scan": "coordinate scan (D$\\times$ cost)"}

fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.7))

# (a) E1: power vs D
e1 = pd.read_csv("e1_summary.csv")
a = ax[0]
for s in ["iso_total", "iso_coord", "iso_resid", "iso_resid_rob", "coord_scan"]:
    d = e1[e1.strategy == s]
    a.plot(d.D, d.power, "o-", color=C[s], label=L[s], ms=4)
a.set_xscale("log", base=2); a.set_xticks([2, 4, 8, 16]); a.set_xticklabels([2, 4, 8, 16])
a.set_xlabel("input dimension D"); a.set_ylabel("power (detectable anchors)")
a.set_title("E1: full-vector probing", fontsize=10)
a.legend(fontsize=6, frameon=False); a.set_ylim(-0.03, 1.08)

# (b) E2: K selection accuracy raw vs merged; gap recovery inset text
a = ax[1]
a.bar([0, 1], [0.640, 0.747], color=["#8C8C8C", "#2E7D32"], width=0.55)
a.set_xticks([0, 1]); a.set_xticklabels(["raw BIC", "+ effect-size\nmerge rule"], fontsize=8)
a.set_ylabel("$\\hat K$ accuracy"); a.set_ylim(0, 1.05)
a.axhline(1.0, ls=":", c="k", lw=0.6)
a.set_title("E2: $K>2$ regimes", fontsize=10)
a.text(0.5, 0.30, "gaps at boundary scales:\n0.228 / 0.477\n(true 0.25 / 0.50)\n\n"
       "global label acc. 0.997", ha="center", fontsize=7,
       transform=a.transAxes)

# (c) E3: power vs theta
e3 = pd.read_csv("e3_summary.csv")
a = ax[2]
cs = {"pc1": "#C44E52", "pc2_bonf": "#2E7D32", "axes_bonf": "#1f77b4",
      "rand8_bonf": "#9467bd"}
for s, lab in [("pc1", "PC1 only"), ("pc2_bonf", "top-2 PCs (Bonf.)"),
               ("axes_bonf", "output axes (Bonf.)"), ("rand8_bonf", "8 random proj.")]:
    d = e3[(e3.V == 3) & (e3.strategy == s)]
    a.plot(d.theta, d.power, "o-", color=cs[s], label=lab, ms=4)
a.set_xticks([0, 30, 60, 90])
a.set_xlabel(r"angle $\theta$(penalty, within-branch variation)")
a.set_ylabel("power"); a.set_ylim(-0.03, 1.08)
a.set_title("E3: vector outputs (V=3)", fontsize=10)
a.legend(fontsize=6, frameon=False)
a.annotate("parallel penalty:\nmasked for all", xy=(2, 0.06), fontsize=7, color="#666")

# (d) E4: FP and power by manifold x method
e4 = pd.read_csv("e4_summary.csv")
a = ax[3]
order = [("clumpy", "raw"), ("clumpy", "resid"), ("clumpy", "resid_rob"),
         ("uniform", "raw"), ("uniform", "resid_rob")]
x = np.arange(len(order)); w = 0.38
fp = [float(e4[(e4.manifold == m) & (e4.method == me)].fp_honest.iloc[0]) for m, me in order]
pw = [float(e4[(e4.manifold == m) & (e4.method == me)].power_detectable.iloc[0]) for m, me in order]
a.bar(x - w / 2, fp, w, color="#C44E52", label="FP (honest)")
a.bar(x + w / 2, pw, w, color="#2E7D32", label="power (detectable)")
a.axhline(0.05, ls="--", c="k", lw=0.7)
a.set_xticks(x)
a.set_xticklabels(["clumpy\nraw", "clumpy\nOLS res.", "clumpy\ntrimmed",
                   "uniform\nraw", "uniform\ntrimmed"], fontsize=7)
a.set_ylabel("rate"); a.set_title("E4: clumpy manifold (A12)", fontsize=10)
a.legend(fontsize=7, frameon=False)

fig.tight_layout()
fig.savefig("fig_future_work.png", dpi=180)
print("wrote fig_future_work.png")
