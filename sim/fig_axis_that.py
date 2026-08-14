"""
Figure for Experiment T -- axis dominance and t_hat.

  Left    the three dominance rules against the gate's tilt off a coordinate axis.
          Above 0 degrees every fire is a WRONG claim, not a weak one.
  Centre  interval coverage on t_hat: the offset as specified against the repair.
  Right   where the bias comes from -- it scales with the threshold's own value, which
          is why Experiment P could not see it at c_true = 0.

Palette matches the method notes (boundary_recovery_v5.tex).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

NAVY, MUTED, OKGREEN = "#19375C", "#5A6069", "#006E3C"
SPECBLUE, NEWMAG, AMBER, LIMRED, LG = "#2A78D6", "#A02D78", "#AA6E00", "#AF2D2D", "#AFB1AC"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": NAVY, "axes.titlecolor": NAVY,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.titlesize": 9.5, "axes.titleweight": "bold",
})


def main():
    ax_df = pd.read_csv(os.path.join(HERE, "axis_dominance_rows.csv"))
    tc = pd.read_csv(os.path.join(HERE, "that_coverage_rows.csv"))
    ob = pd.read_csv(os.path.join(HERE, "offset_bias_rows.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.15))

    # ------------------------------------------------------------- panel A
    ax = axes[0]
    s = ax_df[(ax_df.N == 100) & (ax_df.dt_ratio == 2.5)]
    g = s.groupby("theta_deg").mean(numeric_only=True).reset_index().sort_values("theta_deg")
    ax.axvspan(-2, 0.9, color=OKGREEN, alpha=0.10, lw=0)
    ax.text(1.5, 1.20, "single-coordinate\ntruth exists", fontsize=7.0, color=OKGREEN,
            ha="left", va="top")
    ax.text(93, 1.20, "no single-coordinate truth here —\nevery fire is a wrong claim",
            fontsize=7.2, color=LIMRED, ha="right", va="top")
    ax.plot(g.theta_deg, g.fire_boot, "-o", color=LIMRED, ms=5.5, lw=1.6,
            label="bootstrap stability (my repair)", mec="white", mew=0.7)
    ax.plot(g.theta_deg, g.fire_perm, "-s", color=AMBER, ms=5.5, lw=1.6,
            label="permutation (Stage 7 as written)", mec="white", mew=0.7)
    if "fire_equiv_10" in g.columns:
        ax.plot(g.theta_deg, g.fire_equiv_10, "-D", color=OKGREEN, ms=5.5, lw=1.8,
                label="equivalence, 10° tolerance", mec="white", mew=0.7)
    ax.axhline(0.05, color=MUTED, lw=0.8, ls="--")
    ax.text(62, 0.085, "0.05", fontsize=7.2, color=MUTED)
    ax.set_xlabel("tilt of the true gate off a coordinate axis  (degrees)")
    ax.set_ylabel("rate of claiming “routes on feature i”")
    ax.set_title("A.  two rules fail in opposite directions", loc="left")
    ax.set_ylim(-0.05, 1.24)
    ax.set_xlim(-3, 96)
    ax.legend(fontsize=7.2, frameon=False, loc="center left", bbox_to_anchor=(0.06, 0.42))

    # ------------------------------------------------------------- panel B
    ax = axes[1]
    g = tc.groupby(["dt", "N", "mode"]).mean(numeric_only=True).reset_index()
    style = {1.5: (LG, "o"), 2.5: (NEWMAG, "s"), 5.0: (NAVY, "^")}
    for dt, (col, mk) in style.items():
        a = g[(g.dt == dt) & (g["mode"] == "A")].sort_values("N")
        c = g[(g.dt == dt) & (g["mode"] == "C")].sort_values("N")
        ax.plot(a.N, a.coverage, "--" + mk, color=col, ms=6, lw=1.5, mec="white",
                mew=0.7, label=fr"$\Delta/\tau$ = {dt}")
        ax.plot(c.N, c.coverage, "-" + mk, color=col, ms=6, lw=2.0, mec="white", mew=0.7)
    ax.axhline(0.95, color=OKGREEN, lw=1.0, ls=":")
    ax.text(24, 0.885, "nominal 0.95", fontsize=7.4, color=OKGREEN)
    ax.text(38, 0.68, "repaired offset\n(solid)", fontsize=7.6, color=OKGREEN, ha="center")
    ax.text(150, 0.30, "as specified\n(dashed)", fontsize=7.6, color=LIMRED, ha="center")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([25, 50, 100, 200])
    ax.set_xticklabels(["25", "50", "100", "200"])
    ax.set_xlabel("anchors pooled")
    ax.set_ylabel(r"coverage of the 95% interval on $\hat{t}$")
    ax.set_ylim(-0.05, 1.06)
    ax.set_title(r"B.  the interval on $\hat{t}$ covers at 0.00, not 0.95", loc="left")
    ax.legend(fontsize=7.2, frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.38))

    # ------------------------------------------------------------- panel C
    ax = axes[2]
    for dt, (col, mk) in style.items():
        s = ob[ob.dt == dt].sort_values("T_true")
        ax.plot(s.T_true, np.abs(s.bias_A), "-" + mk, color=col, ms=6, lw=1.6,
                mec="white", mew=0.7, label=fr"$\Delta/\tau$ = {dt}")
        ax.plot(s.T_true, np.abs(s.bias_C), ":" + mk, color=col, ms=4, lw=1.2,
                alpha=0.75, mec="white", mew=0.5)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("true threshold  $T$  (the value being recovered)")
    ax.set_ylabel(r"|bias| in $\hat{c}$")
    ax.set_title(r"C.  the bias scales with $T$ — invisible at $T=0$", loc="left")
    ax.text(11.5, 1.6, "as specified (solid):\nbias $\\propto T$", fontsize=7.4,
            color=LIMRED, ha="center")
    ax.text(12.5, 0.0022, "repaired (dotted)", fontsize=7.4, color=OKGREEN, ha="center")
    ax.axvline(0.0, color=MUTED, lw=0.8, ls="--")
    ax.text(0.5, 3.0, "where Experiment P\nmeasured it", fontsize=7.2, color=MUTED)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right")

    fig.tight_layout(w_pad=2.2)
    out = os.path.join(HERE, "fig_axis_that.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
