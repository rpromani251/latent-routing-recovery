"""
Figure for Experiment B -- the full-pipeline bootstrap.

  Left    the three null-generating arms against the pipeline's own null. The
          registered recipe (resample inlier residuals) sits two orders of magnitude
          to the right of where a null belongs.
  Centre  detection power against d/sigma, with the orientation-useful shell from
          step 2 marked, and the minimum-mass rule's pass rate on the same axis.
  Right   the dip's floor as a function of the mixing fraction.

Palette matches the method notes (boundary_recovery_v5.tex).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

NAVY = "#19375C"
MUTED = "#5A6069"
OKGREEN = "#006E3C"
SPECBLUE = "#2A78D6"
NEWMAG = "#A02D78"
AMBER = "#AA6E00"
LIMRED = "#AF2D2D"
LG = "#AFB1AC"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": NAVY, "axes.titlecolor": NAVY,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.titlesize": 9.5, "axes.titleweight": "bold",
})


def main():
    nulls = pd.read_csv(os.path.join(HERE, "bootstrap_nulls.csv"))
    stats = pd.read_csv(os.path.join(HERE, "bootstrap_anchor_stats.csv"))
    ref = np.load(os.path.join(HERE, "bootstrap_null_reference.npz"))
    floor = pd.read_csv(os.path.join(HERE, "dip_floor_rows.csv"))
    thr = float(np.quantile(ref["lrt_pipeline"], 0.95))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))

    # ---------------------------------------------------------------- panel 1
    ax = axes[0]
    rung = nulls[nulls.condition.str.startswith("rung")]
    arms = [("param", OKGREEN, "param (Gaussian)"),
            ("emp_all", AMBER, "resample all residuals"),
            ("emp_inlier", LIMRED, "resample inlier residuals")]
    ypos = {a: i for i, (a, _, _) in enumerate(arms)}
    for arm, col, lab in arms:
        s = rung[rung.arm == arm]
        y = ypos[arm]
        ax.hlines(y, s.lrt_med.min(), s.lrt_q95.max(), color=col, lw=1.2, alpha=0.45)
        ax.scatter(s.lrt_med, [y] * len(s), s=30, color=col, marker="o",
                   zorder=3, label=None)
        ax.scatter(s.lrt_q95, [y] * len(s), s=42, color=col, marker="|", lw=2, zorder=3)
    cg = nulls[nulls.condition == "clean_gauss_untrimmed"].iloc[0]
    ax.scatter([cg.lrt_med], [-0.8], s=30, color=NAVY, marker="o")
    ax.scatter([cg.lrt_q95], [-0.8], s=42, color=NAVY, marker="|", lw=2)
    ax.hlines(-0.8, cg.lrt_med, cg.lrt_q95, color=NAVY, lw=1.2, alpha=0.45)
    ax.axvline(thr, color=NAVY, ls=":", lw=1.1)
    ax.text(thr * 1.15, 2.42, f"calibrated\nthreshold {thr:.2f}", fontsize=7.5,
            color=NAVY, va="top")
    ax.set_xscale("log")
    ax.set_yticks([-0.8, 0, 1, 2])
    ax.set_yticklabels(["clean Gaussian\n(untrimmed)", "param\n(Gaussian)",
                        "resample ALL\nresiduals", "resample INLIER\nresiduals"],
                       fontsize=7.6)
    ax.set_ylim(-1.5, 2.6)
    ax.set_xlabel("null LRT   (circle = median, bar = q95; one pair per rung x replicate)")
    ax.set_title("A.  the registered recipe does not produce a null", loc="left")
    ax.text(0.98, 0.04, "v5's recipe", transform=ax.transAxes, ha="right",
            fontsize=7.5, color=LIMRED, style="italic")

    # ---------------------------------------------------------------- panel 2
    ax = axes[1]
    g = stats[stats.cell.str.startswith("gated")].copy()
    g["fire"] = g.lrt > thr
    agg = (g.groupby(["cell", "rung"])
             .agg(dos=("d_over_sigma", "median"), power=("fire", "mean"),
                  mm=("minmass_pass", "mean"), n=("fire", "size"))
             .reset_index().sort_values("dos"))
    # the honest surface is the d/sigma -> infinity limit of the same question
    h = stats[stats.cell == "honest"]
    ax.axvspan(1.28, 1.64, color=SPECBLUE, alpha=0.12, lw=0)
    ax.text(1.45, 1.055, "orientation-useful\nshell (step 2)", ha="center",
            fontsize=7.3, color=SPECBLUE)
    style = {"gated_dt1.5": ("o", LG, r"$\Delta/\tau$ = 1.5"),
             "gated_dt1.95": ("s", NEWMAG, r"$\Delta/\tau$ = 1.95  (housing)"),
             "gated_dt2.5": ("^", NAVY, r"$\Delta/\tau$ = 2.5"),
             "gated_dt2.5_pi35": ("^", NAVY, None),
             "gated_dt5.0": ("D", OKGREEN, r"$\Delta/\tau$ = 5.0")}
    s25 = agg[agg.cell.isin(["gated_dt2.5", "gated_dt2.5_pi35"])].sort_values("dos")
    ax.plot(s25.dos, s25.power, "-", color=NAVY, lw=1.2, alpha=0.45, zorder=1)
    for cell, (mk, col, lab) in style.items():
        s = agg[agg.cell == cell]
        ax.plot(s.dos, s.power, mk, color=col, ms=7, label=lab, mec="white",
                mew=0.8, zorder=3, ls="none")
    ax.plot(agg.dos, agg.mm, ":", color=LIMRED, lw=1.5, marker="x", ms=5.5,
            label="minimum-mass rule passes", zorder=2)
    ax.plot([14.0], [h.minmass_pass.mean()], "x", color=LIMRED, ms=5.5)
    ax.plot([14.0], [(h.lrt > thr).mean()], "o", color=MUTED, ms=5)
    ax.text(14.0, -0.10, "honest\n(no gate)", fontsize=6.8, color=MUTED, ha="center")
    ax.axhline(0.05, color=MUTED, lw=0.8, ls="--")
    ax.text(0.36, 0.085, "nominal 0.05", fontsize=7.2, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([0.4, 0.8, 1.45, 2.6, 5.1, 11.0])
    ax.set_xticklabels(["0.4", "0.8", "1.28-1.64", "2.6", "5.1", "11"], fontsize=7.5)
    ax.set_xlim(0.32, 19)
    ax.set_xlabel(r"$d/\sigma_s$   (distance to the gate, in probe scales)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.16, 1.10)
    ax.set_title("B.  power peaks where orientation does", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.02, 0.93))

    # ---------------------------------------------------------------- panel 3
    ax = axes[2]
    cmap = {0.50: NAVY, 0.35: SPECBLUE, 0.25: OKGREEN, 0.10: NEWMAG, 0.05: LIMRED}
    for pi, s in floor.groupby("pi"):
        s = s.sort_values("sep")
        ax.plot(s.sep, s.fire_calibrated, "-o", color=cmap[pi], ms=4, lw=1.4,
                label=fr"$\pi$ = {pi:.2f}", mec="white", mew=0.6)
    ax.axvline(2.0, color=MUTED, ls="--", lw=0.9)
    ax.text(2.12, 0.60, "the ~2 sd floor\nv5 states", fontsize=7.4, color=MUTED)
    ax.axvspan(1.6, 1.95, color=NEWMAG, alpha=0.13, lw=0)
    ax.text(1.775, 0.30, "real scaffold", fontsize=7.0, color=NEWMAG, ha="center",
            rotation=90, va="center")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel(r"component separation  ($\sigma$)")
    ax.set_ylabel("dip power (pipeline-calibrated threshold)")
    ax.set_title(r"C.  the dip's floor depends on $\pi$", loc="left")
    ax.legend(fontsize=7.3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(1.35, 7.3)

    fig.tight_layout(w_pad=2.0)
    out = os.path.join(HERE, "fig_bootstrap.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
