"""Figures for the known-regimes simulation suite."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_OK, C_BAD, C_N = "#2E7D32", "#C44E52", "#8C8C8C"


def fig_main():
    sn = pd.read_csv("sim1d_noise_summary.csv")
    bd = pd.read_csv("sim1d_boundary_summary.csv")
    rs = pd.read_csv("sim1d_robustness_summary.csv")
    p2 = pd.read_csv("sim2d_pi_summary.csv")

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))

    # (a) noise axis
    a = ax[0]
    a.plot(sn.tau_obs, sn.power, "o-", color=C_OK, label="power (detectable)")
    a.plot(sn.tau_obs, sn.fp_honest, "s--", color=C_BAD, label="FP honest")
    a.plot(sn.tau_obs, sn.abstain_honest, "^:", color=C_N, label="abstain honest")
    a.axvline(0.30, ls=":", c="k", lw=0.8)
    a.set_xscale("log"); a.set_xlabel(r"observation noise $\tau_{obs}$")
    a.set_ylabel("rate"); a.set_title(r"noise sweep ($\Delta=0.3$), 1-D", fontsize=10)
    a.legend(fontsize=7, frameon=False); a.set_ylim(-0.03, 1.05)

    # (b) boundary axis: power vs pi_true (1-D and 2-D)
    a = ax[1]
    labels = [str(iv) for iv in bd.pi_bin]
    x = np.arange(len(bd))
    a.bar(x - 0.2, bd.flag_rate, 0.38, color=C_OK, label="1-D")
    a.bar(x + 0.2, p2.flag_rate, 0.38, color="#1f77b4", label="2-D")
    a.set_xticks(x); a.set_xticklabels(labels, rotation=30, fontsize=6)
    a.axhline(1.0, ls=":", c="k", lw=0.6)
    a.set_xlabel(r"true mixing fraction $\pi_{true}$"); a.set_ylabel("flag rate")
    a.set_title("power near the regime boundary", fontsize=10)
    a.legend(fontsize=7, frameon=False)

    # (c) recovered penalty vs noise
    a = ax[2]
    m = sn.dropna(subset=["delta_pmin_med"])
    a.errorbar(m.tau_obs, m.delta_pmin_med,
               yerr=[m.delta_pmin_med - m.delta_pmin_q25,
                     m.delta_pmin_q75 - m.delta_pmin_med],
               fmt="o-", color=C_OK, capsize=3, label=r"$\hat\Delta$ at $p_{min}$ scale")
    a.plot(sn.tau_obs, sn.delta_honest, "s--", color=C_N, label="honest model")
    a.axhline(0.30, ls="--", c=C_BAD, lw=1, label=r"true $\Delta=0.30$")
    a.set_xscale("log"); a.set_xlabel(r"$\tau_{obs}$"); a.set_ylabel(r"$\hat\Delta$")
    a.set_title("penalty recovery vs noise", fontsize=10)
    a.legend(fontsize=7, frameon=False)

    # (d) robustness: the lengthscale precondition
    a = ax[3]
    order = ["gp_A.10_l.30_ok", "gp_A.20_l.30_ok", "gp_A.10_l.10_mid",
             "gp_A.10_l.05_bad", "gp_A.20_l.05_bad", "kink_control"]
    rs = rs.set_index("model").loc[order].reset_index()
    cols = [C_OK, C_OK, "#DD8452", C_BAD, C_BAD, C_N]
    a.bar(np.arange(len(rs)), rs.flag_rate, color=cols)
    a.axhline(0.05, ls="--", c="k", lw=0.8, label="nominal 5%")
    a.set_xticks(np.arange(len(rs)))
    a.set_xticklabels(["GP .1/.3\n(l>ladder)", "GP .2/.3\n(l>ladder)",
                       "GP .1/.1\n(mid)", "GP .1/.05\n(violates)",
                       "GP .2/.05\n(violates)", "kink\n(control)"], fontsize=6)
    a.set_ylabel("false-positive rate")
    a.set_title("smoothness precondition is sharp,\ncheckable via ladder vs lengthscale",
                fontsize=9)
    a.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig("fig_sim_known_regimes.png", dpi=180)
    print("wrote fig_sim_known_regimes.png")


def fig_map():
    df = pd.read_csv("sim2d_anchors.csv")
    g = df[df.model == "gated"]
    t = np.linspace(0, 1, 800)
    by = 0.5 + 0.08 * np.sin(4 * np.pi * t)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a in ax:
        a.plot(t, by, "k-", lw=1.2)
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")

    fl = g[g.flag]; nf = g[~g.flag]
    ax[0].scatter(nf.x1, nf.x2, s=8, c="#cccccc", label="not flagged")
    ax[0].scatter(fl.x1, fl.x2, s=14, c="#C44E52", label="flagged")
    ax[0].set_title("dip-scan flags trace the boundary", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False, loc="lower left")

    sc = ax[1].scatter(fl.x1, fl.x2, s=14, c=fl.delta_at_pmin,
                       cmap="viridis", vmin=0, vmax=0.35)
    plt.colorbar(sc, ax=ax[1], label=r"$\hat\Delta$")
    ax[1].set_title(r"recovered penalty $\hat\Delta$ (true 0.30)", fontsize=10)

    fig.tight_layout()
    fig.savefig("fig_sim2d_map.png", dpi=180)
    print("wrote fig_sim2d_map.png")


if __name__ == "__main__":
    fig_main()
    fig_map()
