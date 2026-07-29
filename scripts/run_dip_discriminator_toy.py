#!/usr/bin/env python3
"""
Toy validation: does MODALITY separate a gate from smooth structure when DISPERSION
cannot? Writes results/figures/fig_dip_discriminator.png (a CURRENT supporting figure,
see docs/results_2026-07-28.md S1/S7). Self-contained -- no Seattle data required.

Finding that motivates this: a hard gate viewed through a probe of scale sigma is
observationally a transition of width ~sigma. A stationary GP with lengthscale
ell ~ sigma reproduces the same dispersion curve, so no statistic computed from
r_i(sigma) alone -- range, shape, or studentized -- can separate them. Confirmed
empirically: the calibrated T_scale has 0.00 power once ell_min is small enough
for the null to fit the bump.

But the two differ in the DISTRIBUTION of responses, not their spread:
  gate  -> some probes land on the far branch and get f - Delta; the rest do not.
           Responses are BIMODAL, separated by the penalty Delta.
  GP    -> responses are a smooth transformation of a Gaussian. UNIMODAL.

So Hartigan's dip at sigma* is not corroboration of the dispersion evidence.
It is the discriminator, and dispersion's job is only to localize sigma*.
"""
import sys
from pathlib import Path

import numpy as np
import diptest
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.routing_audit.dispersion import dispersion_curve, r_log, MIN_VALID_SCALES
from src.routing_audit.gp_utils import rff_gp_path
from src.routing_audit import paths

RNG = np.random.default_rng(20260728)

D = 1
TAU_GATE, DELTA, SLOPE = 0.5, 0.3, 1.0
TAU_OBS = 0.02
M = 250
M_CONFIRM = 600          # fresh S2 sample at sigma*
SIGMAS = np.geomspace(0.01, 0.5, 16)
N_ANCHOR = 150
BOUNDARY_WIN = 0.12


def gp_null(amp, ell, seed):
    g = rff_gp_path(D, amp, ell, 256, np.random.default_rng(seed))
    return lambda x: SLOPE * x[:, 0] + g(x)

def linear():  return lambda x: SLOPE * x[:, 0]
def gated():   return lambda x: SLOPE * x[:, 0] - DELTA * (x[:, 0] >= TAU_GATE)
def kink(s2):  return lambda x: SLOPE * x[:, 0] + s2 * np.maximum(x[:, 0] - TAU_GATE, 0.0)


def run(f, anchors, rng):
    R, DIP, DIPP, SS = [], [], [], []
    for a in anchors:
        dc = dispersion_curve(f, np.array([a]), SIGMAS, M, TAU_OBS, rng)
        if dc["valid"].sum() < MIN_VALID_SCALES:
            R.append(np.nan); DIP.append(np.nan); DIPP.append(np.nan); SS.append(np.nan)
            continue
        R.append(r_log(dc["r"], dc["valid"]))
        # sigma* from the dispersion curve -- this is ALL dispersion is used for
        t = int(np.argmax(np.where(dc["valid"], dc["r"], -np.inf)))
        s = SIGMAS[t]; SS.append(s)
        # fresh confirmation sample at sigma* (v2 budget split S2)
        delta = rng.normal(0.0, s, size=(M_CONFIRM, D))
        y = f(np.array([a])[None, :] + delta) + rng.normal(0.0, TAU_OBS, size=M_CONFIRM)
        d_, p_ = diptest.diptest(np.ascontiguousarray(y / TAU_OBS))
        DIP.append(d_); DIPP.append(p_)
    return (np.array(R, float), np.array(DIP, float),
            np.array(DIPP, float), np.array(SS, float))


def main():
    anchors = np.linspace(0.05, 0.95, N_ANCHOR)
    boundary = np.abs(anchors - TAU_GATE) < BOUNDARY_WIN

    models = {
        "null: linear":         linear(),
        "null: GP A=.05 l=.20": gp_null(0.05, 0.20, 11),
        "null: GP A=.10 l=.10": gp_null(0.10, 0.10, 12),
        "null: GP A=.10 l=.05": gp_null(0.10, 0.05, 13),
        "null: GP A=.20 l=.10": gp_null(0.20, 0.10, 14),
        "null: GP A=.05 l=.03": gp_null(0.05, 0.03, 15),
        "null: GP A=.20 l=.03": gp_null(0.20, 0.03, 16),
        "control: kink":        kink(1.5),
        "ALT: gated":           gated(),
    }

    res = {}
    for name, f in models.items():
        res[name] = run(f, anchors, RNG)
        print(f"  {name}", flush=True)

    thrR = float(np.nanquantile(res["null: linear"][0], 0.95))
    ALPHA = 0.05

    print(f"\nm={M} confirm={M_CONFIRM} anchors={N_ANCHOR}")
    print(f"R_log threshold at 5% on linear null = {thrR:.3f}"
          f"    dip flagged when p < {ALPHA}\n")
    hdr = f"{'model':<24}{'R_log':>9}{'dip':>9}   {'target':<22}"
    print(hdr); print("-" * len(hdr))
    summ = {}
    for name, (R, DP, PP, SS) in res.items():
        sel = boundary if name == "ALT: gated" else np.ones(len(anchors), bool)
        r_ = R[sel]; p_ = PP[sel]
        fr = float(np.nanmean(r_[np.isfinite(r_)] >= thrR))
        fd = float(np.nanmean(p_[np.isfinite(p_)] < ALPHA))
        summ[name] = (fr, fd)
        tgt = "POWER (want high)" if name == "ALT: gated" else "false pos (want .05)"
        print(f"{name:<24}{fr:>9.3f}{fd:>9.3f}   {tgt:<22}")

    nulls = [k for k in res if not k.startswith("ALT")]
    wR = max(summ[k][0] for k in nulls); wD = max(summ[k][1] for k in nulls)
    pR, pD = summ["ALT: gated"]
    print(f"\nWORST-CASE null   R_log={wR:.3f}   dip={wD:.3f}")
    print(f"power on gate     R_log={pR:.3f}   dip={pD:.3f}")
    print(f"power/worst-null  R_log={pR/max(wR,1e-9):.2f}x   dip={pD/max(wD,1e-9):.2f}x")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))

    for k, (name, col, a) in enumerate([
            ("null: GP A=.20 l=.03", "#DD8452", 0.20),
            ("ALT: gated", "#C44E52", TAU_GATE - 0.03)]):
        f = models[name]
        dc = dispersion_curve(f, np.array([a]), SIGMAS, M, TAU_OBS, RNG)
        t = int(np.argmax(np.where(dc["valid"], dc["r"], -np.inf)))
        delta = RNG.normal(0.0, SIGMAS[t], size=(3000, D))
        y = f(np.array([a])[None, :] + delta) + RNG.normal(0, TAU_OBS, 3000)
        ax[k].hist(y, bins=60, color=col, alpha=0.85)
        d_, p_ = diptest.diptest(np.ascontiguousarray(y))
        ax[k].set_title(f"{'smooth GP null' if k==0 else 'GATED'}\n"
                        f"dip={d_:.4f}  p={p_:.3f}", fontsize=10)
        ax[k].set_xlabel(r"response at $\sigma^\star$")
    ax[0].set_ylabel("count")

    names = ["null: linear", "null: GP A=.10 l=.10", "null: GP A=.20 l=.10",
             "null: GP A=.20 l=.03", "control: kink", "ALT: gated"]
    short = ["linear", "GP\n.1/.1", "GP\n.2/.1", "GP\n.2/.03", "kink", "GATED"]
    xs = np.arange(len(names)); w = 0.38
    ax[2].bar(xs - w/2, [summ[n][0] for n in names], w, color="#8C8C8C",
              label=r"$R^{\log}$ (dispersion)")
    ax[2].bar(xs + w/2, [summ[n][1] for n in names], w, color="#2E7D32",
              label="dip at $\\sigma^\\star$ (modality)")
    ax[2].axhline(0.05, ls="--", c="k", lw=1, label="nominal 5%")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(short, fontsize=8)
    ax[2].set_ylabel("flag rate"); ax[2].set_ylim(0, 1.05)
    ax[2].set_title("Modality separates; dispersion does not", fontsize=10)
    ax[2].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / "fig_dip_discriminator.png"
    fig.savefig(out, dpi=180)
    np.savez(paths.RESULTS / "dip_toy.npz", anchors=anchors, boundary=boundary, thrR=thrR,
             **{f"{k}|{s}": res[k][i] for k in res
                for i, s in enumerate(("R", "dip", "dip_p", "sstar"))})
    print(f"\nwrote {out}, {paths.RESULTS / 'dip_toy.npz'}")


if __name__ == "__main__":
    main()
