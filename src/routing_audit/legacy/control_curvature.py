"""
Negative control: can the log-range statistic tell a boundary from smooth curvature?
SUPERSEDED — see legacy/README.md. Kept for provenance; not used by run_all.sh.
Subsumed by the modality-vs-dispersion finding (dispersion statistics are demoted
entirely, not just on this control) — see the current dip-discriminator toy instead
(scripts/run_dip_discriminator_toy.py).

routing_audit_v2 2.3 registers "smooth curved single-regime functions" as a control,
and 3.1 warns that curvature makes r_i rise with sigma under the null. This measures
whether R_log separates the two, and whether the isotonic-residual statistic does.

Analytic expectation
--------------------
  curvature   h(x) = b x + c x^2
              Var over a Gaussian probe ~ sigma^2 b^2 + 2 c^2 sigma^4
              r(sigma) = sqrt(b^2 + 2 c^2 sigma^2)/tau      -> MONOTONE INCREASING

  gate        f(x) = b x - D 1[x >= tau], anchor at signed distance s
              crossing fraction p(sigma) = Phi(-s/sigma)
              Var ~ sigma^2 b^2 + D^2 p(1-p)
              r(sigma) = sqrt(b^2 + D^2 p(1-p)/sigma^2)/tau -> PEAKED

  R_log = max - min scores both shapes. T_iso scores only the non-monotone one.

Run:  python -m src.routing_audit.legacy.control_curvature
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..dispersion import dispersion_curve, r_log, t_iso, MIN_VALID_SCALES
from ..paths import FIGURES

RNG = np.random.default_rng(20260728)

TAU_GATE = 0.5
DELTA = 0.3          # log-scale penalty, codebase default
SLOPE = 1.0
TAU_OBS = 0.02
M = 400
SIGMAS = np.geomspace(0.01, 0.5, 16)
N_ANCHOR = 300
BOUNDARY_WIN = 0.12


def make_linear():
    return lambda x: SLOPE * x[:, 0]

def make_quadratic(c):
    return lambda x: SLOPE * x[:, 0] + c * x[:, 0] ** 2

def make_wiggly(amp, ell):
    """Smooth single branch with an explicit curvature lengthscale -- the closest
    analogue to drawing from the hierarchical GP null of v2 3.3."""
    return lambda x: SLOPE * x[:, 0] + amp * np.sin(2 * np.pi * x[:, 0] / ell)

def make_gated():
    return lambda x: SLOPE * x[:, 0] - DELTA * (x[:, 0] >= TAU_GATE)

def make_kink(s2):
    """v2 2.3 kink control: one mechanism, no routing, sharp derivative change.
    Expected to flag -- that is the intended behavior (v2 2.3)."""
    return lambda x: SLOPE * x[:, 0] + s2 * np.maximum(x[:, 0] - TAU_GATE, 0.0)


def run_model(f, anchors, rng):
    R, T, A = [], [], []
    for a in anchors:
        dc = dispersion_curve(f, np.array([a]), SIGMAS, M, TAU_OBS, rng)
        if dc["valid"].sum() < MIN_VALID_SCALES:
            R.append(np.nan); T.append(np.nan); A.append(True); continue
        R.append(r_log(dc["r"], dc["valid"]))
        T.append(t_iso(dc["r"], dc["sigmas"], dc["valid"]))
        A.append(False)
    return np.array(R), np.array(T), np.array(A)


def rate(v, thr):
    v = v[np.isfinite(v)]
    return float(np.mean(v >= thr)) if v.size else np.nan


def main():
    anchors = np.linspace(0.05, 0.95, N_ANCHOR)
    boundary = np.abs(anchors - TAU_GATE) < BOUNDARY_WIN

    models = {
        "null: linear":               make_linear(),
        "null: quadratic c=1":        make_quadratic(1.0),
        "null: quadratic c=3":        make_quadratic(3.0),
        "null: wiggly A=.05 l=.5":    make_wiggly(0.05, 0.5),
        "null: wiggly A=.05 l=.2":    make_wiggly(0.05, 0.2),
        "null: wiggly A=.10 l=.2":    make_wiggly(0.10, 0.2),
        "null: wiggly A=.10 l=.1":    make_wiggly(0.10, 0.1),
        "control: kink":              make_kink(1.5),
        "ALT: gated":                 make_gated(),
    }

    res = {n: run_model(f, anchors, RNG) for n, f in models.items()}

    # calibrate both statistics to a 5% flag rate on the linear null
    bR, bT, _ = res["null: linear"]
    thr = {"R_log": float(np.nanquantile(bR, 0.95)),
           "T_iso": float(np.nanquantile(bT, 0.95))}
    print(f"thresholds calibrated to 5% on the linear null:  "
          f"R_log={thr['R_log']:.3f}   T_iso={thr['T_iso']:.3f}\n")

    hdr = f"{'model':<26}{'R_log':>9}{'T_iso':>9}   {'target':<26}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for name, (R, T, A) in res.items():
        if name == "ALT: gated":
            R, T = R[boundary], T[boundary]
            tgt = "POWER (want high)"
        else:
            tgt = "false positive (want .05)"
        fr, ft = rate(R, thr["R_log"]), rate(T, thr["T_iso"])
        summary[name] = (fr, ft)
        print(f"{name:<26}{fr:>9.3f}{ft:>9.3f}   {tgt:<26}")

    # worst-case null flag rate = the number that matters
    nulls = [k for k in res if k.startswith("null")]
    wR = max(summary[k][0] for k in nulls)
    wT = max(summary[k][1] for k in nulls)
    pR, pT = summary["ALT: gated"]
    print(f"\nworst-case null flag rate   R_log={wR:.3f}   T_iso={wT:.3f}")
    print(f"power on the gate           R_log={pR:.3f}   T_iso={pT:.3f}")
    print(f"power / worst-null ratio    R_log={pR/max(wR,1e-9):.2f}x  T_iso={pT/max(wT,1e-9):.2f}x")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    demo = [
        ("interior, linear (flat)",     make_linear(),            0.15, "#4C72B0"),
        ("smooth curvature (null)",     make_wiggly(0.10, 0.1),   0.15, "#DD8452"),
        ("boundary-adjacent (GATED)",   make_gated(),   TAU_GATE - 0.04, "#C44E52"),
    ]
    for label, f, a, col in demo:
        dc = dispersion_curve(f, np.array([a]), SIGMAS, 8000, TAU_OBS, RNG)
        v = dc["valid"]
        ax[0].plot(SIGMAS[v], dc["r"][v], "o-", color=col, label=label, lw=2, ms=4)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"probe scale $\sigma$")
    ax[0].set_ylabel(r"dispersion  $r_i(\sigma)$")
    ax[0].set_title("Shape, not size, is the signal", fontsize=11)
    ax[0].legend(fontsize=8, frameon=False)

    names = ["null: linear", "null: wiggly A=.10 l=.1", "null: quadratic c=3",
             "control: kink", "ALT: gated"]
    short = ["linear\nnull", "curved\nnull", "quadratic\nnull", "kink\ncontrol", "GATED"]
    xs = np.arange(len(names)); width = 0.38
    for k, stat in enumerate(("R_log", "T_iso")):
        vals = [summary[n][k] for n in names]
        ax[1].bar(xs + (k - 0.5) * width, vals, width,
                  color=["#8C8C8C", "#2E7D32"][k],
                  label={"R_log": r"$R^{\log}$  (v2 S1)",
                         "T_iso": r"$T_{\rm iso}$  (proposed)"}[stat])
    ax[1].axhline(0.05, ls="--", c="k", lw=1, label="nominal 5%")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(short, fontsize=8)
    ax[1].set_ylabel("flag rate")
    ax[1].set_title("Null flag rates vs. power on the gate", fontsize=11)
    ax[1].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "legacy_fig_control_curvature.png"
    fig.savefig(out, dpi=180)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
