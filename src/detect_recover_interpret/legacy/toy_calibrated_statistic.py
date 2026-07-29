"""
TOY EXPERIMENT v2: anchor-specific calibrated statistic vs. the negative controls.
SUPERSEDED — see legacy/README.md. Kept for provenance; not used by run_all.sh.

  R_log     v2 3.2 S1, uncalibrated log-range
  T_scale   v2 3.2 S2, studentized against the anchor's OWN fitted null,
            with the lengthscale floored at a registered ell_min

Nulls are literal draws from the registered class (SE-GP sample paths + linear mean).
The number that matters is the WORST-CASE null flag rate.

Run:  python -m src.detect_recover_interpret.legacy.toy_calibrated_statistic
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..dispersion import dispersion_curve, r_log, MIN_VALID_SCALES
from ..paths import FIGURES
from ..gp_utils import rff_gp_path
from .null_generator import fit_null_hyper
from .null_grid import build_grid, studentize_anchor

RNG = np.random.default_rng(20260728)

D = 1
TAU_GATE, DELTA, SLOPE = 0.5, 0.3, 1.0
TAU_OBS = 0.02
M = 250
SIGMAS = np.geomspace(0.01, 0.5, 16)
N_ANCHOR = 150
BOUNDARY_WIN = 0.12

ELL_MIN = 0.05      # REGISTERED: declares what counts as smooth
SHRINK = 0.35       # hierarchical pull of each anchor's (log rho, log ell) to the pop mean

RHO_G = np.geomspace(0.02, 3.0, 7)
ELL_G = np.geomspace(ELL_MIN, 1.5, 7)


def gp_null(amp, ell, seed):
    g = rff_gp_path(D, amp, ell, 256, np.random.default_rng(seed))
    return lambda x: SLOPE * x[:, 0] + g(x)

def linear():  return lambda x: SLOPE * x[:, 0]
def gated():   return lambda x: SLOPE * x[:, 0] - DELTA * (x[:, 0] >= TAU_GATE)
def kink(s2):  return lambda x: SLOPE * x[:, 0] + s2 * np.maximum(x[:, 0] - TAU_GATE, 0.0)


def run(f, anchors, grid, rng):
    R, T = [], []
    dcs, hyps = [], []
    for a in anchors:
        dc = dispersion_curve(f, np.array([a]), SIGMAS, M, TAU_OBS, rng)
        dcs.append(dc)
        if dc["valid"].sum() < MIN_VALID_SCALES:
            R.append(np.nan); hyps.append(None); continue
        R.append(r_log(dc["r"], dc["valid"]))
        hyps.append(fit_null_hyper(dc["r"], dc["sigmas"], dc["valid"], TAU_OBS, D, ELL_MIN))

    # population mean of (log rho, log ell) over the trimmed anchors -> shrinkage target
    lg = np.array([[np.log(h["s"] / h["beta"]), np.log(h["ell"])]
                   for h in hyps if h is not None and h["beta"] > 0 and h["s"] > 0])
    Rarr = np.array(R, float)
    if lg.size:
        keep = Rarr[[i for i, h in enumerate(hyps)
                     if h is not None and h["beta"] > 0 and h["s"] > 0]]
        cut = np.nanquantile(keep, 0.80)
        tgt = lg[keep <= cut].mean(0) if np.isfinite(cut) else lg.mean(0)
    else:
        tgt = None

    for dc, h in zip(dcs, hyps):
        t, _, _ = studentize_anchor(dc["r"], dc["lam"], dc["valid"], SIGMAS, TAU_OBS,
                                    h, grid, M, shrink_to=tgt, shrink_w=SHRINK)
        T.append(t)
    return Rarr, np.array(T, float), dcs


def main():
    print(f"building null grid ({len(RHO_G)}x{len(ELL_G)})...", flush=True)
    grid = build_grid(SIGMAS, D, M, RHO_G, ELL_G, 40, np.random.default_rng(3))

    anchors = np.linspace(0.05, 0.95, N_ANCHOR)
    boundary = np.abs(anchors - TAU_GATE) < BOUNDARY_WIN

    models = {
        "null: linear":         linear(),
        "null: GP A=.05 l=.20": gp_null(0.05, 0.20, 11),
        "null: GP A=.10 l=.10": gp_null(0.10, 0.10, 12),
        "null: GP A=.10 l=.05": gp_null(0.10, 0.05, 13),
        "null: GP A=.20 l=.10": gp_null(0.20, 0.10, 14),
        "null: GP A=.05 l=.03": gp_null(0.05, 0.03, 15),
        "control: kink":        kink(1.5),
        "ALT: gated":           gated(),
    }

    res = {}
    for name, f in models.items():
        R, T, dcs = run(f, anchors, grid, RNG)
        res[name] = {"R": R, "T": T, "dcs": dcs}
        print(f"  {name}", flush=True)

    thrR = float(np.nanquantile(res["null: linear"]["R"], 0.95))
    thrT = float(np.nanquantile(res["null: linear"]["T"], 0.95))
    print(f"\nell_min={ELL_MIN}  shrink={SHRINK}  m={M}  anchors={N_ANCHOR}")
    print(f"thresholds at 5% on linear null:  R_log={thrR:.3f}  T_scale={thrT:.3f}\n")

    def rate(v, t):
        v = v[np.isfinite(v)]
        return float(np.mean(v >= t)) if v.size else np.nan

    hdr = f"{'model':<24}{'R_log':>9}{'T_scale':>10}   {'target':<22}"
    print(hdr); print("-" * len(hdr))
    summ = {}
    for name, d_ in res.items():
        sel = boundary if name == "ALT: gated" else np.ones(len(anchors), bool)
        fr, ft = rate(d_["R"][sel], thrR), rate(d_["T"][sel], thrT)
        summ[name] = (fr, ft)
        tgt = "POWER (want high)" if name == "ALT: gated" else "false pos (want .05)"
        print(f"{name:<24}{fr:>9.3f}{ft:>10.3f}   {tgt:<22}")

    nulls = [k for k in res if not k.startswith("ALT")]
    wR = max(summ[k][0] for k in nulls); wT = max(summ[k][1] for k in nulls)
    pR, pT = summ["ALT: gated"]
    print(f"\nWORST-CASE null   R_log={wR:.3f}   T_scale={wT:.3f}")
    print(f"power on gate     R_log={pR:.3f}   T_scale={pT:.3f}")
    print(f"power/worst-null  R_log={pR/max(wR,1e-9):.2f}x   T_scale={pT/max(wT,1e-9):.2f}x")

    FIGURES.mkdir(parents=True, exist_ok=True)
    np.savez(FIGURES / "toy_v2.npz", anchors=anchors, boundary=boundary,
             thrR=thrR, thrT=thrT,
             **{f"{k}|{s}": res[k][s] for k in res for s in ("R", "T")})
    print(f"\nwrote {FIGURES / 'toy_v2.npz'}")


if __name__ == "__main__":
    main()
