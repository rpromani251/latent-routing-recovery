"""
Experiment A -- 3-rung Delta-invariance x LRT, end to end.
RE-ACTIVATED 13 August: task 7 showed pooling does NOT reject long-wavelength
resonance (C fires at 1.000 on a surface with no boundary), so per-anchor
confound rejection is back on the critical path.

Runs as ONE experiment because the same EM fit yields both Delta_hat and the LRT
statistic, and because a more sensitive detector should also false-alarm harder
on resonant curvature -- the 0.169 figure in the July review was measured with a
detector having ZERO power at 1.5 sigma.

  detector  in {dip, LRT, dip OR LRT}
  rule      in {naive (top rung only), 2-rung, 3-rung deepest-estimable}
  surface   in {honest smooth, honest resonant x3 wavelengths,
                gated, gated + resonant}

CALIBRATION (task 9, folded in). The LRT threshold is recalibrated PER RUNG on
trimmed residuals from honest-smooth pipeline anchors. The 6.08 figure from clean
Gaussian draws is not reused: LTS truncates the tails before the LRT sees them,
which deflates the null.

THE ESTIMABILITY CONSTRAINT -- and it is sharper than the July design assumed.
For a gate at distance d, the crossing fraction at scale sigma_s is
Phi(-d/sigma_s), so the minimum-mass rule (pi >= 0.05) requires
sigma_s >= d / 1.645. With pi_top = 0.10 we have d = 1.28 sigma, so no rung below
0.78 sigma is estimable -- less than ONE halving. A 3-rung ladder reaching r/4
is therefore vacuous at low pi.

This collides with step 2, which found orientation recovery is best at LOW pi.
So the experiment sweeps pi_top explicitly: the question is not only whether the
rule works but whether any pi serves both jobs at once.

Resonant wavelengths are set relative to the probe diameter L = 2*sigma*sqrt(2),
since the resonance band is defined in units of probe diameter, not absolutely.
"""
import numpy as np
import pandas as pd
import importlib.util
from scipy.stats import norm

spec = importlib.util.spec_from_file_location("p", "/tmp/pexp/exp_p_pooling.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
dspec = importlib.util.spec_from_file_location("dip", "/tmp/dist_est/dip.py")
DIP = importlib.util.module_from_spec(dspec); dspec.loader.exec_module(DIP)

SIGMA, TAU, M = 0.20, 0.02, 800
L_PROBE = 2.0 * SIGMA * np.sqrt(2)          # ~0.566
RUNGS = [1.0, 0.5, 0.25]                    # multiples of sigma_top
PI_MIN = 0.05                               # minimum-mass rule
N_ANCHOR = 220
SEED = 20260813
ALPHA = 0.05

NT = np.array([np.cos(0.7), np.sin(0.7)])
TG = np.array([-NT[1], NT[0]])
BETA = 0.15 * np.array([np.cos(2.1), np.sin(2.1)])

SURFACES = {
    "honest":            dict(gate=False, amp=0.0,  ell=None),
    "resonant_0.5L":     dict(gate=False, amp=2.5 * TAU, ell=0.5 * L_PROBE),
    "resonant_1.0L":     dict(gate=False, amp=2.5 * TAU, ell=1.0 * L_PROBE),
    "resonant_1.5L":     dict(gate=False, amp=2.5 * TAU, ell=1.5 * L_PROBE),
    "gated":             dict(gate=True,  amp=0.0,  ell=None),
    "gated+resonant":    dict(gate=True,  amp=2.5 * TAU, ell=1.0 * L_PROBE),
}
PI_TOP = [0.10, 0.20, 0.35]
DT = 2.5


def respond(T, spec_, rng):
    y = T @ BETA + rng.normal(0.0, TAU, len(T))
    if spec_["gate"]:
        y = y - DT * TAU * ((T @ NT) > 0.0)
    if spec_["amp"] > 0:
        y = y + spec_["amp"] * np.sin(2 * np.pi * (T @ NT) / spec_["ell"])
    return y


def ladder(t_a, spec_, rng):
    """Run the 3-rung ladder at one anchor. Returns a per-rung record list."""
    rec = []
    for k in RUNGS:
        s = k * SIGMA
        Z = rng.normal(0.0, s, size=(M, 2))
        T = t_a[None, :] + Z
        y = respond(T, spec_, rng)
        resid = P.lts_residuals(Z, y)
        fit = P.gmm2_equalvar(resid, rng)
        pi_hat = float(min(fit["w"]))
        gap = float(abs(fit["mu"][1] - fit["mu"][0]))
        _, dp = DIP.dip_pvalue(resid)
        rec.append(dict(k=k, sigma=s, lrt=float(max(fit["lrt"], 0.0)),
                        dip_p=float(dp), pi_hat=pi_hat, gap=gap,
                        estimable=bool(pi_hat * M >= PI_MIN * M and pi_hat >= PI_MIN)))
    return rec


def alpha_exponent(rec):
    """Regress log Delta_hat on log r over ESTIMABLE rungs only. Returns
    (alpha, n_rungs_used). alpha ~ 0 step, ~1 kink, ~2 curvature."""
    e = [r for r in rec if r["estimable"] and r["gap"] > 1e-9]
    if len(e) < 2:
        return np.nan, len(e)
    x = np.log([r["sigma"] for r in e]); yv = np.log([r["gap"] for r in e])
    a = np.polyfit(x, yv, 1)[0]
    return float(a), len(e)


def decide(rec, detector, rule, lrt_thr):
    """Returns (fired, abstained). lrt_thr is per-rung, keyed by k."""
    top = rec[0]
    fired_top = ((top["dip_p"] < ALPHA) if detector == "dip"
                 else (top["lrt"] > lrt_thr[top["k"]]) if detector == "LRT"
                 else (top["dip_p"] < ALPHA or top["lrt"] > lrt_thr[top["k"]]))
    if not fired_top:
        return False, False
    if rule == "naive":
        return True, False
    if rule == "2-rung":
        r2 = rec[1]
        if not r2["estimable"] or top["gap"] <= 1e-9:
            return False, True                      # cannot check -> A5 abstain
        return (r2["gap"] / top["gap"]) > 0.5, False
    a, nr = alpha_exponent(rec)                     # 3-rung, deepest estimable
    if not np.isfinite(a) or nr < 2:
        return False, True
    return a < 0.5, False


def main():
    rng = np.random.default_rng(SEED)
    # ---- calibration: LRT null per rung, from honest-smooth pipeline anchors
    cal = []
    for _ in range(300):
        t_a = NT * rng.normal(0, 0.8) + TG * rng.normal(0, 0.8)
        cal.append(ladder(t_a, SURFACES["honest"], rng))
    lrt_thr = {k: float(np.quantile([r[i]["lrt"] for r in cal], 1 - ALPHA))
               for i, k in enumerate(RUNGS)}
    print("  LRT thresholds recalibrated on trimmed residuals (NOT 6.08):")
    for k in RUNGS:
        v = [r[RUNGS.index(k)]["lrt"] for r in cal]
        print(f"    rung r*{k:<5} threshold {lrt_thr[k]:7.2f}   null median {np.median(v):6.2f}")

    rows = []
    for pit in PI_TOP:
        d0 = -SIGMA * norm.ppf(pit)
        est_floor = d0 / 1.645
        print(f"\n  pi_top={pit}: d={d0:.3f}, deepest estimable sigma={est_floor:.3f} "
              f"({est_floor/SIGMA:.2f} x sigma_top) -> rungs usable: "
              f"{[k for k in RUNGS if k*SIGMA >= est_floor]}")
        for sname, sp in SURFACES.items():
            recs = []
            for _ in range(N_ANCHOR):
                if sp["gate"]:
                    t_a = NT * (d0 * rng.uniform(.95, 1.05) * rng.choice([-1., 1.])) \
                        + TG * rng.normal(0, 0.8)
                else:
                    t_a = NT * rng.normal(0, 0.8) + TG * rng.normal(0, 0.8)
                recs.append(ladder(t_a, sp, rng))
            for det in ("dip", "LRT", "dip|LRT"):
                for rule in ("naive", "2-rung", "3-rung"):
                    f = [decide(r, det, rule, lrt_thr) for r in recs]
                    rows.append(dict(pi_top=pit, surface=sname, detector=det,
                                     rule=rule,
                                     fire=float(np.mean([a for a, _ in f])),
                                     abstain=float(np.mean([b for _, b in f]))))
            al = [alpha_exponent(r)[0] for r in recs]
            nr = [alpha_exponent(r)[1] for r in recs]
            rows.append(dict(pi_top=pit, surface=sname, detector="-", rule="alpha",
                             fire=float(np.nanmedian(al)),
                             abstain=float(np.mean(np.array(nr) < 2))))
            print(f"    {sname:16s} alpha med {np.nanmedian(al):6.2f}  "
                  f"rungs<2 {np.mean(np.array(nr)<2):.2f}", flush=True)
    pd.DataFrame(rows).to_csv("/tmp/aexp/a_invariance_rows.csv", index=False)
    print("\nwrote a_invariance_rows.csv")


if __name__ == "__main__":
    main()
