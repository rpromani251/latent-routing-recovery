"""
Experiment L -- which way should the ladder point?

THE PROBLEM. Stage 5(a) discriminates a gate from curvature by regressing log Delta_hat
on log r across rungs: a step gives r^0, a kink r^1, curvature r^2. The 14 August
calibration run shows that check being measured where it has no signal. With the ladder
topping out at the orientation-useful shell (d/sigma_1 = 1.28) the deeper rungs land at
d/sigma = 2.56 and 5.13, where the gate is simply out of reach -- there are no crossers,
Delta_hat is undefined, and the exponent cannot be fitted at all:

    fire rate by rung        1.0 sigma   0.5 sigma   0.25 sigma
    gated, Delta/tau = 2.5      0.627       0.080        0.033
    resonant 0.5 L              1.000       0.935        0.752

The gate's own trace vanishes fastest. That is not fragility, it is the ladder pointing
the wrong way. Commit 125555a saw the other half: at pi_top = 0.35, where all three rungs
ARE estimable, the r^0 signature is recovered exactly, while at pi_top = 0.10 the exponent
reads 0.37.

THE PARAMETER. For an anchor at distance d from the boundary, let

    kappa = sigma_top / d,      rungs = sigma_top * [1, 1/2, 1/4]

so the deepest rung sits at d/sigma_3 = 4/kappa. The current design is kappa = 0.78
(d/sigma_3 = 5.1, one estimable rung). kappa = 2.6 puts the DEEPEST rung on the shell
(d/sigma_3 = 1.54) and the two above it at d/sigma = 0.77 and 0.385, all estimable.

Operationally this says: probe WIDER. The ladder's largest radius should be a few times
the anchor's distance to the boundary, not below it. The cost is that a wider ball spans
more curvature, so this trades estimability against A11 -- which is exactly the tension the
sweep is meant to price. The resonant wavelengths are therefore held FIXED in absolute
units while the ladder moves, or the comparison would be circular.

PRE-REGISTERED PREDICTIONS.

  L-1  Estimable rungs (by the CALIBRATED test, not the minimum-mass rule, which the
       14 August run showed passes ~99% of pure noise) rise from 1 to 3 as kappa goes
       0.78 -> 2.6, and the fitted exponent becomes measurable at all.

  L-2  Once >= 2 rungs are estimable, the gate's exponent concentrates near 0 and the
       resonant surfaces' near 2, so the exponent SEPARATES them -- which it cannot do
       at kappa = 0.78 because the gate has no second rung to regress against.

  L-3  The A11 cost is real and non-monotone in the other direction: as kappa grows the
       probe ball spans more oscillations, so honest and resonant fire rates rise. There
       should be an interior optimum, and the sweep should locate it.

    python3 exp_ladder_anchor.py
"""
import os
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("p", "exp_p_pooling.py")

# ------------------------------------------------------------------ settings
TAU, M_PROBE = 0.02, 800
SIGMA_REF = 0.20                       # the reference probe scale of Experiments A/B
L_REF = 2.0 * SIGMA_REF * np.sqrt(2)   # reference probe diameter, ~0.566
D_ANCHOR = -SIGMA_REF * norm.ppf(0.10)  # 0.2563: the shell distance at the reference
BETA = 0.15 * np.array([np.cos(2.1), np.sin(2.1)])
NU_TRUE = np.array([1.0, 0.0])
TG = np.array([-NU_TRUE[1], NU_TRUE[0]])
TRIM = 0.75
LRT_THRESH = 5.459                     # the calibrated pipeline null q95 (Experiment B)
PI_MIN, PI_MAX = 0.05, 0.50
RUNG_RATIOS = [1.0, 0.5, 0.25]
N_ANCHOR = 250
SEED = 20260815

# kappa = sigma_top / d.  0.78 is the current design.
KAPPAS = [0.78, 1.3, 2.0, 2.6, 3.5, 5.0]

SURFACES = {
    "gated_dt1.95":  dict(dt=1.95, amp=0.0, ell=None),
    "gated_dt2.5":   dict(dt=2.5,  amp=0.0, ell=None),
    "resonant_0.5L": dict(dt=None, amp=2.5 * TAU, ell=0.5 * L_REF),
    "resonant_1.0L": dict(dt=None, amp=2.5 * TAU, ell=1.0 * L_REF),
    "resonant_1.5L": dict(dt=None, amp=2.5 * TAU, ell=1.5 * L_REF),
    "honest":        dict(dt=None, amp=0.0, ell=None),
}
PARTS = os.path.join(HERE, "_parts_ladder")


def respond(T, surf, rng):
    y = T @ BETA + rng.normal(0.0, TAU, len(T))
    if surf["dt"] is not None:
        y = y - surf["dt"] * TAU * ((T @ NU_TRUE) > 0.0)
    if surf["amp"] > 0:
        y = y + surf["amp"] * np.sin(2 * np.pi * (T @ NU_TRUE) / surf["ell"])
    return y


def rung_stats(Z, y, rng):
    resid = P.lts_residuals(Z, y)
    fit = P.gmm2_equalvar(resid, rng)
    pi_hat = float(min(fit["w"]))
    return dict(lrt=float(max(fit["lrt"], 0.0)), pi_hat=pi_hat,
                gap=float(abs(fit["mu"][1] - fit["mu"][0])),
                fires=bool(fit["lrt"] > LRT_THRESH),
                # the calibrated test replaces the minimum-mass rule as the
                # estimability gate; pi is kept only to bound the LTS breakdown
                estimable=bool(fit["lrt"] > LRT_THRESH and PI_MIN <= pi_hat <= PI_MAX))


def run_unit(args):
    kappa, surf_name = args
    path = os.path.join(PARTS, f"k{kappa}__{surf_name}.csv")
    if os.path.exists(path):
        return path
    surf = SURFACES[surf_name]
    rng = np.random.default_rng(SEED + int(kappa * 1000) + abs(hash(surf_name)) % 9973)
    sigma_top = kappa * D_ANCHOR
    radii = [sigma_top * r for r in RUNG_RATIOS]

    rows = []
    for a in range(N_ANCHOR):
        side = rng.choice([-1.0, 1.0])
        along = rng.normal(0.0, 4.0 * SIGMA_REF)
        t_a = NU_TRUE * (side * D_ANCHOR) + TG * along
        per = []
        for s in radii:
            Z = rng.normal(0.0, s, size=(M_PROBE, 2))
            y = respond(t_a[None, :] + Z, surf, rng)
            st = rung_stats(Z, y, rng)
            st["sigma_s"] = s
            st["d_over_sigma"] = D_ANCHOR / s
            per.append(st)

        est = [p for p in per if p["estimable"]]
        # the scaling exponent, fitted only where Delta_hat is actually defined
        if len(est) >= 2:
            lr = np.log([p["sigma_s"] for p in est])
            lg = np.log([max(p["gap"], 1e-12) for p in est])
            alpha = float(np.polyfit(lr, lg, 1)[0])
        else:
            alpha = np.nan
        rows.append(dict(kappa=kappa, surface=surf_name, anchor=a,
                         sigma_top=sigma_top,
                         n_estimable=len(est), n_fire=sum(p["fires"] for p in per),
                         alpha=alpha,
                         fire_top=per[0]["fires"], fire_mid=per[1]["fires"],
                         fire_deep=per[2]["fires"],
                         pi_top=per[0]["pi_hat"], pi_deep=per[2]["pi_hat"],
                         gap_top=per[0]["gap"], gap_deep=per[2]["gap"],
                         dos_top=per[0]["d_over_sigma"], dos_deep=per[2]["d_over_sigma"]))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    os.makedirs(PARTS, exist_ok=True)
    units = [(k, s) for k in KAPPAS for s in SURFACES]
    todo = [u for u in units
            if not os.path.exists(os.path.join(PARTS, f"k{u[0]}__{u[1]}.csv"))]
    print(f"{len(units)} units, {len(todo)} to run", flush=True)
    if todo:
        import multiprocessing as mp
        nproc = min(int(os.environ.get("NPROC", "2")), max(1, mp.cpu_count()))
        with mp.Pool(nproc) as pool:
            for i, p in enumerate(pool.imap_unordered(run_unit, todo), 1):
                print(f"  [{i}/{len(todo)}] {os.path.basename(p)}", flush=True)
    df = pd.concat([pd.read_csv(os.path.join(PARTS, f"k{u[0]}__{u[1]}.csv"))
                    for u in units], ignore_index=True)
    df.to_csv(os.path.join(HERE, "ladder_anchor_rows.csv"), index=False)

    print("\nL-1  estimable rungs (calibrated test), mean per anchor")
    print(f"  {'kappa':>6} {'d/s top':>8} {'d/s deep':>9} | "
          + " ".join(f"{s.split('_')[0][:8]:>9}" for s in SURFACES))
    for k in KAPPAS:
        g = df[df.kappa == k]
        cells = " ".join(f"{g[g.surface==s].n_estimable.mean():>9.2f}" for s in SURFACES)
        print(f"  {k:>6} {g.dos_top.iloc[0]:>8.2f} {g.dos_deep.iloc[0]:>9.2f} | {cells}")

    print("\nL-2  fitted scaling exponent alpha  (gate -> 0, curvature -> 2)"
          "   [median, and % of anchors where it is fittable]")
    print(f"  {'kappa':>6} | " + " ".join(f"{s[:13]:>15}" for s in SURFACES))
    for k in KAPPAS:
        g = df[df.kappa == k]
        cells = []
        for s in SURFACES:
            gg = g[g.surface == s]
            frac = gg.alpha.notna().mean()
            med = gg.alpha.median()
            cells.append(f"{med:>7.2f} ({frac:>4.0%})" if frac > 0 else f"{'--':>15}")
        print(f"  {k:>6} | " + " ".join(f"{c:>15}" for c in cells))

    print("\nL-3  A11 cost: fire rate at the TOP rung as the ladder widens")
    print(f"  {'kappa':>6} | " + " ".join(f"{s[:13]:>14}" for s in SURFACES))
    for k in KAPPAS:
        g = df[df.kappa == k]
        print(f"  {k:>6} | " + " ".join(
            f"{g[g.surface==s].fire_top.mean():>14.3f}" for s in SURFACES))
    print("\nwrote ladder_anchor_rows.csv")


if __name__ == "__main__":
    main()
