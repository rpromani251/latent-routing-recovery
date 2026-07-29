"""
Known-regimes simulation, 1-D: recovery under noise and near regime boundaries.

TRUE MODEL (regimes known by construction)
  gated   f(x) = slope*x - DELTA * 1[x >= 0.5]        two regimes, gap DELTA
  honest  f(x) = slope*x                              matched no-gate control
  robustness nulls: honest + GP path with amplitude/lengthscale comparable to
  the gate (the hardest smooth confounders, RESULTS sec. 4), and a kink control
  (continuous, non-routed) whose flagging is DOCUMENTED DESIGNED BEHAVIOR
  (observational equivalence, v2 sec. 1.1).

AXES (Peng's two, explicitly)
  noise    tau_obs sweep {0.005 .. 0.2} (a 40x range) against DELTA = 0.3
  boundary anchors on a grid of distances to x = 0.5, scored against pi_true
           (minority probe mass across the gate) -- the corrected ground truth

PROTOCOL  naive scan, 3 log-spaced scales, Bonferroni over scales dipped,
          m_dip = m_rec = 1000 per scale (the 6e allocation), K=2 GMM recovery.

Per cell: power on detectable anchors, FP on the matched no-gate model,
Delta_hat bias and IQR, abstention rate.

Resumable: each (stage) writes its own CSV and is skipped when present.
Run `python3 sim1d_known_regimes.py` repeatedly until it prints ALL DONE,
then `python3 sim1d_known_regimes.py summarize`.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sim_core import audit_anchor, rff_gp_path, summarize
from dip import preload_null_table

SLOPE = 0.15
DELTA = 0.30
SCALES = np.geomspace(0.02, 0.2, 3)
M_DIP = M_REC = 1000
NOISES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
N_ANCHOR = 240
SEED = 20260729

# GP nulls ordered by lengthscale vs the ladder top (0.2): ell > 0.2 satisfies
# the smoothness precondition (honest model straight at every probed scale);
# ell < 0.2 VIOLATES it -- designed-failure cases documenting that the ladder
# must be capped below the honest lengthscale (Seattle: ell 2367m vs top 1200m).
GP_SETTINGS = [("gp_A.10_l.30_ok", 0.10, 0.30), ("gp_A.20_l.30_ok", 0.20, 0.30),
               ("gp_A.10_l.10_mid", 0.10, 0.10), ("gp_A.10_l.05_bad", 0.10, 0.05),
               ("gp_A.20_l.05_bad", 0.20, 0.05)]


def gated_f(X):   return SLOPE * X[:, 0] - DELTA * (X[:, 0] >= 0.5)
def honest_f(X):  return SLOPE * X[:, 0]
def gate_ind(X):  return X[:, 0] >= 0.5


def run_noise_cell(tau, model):
    out = f"s1v4_tau{tau:g}_{model}.csv"
    if os.path.exists(out):
        return False
    rng = np.random.default_rng(SEED + int(tau * 1e6) + (0 if model == "gated" else 1))
    anchors = np.linspace(0.02, 0.98, N_ANCHOR)
    f = gated_f if model == "gated" else honest_f
    rows = []
    for a in anchors:
        r = audit_anchor(f, gate_ind, [a], SCALES, rng,
                         m_dip=M_DIP, m_rec=M_REC, tau_obs=tau)
        r.update(model=model, tau_obs=tau, anchor=a, dist=abs(a - 0.5))
        rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)
    return True


def run_rob_cell(name):
    out = f"s1v3_rob_{name}.csv"
    if os.path.exists(out):
        return False
    rng = np.random.default_rng(SEED + (hash(name) % 10000))
    anchors = np.linspace(0.02, 0.98, 120)
    if name == "kink_control":
        f = lambda X: SLOPE * X[:, 0] + 1.5 * np.maximum(X[:, 0] - 0.5, 0.0)
    else:
        amp, ell = dict((n, (a, l)) for n, a, l in GP_SETTINGS)[name]
        g = rff_gp_path(1, amp, ell, 256, np.random.default_rng(hash(name) % 2**31))
        f = lambda X: SLOPE * X[:, 0] + g(X)
    rows = []
    for a in anchors:
        r = audit_anchor(f, gate_ind, [a], SCALES, rng, m_dip=M_DIP,
                         m_rec=M_REC, tau_obs=0.02)
        r.update(model=name, tau_obs=0.02, anchor=a, dist=np.nan)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)
    return True


def do_summarize():
    df = pd.concat([pd.read_csv(f"s1v4_tau{t:g}_{m}.csv")
                    for t in NOISES for m in ("gated", "honest")])
    df.to_csv("sim1d_anchors.csv", index=False)

    summ = []
    for tau in NOISES:
        s = summarize(df[df.tau_obs == tau].to_dict("records"))
        s["tau_obs"] = tau
        summ.append(s)
    sn = pd.DataFrame(summ)
    sn.to_csv("sim1d_noise_summary.csv", index=False)

    d2 = df[(df.tau_obs == 0.02) & (df.model == "gated")].copy()
    bins = [0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.51]
    d2["pi_bin"] = pd.cut(d2.pi_true_max.fillna(0.0), bins, include_lowest=True)
    bd = (d2.groupby("pi_bin", observed=False)
             .agg(n=("flag", "size"), flag_rate=("flag", "mean"),
                  delta_med=("delta_med", "median"),
                  abstain=("abstain", "mean")).reset_index())
    bd.to_csv("sim1d_boundary_summary.csv", index=False)

    dbins = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    d2["d_bin"] = pd.cut(d2.dist, dbins, include_lowest=True)
    dd = (d2.groupby("d_bin", observed=False)
             .agg(n=("flag", "size"), flag_rate=("flag", "mean"),
                  delta_med=("delta_med", "median")).reset_index())
    dd.to_csv("sim1d_distance_summary.csv", index=False)
    print("\n=== boundary axis (tau=0.02, distance bins) ===\n", dd.round(3).to_string())

    rob = pd.concat([pd.read_csv(f"s1v3_rob_{n}.csv")
                     for n, _, _ in GP_SETTINGS] +
                    [pd.read_csv("s1v3_rob_kink_control.csv")])
    rob.to_csv("sim1d_robustness_anchors.csv", index=False)
    rs = (rob.groupby("model")
             .agg(n=("flag", "size"), flag_rate=("flag", "mean"),
                  abstain=("abstain", "mean")).reset_index())
    rs.to_csv("sim1d_robustness_summary.csv", index=False)

    print("=== noise axis ===\n", sn.round(3).to_string())
    print("\n=== boundary axis (tau=0.02, pi_true bins) ===\n", bd.round(3).to_string())
    print("\n=== robustness nulls ===\n", rs.round(3).to_string())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        do_summarize()
        return
    preload_null_table(M_DIP, f"null_table_{M_DIP}.npz")
    t0 = time.time()
    for tau in NOISES:
        for model in ("gated", "honest"):
            if run_noise_cell(tau, model):
                print(f"tau={tau:g}/{model} done ({time.time()-t0:.0f}s)", flush=True)
                if time.time() - t0 > 25:
                    print("CHUNK LIMIT -- rerun to continue"); return
    for name, _, _ in GP_SETTINGS:
        if run_rob_cell(name):
            print(f"{name} done ({time.time()-t0:.0f}s)", flush=True)
            if time.time() - t0 > 25:
                print("CHUNK LIMIT -- rerun to continue"); return
    if run_rob_cell("kink_control"):
        print(f"kink done ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
