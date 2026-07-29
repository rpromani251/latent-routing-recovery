"""
Graph-based (SBM) baseline on the 2-D known-regimes simulation.

Runs the April behavioural-fingerprint SBM thread (geospatial-xai-attacks
src/sbm/) like-for-like on the same gated/honest models as sim2d:

  fingerprint  E[i] = f(x_i + delta_j) - f(x_i) over a fixed shared battery of
               offsets (two rings, radii 0.05 and 0.10), 32 queries/anchor
  graph        kNN cosine-similarity graph (k = 8), binarised (their default)
  model        Bernoulli SBM, K = 2, collapsed Gibbs (their fit_sbm)
  partition    modal post-burn-in assignment; scored vs true gate side by
               accuracy-up-to-permutation and ARI/NMI (their compare module)

The SBM returns a partition unconditionally, so there is no calibrated
existence claim; on the honest model we report how strongly its partition
spuriously aligns with the (nonexistent) gate.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The SBM implementation lives in the sibling geospatial-xai-attacks repo (this is a
# like-for-like comparison against *their* code, deliberately not a reimplementation).
# Defaults to the sibling checkout; override with DRI_GEOXAI_REPO.
REPO = os.environ.get(
    "DRI_GEOXAI_REPO",
    str(Path(__file__).resolve().parents[2] / "geospatial-xai-attacks"),
)
if not (Path(REPO) / "src" / "sbm").is_dir():
    raise SystemExit(
        f"Could not find src/sbm under {REPO}.\n"
        "This baseline runs the SBM thread from the geospatial-xai-attacks repo. "
        "Clone it beside this one, or set DRI_GEOXAI_REPO to its path."
    )
sys.path.insert(0, REPO)

from src.sbm import build_knn_similarity_graph, fit_sbm, partition_ari, partition_nmi
from sim2d_known_regimes import anchors_all, honest_f, gated_f, gate_ind, TAU_OBS, SEED

N_BATTERY = 32
RADII = (0.05, 0.10)
K_GRAPH = 8
K_BLOCKS = 2


def fingerprints(f, A, rng):
    th = np.linspace(0, 2 * np.pi, N_BATTERY // len(RADII), endpoint=False)
    offs = np.concatenate([np.column_stack([r * np.cos(th), r * np.sin(th)])
                           for r in RADII])
    E = np.empty((len(A), len(offs)))
    for i, x0 in enumerate(A):
        base = f(x0[None, :])[0]
        E[i] = f(x0[None, :] + offs) - base + rng.normal(0, TAU_OBS, len(offs))
    return E


def modal_partition(sbm):
    tr = np.asarray(sbm.z_trace)
    z = np.empty(tr.shape[1], int)
    for i in range(tr.shape[1]):
        z[i] = np.bincount(tr[:, i]).argmax()
    return z


def acc_perm(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    a = (pred == truth).mean()
    return float(max(a, 1 - a))


def main():
    rng = np.random.default_rng(SEED + 77)
    A = anchors_all()
    truth = gate_ind(A).astype(int)

    rows = []
    for model, f in (("gated", gated_f), ("honest", honest_f)):
        E = fingerprints(f, A, rng)
        G = build_knn_similarity_graph(E, k=K_GRAPH)
        sbm = fit_sbm(G, K=K_BLOCKS, n_iter=400, burn_in=150, thin=5,
                      seed=0, verbose=False)
        z = modal_partition(sbm)
        rows.append(dict(model=model,
                         acc_perm=acc_perm(z.astype(bool), truth.astype(bool)),
                         ari=partition_ari(z, truth),
                         nmi=partition_nmi(z, truth),
                         queries_per_anchor=N_BATTERY + 1))
        print(f"{model}: acc={rows[-1]['acc_perm']:.3f} "
              f"ari={rows[-1]['ari']:.3f} nmi={rows[-1]['nmi']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("sbm_baseline_result.csv", index=False)

    # upsert a comparison row in the common format (idempotent: rerunning this script
    # must not append a duplicate SBM row to the table)
    bc = pd.read_csv("baseline_comparison.csv")
    g = out[out.model == "gated"].iloc[0]
    h = out[out.model == "honest"].iloc[0]
    method = "pEx-SBM graph (plain SBM, K=2)"
    row = dict(method=method,
               exist_honest=np.nan, exist_gated=np.nan,
               part_acc_all=float(g.acc_perm),
               part_acc_detectable=np.nan,
               delta_err=np.nan,
               queries_per_anchor=N_BATTERY + 1,
               delta_honest_spurious=np.nan)
    bc = bc[bc["method"] != method]
    bc = pd.concat([bc, pd.DataFrame([row])], ignore_index=True)
    bc.to_csv("baseline_comparison.csv", index=False)
    print("\nupserted SBM row into baseline_comparison.csv")
    print(f"honest-model spurious alignment: ari={h.ari:.3f}")


if __name__ == "__main__":
    main()
