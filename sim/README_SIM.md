# Known-regimes simulation suite

Peng's ask #2 (simulation with known true regimes, evaluated under noise and
near regime boundaries) and #3 (comparison with the graph-based approach and
simple baselines). All ground truth is known by construction; the audit never
sees it.

## Protocol under test

The **simplified protocol** from `RESULTS_2026-07-28.md` §6c–6e: naive scan
over 3 log-spaced scales, Hartigan dip at each, Bonferroni over scales dipped,
per-scale minimum-signal abstention, K=2 GMM recovery on a fresh draw per
scale (soft composition π̂, recovered gap Δ̂, anchor orientation ẑ), v2
minimum-mass rule. `Δ̂ at the p_min scale` is reported alongside the
across-scale median — the median dilutes Δ̂ with non-crossing scales.

## Files

| file | role |
|---|---|
| `dip.py` | dip statistic + p-value; uses `diptest` if installed, else seeded Monte-Carlo against the uniform null (identical calibration logic) |
| `sim_core.py` | audit primitive, 1-D K=2 EM (no sklearn needed), GP paths, metrics |
| `sim1d_known_regimes.py` | 1-D: noise sweep × boundary axis + robustness nulls |
| `sim2d_known_regimes.py` | 2-D wavy spatial gate (Seattle geometry class) |
| `sim_baselines.py` | global clustering, residual-outlier, SPA profile-mean |
| `sim_sbm_baseline.py` | the April pEx-SBM/behavioural-fingerprint thread, run via `geospatial-xai-attacks/src/sbm` on the same models |
| `fig_sim_results.py` | builds `fig_sim_known_regimes.png`, `fig_sim2d_map.png` |
| `run_cells.py` | parallel/resumable driver for sim1d cells |

Settings registered in each file header: Δ=0.30, ladder geomspace(0.02, 0.2, 3),
m_dip = m_rec = 1000, α=0.05, seed 20260729. Honest models satisfy the
measured Seattle ordering (smooth amplitude ≪ Δ; lengthscale > ladder top).

## Headline results

**Noise axis (1-D, Δ/τ from 60× to 1.5×):** power on detectable anchors
0.98 → 0.89 → 0.57 → 0.01 as τ_obs goes 0.005 → 0.02 → 0.05 → 0.1; FP on the
matched no-gate model ≤ 0.004 at every noise level; the honest model's
abstention rate rises to 0.79 at τ=0.2 (min-signal rule working as designed);
Δ̂ at the p_min scale stays 0.248–0.288 against a true 0.30 wherever anything
is flagged.

**Boundary axis:** power is 1.00 for π_true > 0.1, 0.55–0.71 in the
0.05–0.10 band, 0 below 0.05 — detection is limited by minor-mode mass m·π,
reproducing the Seattle finding in a controlled setting. Flag rate vs distance
falls to ~0 beyond the ladder top (detection radius = probe reach).

**Robustness (the new, sharp precondition):** GP nulls with lengthscale
*above* the ladder top flag at 0.00–0.04; lengthscale *below* it flags at
0.33–0.81. Dipping at scales beyond the honest lengthscale reads smooth
wiggles as modes. The precondition is checkable (cap the ladder below the
honest lengthscale — Seattle: 2367 m vs 1200 m) and its violation is loud,
not silent. The kink control (sharp but continuous, non-routed) flags at
0.05 = nominal: the dip targets jumps, not corners — sharper specificity than
the v2 discussion assumed.

**2-D spatial gate:** power 0.93 on detectable anchors, FP 0.000, Δ̂ 0.27
(0.297 within 0.02 of the boundary), orientation accuracy 0.956; flags trace
the wavy boundary (see `fig_sim2d_map.png`).

**Baselines (`baseline_comparison.csv`):**

| method | exist. FP (honest) | partition acc (all / detectable) | Δ̂ err | queries/anchor |
|---|---|---|---|---|
| dip-scan (ours) | 0.000 | 0.80 / 0.93 | 0.03 | 6001 |
| global-cluster (BIC) | 0.00 | 0.54 / 0.54 | 0.30 | 25 |
| residual-outlier | 0.008 | 0.52 / 0.52 | — | 25 |
| SPA profile-mean | no test exists | 0.52 / 0.58 | 0.21 | 50 |
| plain SBM K=2 (their code) | no test exists | 0.61 / — | — | 33 |

Global clustering does answer *existence* here (BIC picks K=2 on gated, K=1 on
honest) but its partition is chance-level: detrending absorbs most of the gate
and the residual split follows covariates, not mechanism. SPA and the SBM
return structure unconditionally (no calibrated existence claim); the SBM's
partition aligns only weakly with the true regime (ARI 0.05 gated, 0.00
honest). Note the SBM fingerprint subtracts the anchor's own response, which
cancels the constant penalty — level effects are invisible to it except across
boundaries; this is intrinsic to similarity-graph methods, not a tuning issue.

## Reproduce

    python3 run_cells.py            # repeat until ALL DONE (resumable)
    python3 sim1d_known_regimes.py summarize
    python3 sim2d_known_regimes.py  # then: summarize
    python3 sim_baselines.py
    python3 sim_sbm_baseline.py     # needs geospatial-xai-attacks on sys.path
    python3 fig_sim_results.py

Dependencies: numpy, pandas, matplotlib. Optional: scipy, diptest, and
scikit-learn + the repo for the SBM baseline.

## Extension experiments (E1–E4, July 29)

Future-work items run as controlled experiments with pre-stated propositions;
full write-up in `future_work_experiment.pdf` (vault root and
`docs/routing_audit/` in the repo). Files: `ext_core.py`, `ext1_fullvec.py`
(full input-vector probing), `ext2_kgt2.py` (K>2), `ext3_vector_output.py`
(non-scalar outputs), `ext4_nongeo.py` (clumpy manifolds / A12), driver
`run_ext.py`, figure `fig_future_work.png`.

Headlines: crossing-rate law Φ(−d√D/σ) verified to 3 decimals; plain OLS
residualization absorbs ~64% of the gap (closed form) — trimmed residualization
repairs it and makes full-vector probing D-independent (0.74 at D=16), fixes
clumpy-manifold FP 0.49→0.00 (power 0.57), and unmasks strong trends
(0.14→0.90); BIC + effect-size merge rule recovers K̂ at 0.75 with 99.7%
global label matching; vector outputs need any projection with weight
orthogonal to the within-branch variation, and intercept-at-anchor recovery
returns the penalty vector at cos 1.000.
