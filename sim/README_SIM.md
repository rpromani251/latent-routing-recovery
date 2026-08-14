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

## Distance estimator (13 August)

`exp_distance_estimator.py` tests the inversion of the E1(a) crossing law,
**d_hat = -sigma * Phi^-1(pi_hat)** — the first validation of the central link in
`docs/boundary_recovery_v5`. It reruns the registered 1-D known-regimes setting
(DELTA 0.30, ladder geomspace(0.02, 0.2, 3), m = 1000/rung) recording pi_hat **per
scale**; the existing `sim1d_anchors.csv` stores only a median across scales, so the
inversion was not testable from it. Orientation is query-only: the anchor's own
response says which component it sits in, and the crossing mass is the other one.

Outputs `distance_estimator_rows.csv` (3,218 anchor-by-rung rows over six models);
`fig_distance_estimator.py` builds the two-panel figure.

**Headline.** With two query-only guards (dip fires under Bonferroni, pi_hat in
[0.05, 0.50]): corr(d_hat, d_true) = **0.9958** (n = 217), median relative error
**-0.92%**, median absolute error 0.0031, tracking d = 0.012 to 0.251 and flat across a
10x noise sweep (0.0029 / 0.0031 / 0.0027 at tau = 0.005 / 0.02 / 0.05).

**Second invariance.** Across-rung CV of d_hat is 0.098 for the gated model against
0.557-1.042 for A11-violating GP nulls and 1.241 for the kink control; honest and
lengthscale-satisfying GP nulls never fire at all. At CV < 0.15, 56% of gates are
retained and 93% of confounders rejected.

**Caveat.** This runs the *naive* protocol — 1-D, ambient Gaussian probes, no tangent
frame, no trimming, no density filter. It validates the inversion, not the inversion
inside the full pipeline. The estimate is also explicitly conditional on detection:
without the dip guard, anchors beyond the ladder top return confident nonsense.

    python3 exp_distance_estimator.py     # ~4 min, writes the rows CSV
    python3 fig_distance_estimator.py     # writes fig_distance_estimator.png

Requires `diptest` for tabled p-values (falls back to the seeded MC null otherwise).

## Normal estimator -- step 2 (13 August)

`exp_normal_estimator.py` tests whether the separability classifier's decision
boundary, lifted through the tangent frame, recovers the gate's normal. Intrinsic
d = 2, ambient D = 20, frame **supplied** (Stage 0 deferred), 200 anchors per cell
over a Delta/tau x pi grid, m = 1000 probes/anchor. The true normal is drawn per
anchor so no coordinate is privileged; error is angle(n_hat, n) modulo sign.

**The question.** Responsibilities are estimated, not true branch labels, so the
chain misassignment -> classifier label noise -> bias in n_hat could dominate near
weak separation. The oracle arm (true labels) gives the variance floor; the gap to
the estimated-responsibility arm is the misassignment cost.

**Result: variance, not bias.** Per-cell mean signed rotation is -0.35 to +0.27 deg
for the oracle arm and -6.6 to +5.9 deg for the soft arm, **not significant at 95%
in any of the 20 cells**. The symmetry argument holds: responsibilities are a
function of the 1-D residual, the residual is a function of z.n, so the
class-conditional means of z stay on n however corrupted the labels. Bias survives
pooling; variance does not -- so this is the outcome that matters. Pooled
orientation error falls as 1/sqrt(N): at pi = 0.05, 6.8 -> 0.9 deg from N = 1 to 100
at Delta/tau = 2.5, and 20.4 -> 1.8 deg at Delta/tau = 1.5.

**Prediction falsified.** Orientation error is *monotone increasing* in pi, not
U-shaped. Per-anchor sd at Delta/tau = 5 runs 3.9 deg at pi = 0.05 to 47.5 deg at
pi = 0.50. Fewer crossers, but they sit further out along the normal: the separation
of class-conditional means along n is 2.17 sigma at pi = 0.05 against 1.60 sigma at
pi = 0.50. Useful anchors sit at d ~ 1.3-1.6 sigma. This *inverts* Experiment P's
distance axis, whose small values place anchors close to the boundary.

**Two design findings.** Fit the discriminant on ALL probes, not the LTS inlier set
(5.8 vs 21.9 deg median error) -- the trim removes predominantly the minority branch,
which carries the crossing signal. And soft responsibilities beat hard gamma > 0.5
labels (5.8 vs 7.6 deg).

**Scope.** The operating region is Delta/tau >= 1.5 AND pi <= 0.10. Outside it the
per-anchor sd is ~52 deg, which is what uniformly random directions give. Pooling
uninformative *axial* estimates converges to an arbitrary direction with a tightening
interval -- confidently wrong rather than abstaining. Stage 6 needs a no-information
guard; see boundary_recovery_v5 sec. 12.

**Caveat.** Intrinsic d = 2 only. The discriminant is a direction in R^d, so error
should scale with *intrinsic* d -- E1's dimension-independence result is about
*ambient* D and does not transfer. The d in {4, 8} sweep is a parameter change in
this harness.

    python3 exp_normal_estimator.py    # ~7 min
    python3 fig_normal_estimator.py

Outputs `normal_estimator_rows.csv` (4,000 anchors) and
`normal_estimator_probelog.csv.gz` (per-probe z, response, true label, fitted
responsibility, trim mask, filter mask for 2 anchors/cell) -- enough to reconstruct
any downstream arm (oracle/soft/hard labels, trim on-off, filter on-off) as
post-processing, without a rerun. The script is seeded, so a rerun reproduces it.

## Full-pipeline bootstrap -- Experiment B (14 August)

`exp_bootstrap_calibration.py` closes the open item in `boundary_recovery_v5` sec.17
("no full-pipeline bootstrap yet") and the Stage-3 calibration item in sec.10. Full
write-up with every table: **`RESULTS_bootstrap_2026-08-14.md`**.

**The registered recipe is wrong in both directions.** Regenerating responses from
resampled *inlier* residuals inflates the null LRT from a true median of 0.65 to
**67.8**, so every p-value is 1.000 -- the inlier set is the truncated middle 75% of
the residuals (kurtosis 2.11 against the noise's 2.99), and feeding that back through a
pipeline that trims again compounds the deformation. Resampling *all* residuals fails
differently: under a gate it resamples the gate's own bimodality into the null (median
11.9 at Delta/tau = 1.95). Only Gaussian regeneration works.

**And the practice it was meant to replace was never miscalibrated.** The premise is
that "LTS truncates the tails before the LRT sees them", but Stage 2 residualises ALL
points -- the trim selects the *fit*, not the *test sample*. Pipeline null q95
**5.459** against untrimmed clean-Gaussian **5.543** (B = 20,000 each); using the clean
threshold gives size 0.0476 rather than 0.050. The July figure of 6.08 is reproduced
exactly by clean draws at **m = 400**, so that gap looks like a sample-size difference,
not a trimming effect.

**The null is not a per-anchor, per-rung object.** It is invariant to rung
(between-rung sd 0.049 against within-rung 0.123), to tau, to the trend and to the
anchor: the observed per-anchor spread of the B = 300 threshold is 0.51 against 0.50
for pure Monte-Carlo resampling. Experiment A's 5.50 / 5.35 / 4.60 is one threshold
measured three times, and its deepest-rung 4.60 inflates size to 0.077. **Compute one
calibration once at high B** -- 20x more accurate than B = 300 and 300x cheaper.

**The minimum-mass rule is an anti-gate.** On a surface with no gate anywhere it
declares the rung estimable at **0.990-0.995**; where EM finds a real but tiny minority
(four crossers in 800) it *rejects*, at 0.748. The calibrated test is nominal exactly
where min-mass is 0.99, so the deepest-estimable-rung convention should be defined by
the test, not by pi_hat >= 0.05.

**The dip's floor is a function of pi.** The ~2 sd figure is the equal-weight floor.
Separation for power 0.5: 3 sd at pi = 0.50, 4 at 0.25, **6 at 0.10**, >7 at 0.05. At
the placement the method targets (pi <= 0.10) the dip is inert. Its tabled p-value is
additionally conservative through the pipeline (fires 0.000 on honest); calibrating it
restores nominal size but buys no power and raises resonant-curvature FP from 0.058 to
0.557. **Leave the dip on its tabled p-value.**

**Detection power peaks where orientation does** -- d/sigma ~ 1.3-1.6, the same shell
step 2 identified (power 0.628 at 1.28 and 0.720 at 1.54, against 0.080 at 2.56 and
0.032 at 5.13). So the pi conflict is not orientation-versus-everything-else; it is the
*ladder* versus everything else, since a geometric ladder topping out at the sweet spot
necessarily puts its lower rungs where nothing is estimable.

    python3 exp_bootstrap_calibration.py nulls     # ~15 min, 2 cores
    python3 exp_bootstrap_calibration.py anchors   # ~4 min
    python3 exp_bootstrap_calibration.py boot      # ~22 min
    python3 exp_dip_floor.py ; python3 check_null_dependence.py
    python3 verify_bootstrap.py                    # 10 checks, exit 0
    python3 analyse_bootstrap.py ; python3 fig_bootstrap.py

Outputs `bootstrap_nulls.csv` (40), `bootstrap_anchor_stats.csv` (10,800),
`bootstrap_anchor_pvalues.csv` (960), `bootstrap_reference_thresholds.csv`,
`dip_floor_rows.csv`, `fig_bootstrap.png`. All stages resumable via `_parts_*/`.

`fast_em.py` is a batched rewrite of `gmm2_equalvar` for the null replicates: the
equal-variance responsibility is `sigmoid(a + b*y)`, linear in the observation, so the
(chains x points x components) tensor collapses and `sum(y)`, `sum(y^2)` are fixed
across iterations. Verified against the reference over 60 residual shapes spanning LRT
-0.02 to 218, max relative difference 5e-10.
