# Experiment B — the full-pipeline bootstrap (B = 300)

**Run 14 August 2026.** Closes the open item in `boundary_recovery_v5` §17 ("no
full-pipeline bootstrap yet") and the calibration item in §9 (Stage 3).

The registered recipe was:

> Resample inlier residuals, regenerate responses, rerun filter → trim → refit →
> residualise → both tests, B = 300. Trimming is a nonlinear, data-dependent operator,
> so anything calibrated against untrimmed theory is miscalibrated. In particular, do
> not reuse a threshold derived from clean Gaussian draws: LTS truncates the tails
> before the LRT sees them, which deflates the null.

**Both halves of that are wrong, in opposite directions.** The prescribed resampling
scheme does not produce a null at all, and the practice it was meant to replace was
never miscalibrated. Four further results came out of the same harness, three of which
bear on the π conflict and on the estimability gap carried in the Experiment A commit.

---

## Summary

| # | Claim | Outcome |
|---|---|---|
| B-1 | Resampling **inlier** residuals calibrates the pipeline | **FALSIFIED.** Null LRT median 67.8 against a true 0.65. Every p-value is 1.000. |
| B-2 | Resampling **all** residuals is the safe alternative | **FALSIFIED.** Contaminated by the gate under H₁ (null median 11.9 at Δ/τ = 1.95). Fire rate 0.000. |
| B-3 | Trimming deflates the null, so clean-Gaussian thresholds are unusable | **FALSIFIED.** Pipeline q95 **5.459** against clean-Gaussian **5.543** (B = 20 000 each). The clean threshold has size 0.0476. |
| B-4 | The null is a per-anchor, per-rung object | **FALSIFIED.** Invariant to rung, τ, trend and anchor. All per-anchor variation is B = 300 Monte-Carlo noise. |
| B-5 | The minimum-mass rule gates estimability | **FALSIFIED, and it is an anti-gate.** Passes at 0.990–0.995 on pure noise; fails when EM finds the real minority. |
| B-6 | The dip's floor is ≈ 2 σ | **FALSIFIED as stated.** That is the equal-weight floor. At π = 0.10 it is ≈ 6 σ. |
| B-7 | Detection and orientation want opposite ends of the π axis | **NOT SUPPORTED.** Both peak at d/σ ≈ 1.3–1.6. The ladder is what conflicts. |

---

## 1. The registered recipe does not produce a null

Regenerating responses from resampled **inlier** residuals inflates the null LRT by two
orders of magnitude.

| null-generating arm | null LRT median | q95 |
|---|---|---|
| Gaussian regeneration (`param`) | 0.65 | 5.39 |
| resample **all** residuals (`emp_all`) | 1.08 | 9.99 |
| resample **inlier** residuals (`emp_inlier`) | **67.79** | **91.65** |
| clean Gaussian, untrimmed (the discouraged reference) | 0.70 | 5.54 |

*Medians over 9 conditions (3 rungs × 3 replicates), honest surface, m = 800.*

The mechanism is a **shape** deformation, not a scale one. The inlier set is the
truncated middle 75 % of the residuals, so it is platykurtic; feeding it back through a
pipeline that trims again compounds the deformation, and a flat-topped sample is fitted
far better by an equal-variance two-component mixture than by one Gaussian.

| sample | kurtosis |
|---|---|
| true noise | 2.99 |
| pipeline residuals, all points | 3.14 |
| pipeline residuals, inlier set | **2.11** |
| resampled inliers after one more pipeline pass | 2.14 |
| …and *its* inlier set | **1.92** |

Consequence: with `emp_inlier`, **every** anchor returns p = 1.000, on honest and gated
surfaces alike (0/60 fire at each rung on both). `emp_all` fails differently — under a
gate it resamples the gate's own bimodality into the null (median 11.91 at
Δ/τ = 1.95, against 1.08 on honest), and also fires at 0.000.

Only Gaussian regeneration works. The scale of the regenerated noise is irrelevant: LTS
keeps the same points under a common rescaling, so residuals are scale-equivariant and
both tests are scale-invariant (`verify_bootstrap.py` [2]). The familiar worry that
trimming deflates the resampled variance therefore never bites — **but the shape
deformation it hides does.**

## 2. And the thing it was meant to fix was never broken

The premise is that "LTS truncates the tails before the LRT sees them". It does not.
Stage 2 residualises **all** points against the refit plane — the trim selects the
*fit*, not the *test sample* — so the LRT sees an essentially untouched sample with
three degrees of freedom removed from 800.

| null (B = 20 000) | median | q95 | q99 |
|---|---|---|---|
| through the pipeline | 0.657 | **5.459** | 8.622 |
| clean Gaussian, untrimmed | 0.702 | **5.543** | 8.905 |

Using the clean-Gaussian q95 as a threshold on the pipeline's own null fires at
**0.0476** instead of 0.050. That is a 5 % relative error in the level, well inside the
Monte-Carlo noise of any B = 300 calibration.

The two agree at every sample size tested (`check_null_dependence.py`), and the null
drifts weakly with **m** — which is the dependency that does exist:

| m | pipeline q95 | clean q95 |
|---|---|---|
| 400 | 5.62 | **6.08** |
| 800 | 5.33 | 5.43 |
| 1000 | 5.43 | 5.51 |
| 2000 | 4.99 | 4.91 |

The July figure of 6.08 is reproduced exactly by clean Gaussian draws at **m = 400**.
That is consistent with the 6.08-vs-5.4 gap being a sample-size difference rather than
the trimming effect it was attributed to — *suggestive, not established*, since the
July run's m was not recorded here. **Calibrate at the deployed m.**

## 3. The null is not per-rung, and Experiment A's three thresholds are one threshold

| rung | rep 0 | rep 1 | rep 2 |
|---|---|---|---|
| 1.0 σ | 5.59 | 5.48 | 5.12 |
| 0.5 σ | 5.38 | 5.53 | 5.42 |
| 0.25 σ | 5.30 | 5.39 | 5.35 |

Between-rung sd of q95 **0.049**; within-rung (replicate) sd **0.123**. There is no rung
structure. The null is also invariant to a 10× increase and 100× decrease in τ, to a
100× trend and to no trend at all (q95 5.36 / 5.23 / 5.09 / 5.58) — as it must be, since
the residual is invariant to anything in the span of [1, Z] (`verify_bootstrap.py` [3]).

Experiment A (commit `125555a`) reported **5.50 / 5.35 / 4.60**, a spread of 0.90 from
B = 300 calibration anchors per rung. That spread is Monte-Carlo noise, and it is not
free: A's deepest-rung threshold of 4.60 inflates size to **0.077** on honest anchors
against a nominal 0.05.

## 4. The per-anchor bootstrap adds nothing but its own noise

| | sd of the per-anchor q95 |
|---|---|
| observed across 100 honest anchors | **0.51** |
| pure B = 300 resampling of the reference null | **0.50** |

They match. **All** anchor-to-anchor variation in the bootstrap threshold is Monte-Carlo
noise; there is no anchor structure to condition on. And B = 300 is noisy — its q95 has
a 2.5–97.5 range of **4.47 to 6.51** around a true 5.46, ±20 %.

So the correct object is **one Monte-Carlo calibration through the pipeline, computed
once at high B** — which is both ~20× more accurate than B = 300 and 300× cheaper than
running it at every anchor.

**Scope.** The null is anchor-free *here* because the density filter is a no-op in this
flat setting and the honest surface is exactly linear. Under a real filter the retained
probe geometry varies by anchor, and conditioning could matter. Note also what the
bootstrap structurally cannot do: it regenerates noise around the **fitted plane**, so
it bakes in the linear model and can never detect that the surface is curved. Curvature
rejection stays with the Stage-5 ladder.

**Level yes, p-value no.** The level is exact (fire rate 0.050 at α = 0.05 over 300
honest anchor-rungs), and p-values are uniform below ≈ 0.8 — but 16.7 % pile up at
exactly 1.000, because the equal-variance LRT has an atom at zero under H₀ (19.9 % of
honest anchors return LRT < 0.01, matching the null's 19.9 %). That is the usual
boundary-of-parameter-space behaviour of a mixture LRT, not a calibration failure. Use
these as tests at a fixed level, not as evidence measures.

## 5. The minimum-mass rule is an anti-gate

Carried from `125555a`: *the minimum-mass rule does not enforce estimability — EM splits
noise and returns π̂ above the floor at rungs where the gate is not reached.* Measured:

| surface | rung | d/σₛ | true π | **π̂ ≥ 0.05 passes** | calibrated test fires |
|---|---|---|---|---|---|
| honest | 1.0 | ∞ | 0 | **0.993** | 0.040 |
| honest | 0.5 | ∞ | 0 | **0.995** | 0.065 |
| honest | 0.25 | ∞ | 0 | **0.990** | 0.045 |
| gated Δ/τ=2.5 | 0.25 | 5.13 | 0.00000 | **0.995** | 0.032 |
| gated Δ/τ=1.95 | 0.25 | 5.13 | 0.00000 | **0.998** | 0.037 |

On a surface with **no gate anywhere**, the minimum-mass rule declares the rung
estimable 99 % of the time. It is not a gate; it is very nearly a constant.

Worse, it is anti-correlated with the truth. At gated Δ/τ = 5.0, rung 0.5 (d/σ = 2.56,
true π = 0.0052 — about four crossers in 800), it passes only **0.748**: EM finds the
real, tiny minority, π̂ lands below 0.05, and the rule rejects. **The rule fails when EM
is right and passes when EM splits noise.**

The calibrated test is nominal exactly where min-mass is 0.99. So the deepest-estimable-
rung convention should be defined by *the calibrated per-rung test*, not by π̂ ≥ 0.05 —
and then it does self-enforce.

## 6. The dip's floor is a function of π, and it is much worse than stated

§9 (Stage 3) gives the floor as a property of the mixture: "literally unimodal until separation
exceeds about 2 standard deviations". That is the **equal-weight** floor. Separation at
which the pipeline-calibrated dip first reaches power 0.5 (n = 800):

| π | 0.50 | 0.35 | 0.25 | 0.10 | 0.05 |
|---|---|---|---|---|---|
| separation | 3 σ | 4 σ | 4 σ | **6 σ** | **> 7 σ** |

The method's own anchor placement targets π ≤ 0.10 (step 2: orientation error is
monotone increasing in π). **At its own operating point the dip is inert below ≈ 6 σ** —
three times the stated floor, and the real scaffold sits at 1.6–1.95 σ. This is the
structural reason behind `125555a`'s "the dip contributes nothing at this separation";
dip-OR-LRT ≡ LRT is not a coincidence of that separation but a property of low π.

Two further notes. The dip's **tabled** p-value is additionally conservative through the
pipeline: it fires at **0.000** on honest anchors at every rung, against a nominal 0.05.
Calibrating the dip on the pipeline null restores nominal size (0.048–0.060) — but buys
no power (0.033–0.062 on every gated cell) and **costs confounder false positives**: on
resonant curvature at 0.5 L the calibrated dip fires at **0.557** against the tabled
0.058. The dip's conservatism was doing useful work. **Recommendation: leave the dip on
its tabled p-value.**

## 7. Detection power peaks where orientation does — the conflict is the ladder

Power of the calibrated LRT against distance, pooling the gated cells:

| d/σₛ | 0.39 | 0.77 | **1.28** | **1.54** | 2.56 | 5.13 |
|---|---|---|---|---|---|---|
| π | 0.35 | 0.22 | 0.10 | 0.062 | 0.005 | ~0 |
| power (Δ/τ = 2.5) | 0.158 | 0.200 | **0.628** | **0.720** | 0.080 | 0.032 |

Power peaks at d/σ ≈ 1.3–1.6 — **the same shell step 2 identified as
orientation-useful** (d/σ ∈ [1.28, 1.64], π ∈ [0.05, 0.10]), and the same region the
distance estimator prefers. Detection, distance and orientation all want one rung, and
it is the same rung.

So the π conflict recorded in `125555a` is not *orientation versus everything else*. It
is **the ladder versus everything else**: a geometric ladder whose top rung sits at the
sweet spot necessarily puts its lower rungs at d/σ = 2.56 and 5.13, where nothing is
estimable and power is at size. Run the same ladder at π_top = 0.35 and the ordering
inverts — the *deepest* rung becomes the good one (power 0.720 at d/σ = 1.54) while the
top rung is at 0.158, because at π = 0.35 the top rung is too close to the boundary for
LTS to have a majority branch.

**The design variable is where the sweet spot sits in the ladder, not which π to
prefer.** A ladder that reaches the shell at its *deepest* rung — i.e. anchored so the
ladder climbs *away* from the boundary — keeps every rung estimable and puts the useful
rung where the invariance check can still use the ones above it. That is a change to
Experiment A's rule, and it has not been run.

Power at the real target for reference: Δ/τ = 1.95 at d/σ = 1.28 gives **0.203**;
Δ/τ = 1.5 gives 0.065. Both far below the 0.42–0.54 that §9 quotes at 1.5 σ — because
that figure is at π = 0.25, and anchor placement now targets π ≤ 0.10. **Pooling is not
optional at the real target; it is the whole detector.**

## 8. Confounder false positives, for the record

Per-anchor fire rate of the calibrated LRT on A11-violating resonant curvature
(amplitude 2.5 τ, wavelength in units of the probe diameter L):

| surface | rung 1.0 | rung 0.5 | rung 0.25 |
|---|---|---|---|
| resonant 0.5 L | **1.000** | 0.935 | 0.752 |
| resonant 1.0 L | 0.897 | 0.735 | 0.130 |
| resonant 1.5 L | 0.795 | 0.588 | 0.050 |

Unchanged by calibration (the clean threshold gives the same to three decimals). These
are the input to the Stage-5 ladder, not a verdict — but they confirm that per-anchor
confound rejection cannot come from the test's threshold at any calibration.

---

## Reproduction

All seeded, all under `sim/`. Total ≈ 70 min on 2 cores.

| stage | command | output |
|---|---|---|
| pipeline nulls + invariance | `python3 exp_bootstrap_calibration.py nulls` | `bootstrap_nulls.csv` (40), `bootstrap_null_reference.npz` |
| observed statistics grid | `python3 exp_bootstrap_calibration.py anchors` | `bootstrap_anchor_stats.csv` (10 800) |
| per-anchor B = 300 bootstrap | `python3 exp_bootstrap_calibration.py boot` | `bootstrap_anchor_pvalues.csv` (960) |
| the dip floor | `python3 exp_dip_floor.py` | `dip_floor_rows.csv` |
| null vs sample size | `python3 check_null_dependence.py` | stdout |
| verification (10 checks) | `python3 verify_bootstrap.py` | stdout, exit 0 |
| tables | `python3 analyse_bootstrap.py` | stdout |
| figure | `python3 fig_bootstrap.py` | `fig_bootstrap.png` |

Every stage is resumable — per-unit CSVs under `_parts_*/`, skipped if present.

**`fast_em.py`** is a batched rewrite of `gmm2_equalvar` for the null replicates. The
equal-variance two-component responsibility is `sigmoid(a + b·y)` — linear in the
observation — so the (chains × points × components) tensor collapses to a handful of
(chains × points) passes, and `Σy`, `Σy²` are fixed across iterations. Same
initialisation, update order, convergence test and best-of-inits rule, including the
reference's `+1e-12` on the component counts. Verified against the reference over 60
residual shapes spanning LRT −0.02 to 218: **max relative difference 5 × 10⁻¹⁰**.
Blocked at 250 replicates (measured: B = 6000 costs 140 s unblocked, 40 s blocked;
identical results at every block size).

### Notes for the repo

- `exp_a_invariance.py` and `exp_p_pooling.py` load and write through hardcoded
  `/tmp/pexp/`, `/tmp/dist_est/`, `/tmp/aexp/` paths, so neither runs from a fresh
  checkout. Experiment B uses repo-relative paths throughout; the two older scripts
  still need the same fix.
- `exp_p_pooling.py:313` computes `abs(ch - (c_true + np.mean(cv[i]) - np.mean(cv[i])))`
  — the two means cancel, so it is `abs(ch - c_true)`. Harmless as written (c_true = 0),
  but it looks like a half-applied centring correction and should be resolved either way.
- `gmm2_equalvar`'s convergence test is `ll - ll_old < tol·max(1, |ll|)`. Rescaling the
  residuals shifts `ll` by −n log k, so the stopping point drifts and the LRT is not
  *exactly* scale-invariant in this implementation: relative deviation 1.6 × 10⁻³ over a
  factor of 2, 3 × 10⁻² over 14 orders of magnitude. Negligible against a threshold of
  5.46, but an absolute tolerance would remove it.

## What this did not establish

- Intrinsic **d = 2** only, frame supplied, Stage 0 deferred, density filter a no-op —
  so "full pipeline" here means trim → refit → residualise → both tests. The filter
  branch is uncalibrated.
- The **separability guard** has no calibrated null either. It is the check that covers
  the LRT's robustness hole on curvature, and §8 shows curvature fires the LRT at up to
  1.000, so the guard is now the binding component and its calibration is untested.
- Gaussian noise throughout. The parametric calibration assumes it; the nonparametric
  alternatives that would not are the ones this experiment ruled out, so
  **heteroscedastic or heavy-tailed noise has no calibration route yet.**
- The proposed ladder re-anchoring (§7) is a recommendation from measured power, not a
  run.
