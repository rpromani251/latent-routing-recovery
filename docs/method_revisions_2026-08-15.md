# What the 14–15 August runs change in `boundary_recovery_v5`

*Compiled 15 August 2026. This is the change-list, not a results note — every claim in the
method note that these three experiments moved, with what it says now and what it should
say. Full evidence in `sim/RESULTS_bootstrap_2026-08-14.md`,
`sim/RESULTS_axis_that_2026-08-14.md`, `sim/RESULTS_ladder_2026-08-15.md`.*

Section numbers refer to `boundary_recovery_v5.tex` with continuous numbering across
parts: §6 Stage 0, §7 Stage 1, §8 Stage 2, §9 Stage 3, §10 Stage 4, §11 Stage 5,
§12 Stage 6, §13 Stage 7, §14 the obligation, §15 the critical path, §17 the limits.

---

## First: what survived, unchanged

Worth stating before the change-list, because the list is long and reads worse than the
situation is. None of these moved:

- **The distance estimator.** d̂ = −σΦ⁻¹(π̂), r = 0.996 against truth, median relative
  error −0.9%, flat across a 10× noise sweep.
- **The normal estimator's unbiasedness.** No detectable bias in any of 20 cells; the
  error is pure variance, so it pools away.
- **Estimator ranking C > B > A**, with the largest gap at Δ/τ = 1.5.
- **Trimmed residualisation.** Honest false positives 0.49 → 0.00 on clumpy data.
- **Dispersion cannot discriminate a gate from curvature.** The founding identifiability
  result.
- **The calibrated Stage-3 threshold holds up.** Across the entire ladder sweep — six
  ladder positions, six surfaces — the honest surface fires at 0.028–0.052 against a
  nominal 0.05.

---

## The change-list

### §9 Stage 3 — the calibration paragraph

**Says.** *"Resample inlier residuals, regenerate responses, rerun filter → trim → refit →
residualise → both tests, B = 300 … do not reuse a threshold derived from clean Gaussian
draws: LTS truncates the tails before the LRT sees them, which deflates the null."*

**Found.** Both halves fail, in opposite directions. Resampling inlier residuals inflates
the null LRT from a true median of 0.65 to **67.8**, so every p-value is 1.000 on honest
and gated surfaces alike — the inlier set is the truncated middle 75% (kurtosis 2.11
against the noise's 2.99) and the pipeline truncates it again. And the deflation premise is
simply untrue: Stage 2 residualises **all** points, so the trim selects the *fit*, not the
*test sample*. Pipeline null q95 **5.459** against untrimmed clean-Gaussian **5.543**
(B = 20 000 each); the clean threshold has size 0.0476.

**Should say.** One **parametric Monte-Carlo null through the pipeline**, computed once at
high B — not a bootstrap, and not per anchor. The null is invariant to rung, to τ, to the
trend and to the anchor; all per-anchor variation is B = 300 Monte-Carlo noise (observed sd
0.51 against 0.50 for pure resampling). Threshold **5.459 at m = 800**. The one real
dependency is sample size: 5.62 at m = 400, 4.99 at m = 2000 — **calibrate at the deployed
m.** The July figure of 6.08 is reproduced exactly by clean draws at m = 400, so that gap
looks like a sample-size difference, not a trimming effect.

### §9 Stage 3 — the dip's floor

**Says.** *"an equal-variance two-component Gaussian mixture is literally unimodal until
separation exceeds about 2 standard deviations."*

**Found.** That is the **equal-weight** floor. Separation for power 0.5, n = 800:

| π | 0.50 | 0.35 | 0.25 | 0.10 | 0.05 |
|---|---|---|---|---|---|
| separation | 3 σ | 4 σ | 4 σ | **6 σ** | **> 7 σ** |

**Should say.** The floor is a function of π. Anchor placement targets π ≤ 0.10, so at the
method's own operating point the dip is inert below ≈ 6 σ, against a real scaffold at
1.6–1.95 σ. Its tabled p-value is additionally conservative through the pipeline (fires
0.000 on honest anchors at every rung); calibrating it restores nominal size but buys **no**
power and raises resonant-curvature false positives from 0.058 to **0.557**. **Keep the dip
on its tabled p-value, and stop describing it as primary.** The LRT is the detector.

### §10 Stage 4 — the minimum-mass rule

**Says.** π̂ ≥ 0.05 defines the "deepest estimable rung".

**Found.** It is not a gate. On a surface with **no gate anywhere** it passes at
0.990–0.995. And it is anti-correlated with the truth: where EM finds a real but tiny
minority (≈ 4 crossers in 800) it *rejects*, at 0.748.

**Should say.** **Delete it as an estimability criterion.** Estimability is defined by the
calibrated test, which is nominal exactly where min-mass is 0.99. Retain π̂ bounds only to
keep LTS away from its breakdown point.

### §11 Stage 5(a) — the Δ̂ scaling exponent

**Says.** Regress log Δ̂ on log r; r⁰ for a step, r¹ for a kink, r² for curvature.

**Found.** Fittable at **17% of gate anchors and 100% of curvature anchors** — the check
cannot be computed at the anchors you want to keep. And it stops discriminating once the
probe ball spans the curvature wavelength: resonant α collapses to ≈ 0.00, because a wide
ball sees the sine's fixed full amplitude, which is scale-free.

**Should say.** Not usable as a per-anchor filter, and not usable inside a selection rule.
At most a population-level diagnostic in the regime where the probe ball is *small relative
to the curvature wavelength* — which is the regime the ladder cannot reach and stay
estimable.

### §11 Stage 5(b) — across-rung constancy of d̂

**Not directly tested**, but it inherits the same structural problem: it needs ≥ 2
estimable rungs at one anchor, and the estimable window is a factor of 2.4 against a ladder
spanning 4 (below). **Flag for re-examination** — the CV = 0.098 figure was measured under
the 1-D naive protocol with a different geometry.

### §11 Stage 5(c) — the offset-agreement check

**Says.** The two routes to the offset *"agree by construction under a correctly specified
model"*, so a large gap indicates misspecification.

**Found.** They disagree systematically whenever π ≠ 0.5, and **the classifier route is the
biased one**: the LDA midpoint of the class means sits at 0.780 σ against a true 1.282 σ at
π = 0.10.

**Should say.** Not a specification check — one of its arms is biased. Use the crossing-law
route as the estimator and retire the comparison, or re-derive it after the repair.

### §12 Stage 6 — the pooled offset

**Says.** ĉ = Σ w·sg·(ν_a·t_a + t0_a) / Σ w.

**Found.** That uses each anchor's *own* normal against an anchor position far from the
origin, so the term carrying the threshold's magnitude is attenuated by E[cos φ]. The bias
is **multiplicative** — proportional to the true threshold — and does not shrink with N.
Measured bias/T against the predicted E[cos φ] − 1: −0.29 vs −0.26 at Δ/τ = 1.5.
**Experiment P measured this at c_true = 0, where a multiplicative bias is invisible by
construction.**

**Should say.** Project each anchor's estimated boundary **point** onto the **pooled**
normal, and take the anchor-to-boundary distance from the crossing law rather than the LDA
midpoint. Direction from separability, distance from the crossing law — each instrument
used where it is validated. Not a trade-off: coverage 0.945–0.998 and the interval is **5×
narrower**.

### §13 Stage 7 — axis dominance

**Says.** *"is one |n̂ᵢ| dominant? calibrate against the permutation distribution of
max_j |n̂ⱼ|."*

**Found.** Structurally broken. n̂ always lies in span(U), and permuting residuals *within*
an anchor leaves the frame untouched — and an axis-aligned gate that is visible at all
forces the frame to contain that axis. So the permuted normal is cos φ·u₁ + sin φ·u₂ and
the null is |cos φ|, whose q95 is sin(0.95π/2) = **0.9969**. Measured 0.86–0.999. The rule
therefore reduces to *"is the pooled orientation error below ≈ 4.5°"*, which is not a test
of axis alignment. False-claim rate on oblique gates **0.20–0.79**.

**Should say.** Axis dominance is an **equivalence** claim, not a significance claim.
Bootstrap the anchors, form a one-sided upper confidence bound on the angle between n̂ and
the candidate axis, and claim a feature only if that bound is below a stated tolerance.
False-claim rate 0.000 beyond tolerance, and the bound tracks the true tilt (0.87° / 10.14°
/ 20.13° / 30.09° / 45.12° against 0 / 10 / 20 / 30 / 45). **Report the bound, not the
binary.**

### §7 Stage 1 — "S = 3, not 2"

**Says.** Three rungs are needed because two cannot separate a gate from resonant
curvature.

**Found.** The estimable window in d/σ is bounded below by LTS breakdown (≈ 0.67) and above
by minimum detectable minority mass (≈ 1.64) — a factor of **2.4**. A three-rung geometric
ladder of ratio 2 spans a factor of **4**. **The ladder does not fit inside the window at
any position.** At most two rungs can be simultaneously estimable, and only at a ratio
below ≈ 1.55.

**Should say.** Revisit. The stated justification for S = 3 rests on the ratio test, which
§11(a) now cannot deliver at gate anchors. Either the ratio drops (untested) or the
multi-scale argument needs a different instrument.

### §14 The obligation the reframe creates

**Says.** Coverage of the interval on (n̂, ĉ, t̂) is a first-class deliverable, never
measured.

**Now.** Measured. **0.00** as specified, and *worse* with more anchors. **0.945–0.998**
after the §12 repair. Scope: t̂ only, axis-aligned gate, by-design placement, Gaussian
noise, intrinsic d = 2.

### §17 The limits

*"no full-pipeline bootstrap yet"* — closed, and the answer is that it should not be a
bootstrap.

---

## One retraction outside v5

Commit `125555a` reports that at π_top = 0.35 all three rungs are estimable and the r⁰
signature is *"recovered exactly"*. Reproducing that cell under both estimability rules:

| surface | rule | rungs | α fittable | median α |
|---|---|---|---|---|
| gated Δ/τ = 2.5 | minimum-mass | 2.89 | 100% | −0.160 |
| **honest — no gate anywhere** | minimum-mass | **3.00** | **100%** | **−0.052** |
| honest | calibrated | 0.15 | 0% | — |

An honest surface reproduces it exactly, and lands *closer* to zero than the real gate.
Min-mass passes noise; EM on noise returns a Δ̂ that scales with τ, which is independent of
the probe radius, hence α = 0. **The "exact recovery of r⁰" was measuring noise's
scale-freedom.** Withdraw it.

---

## Where the critical path stands

| step | status |
|---|---|
| 1. Validate d̂ = −σΦ⁻¹(π̂) | done, 13 Aug |
| 2. Validate the separability-derived normal | done, 13 Aug (intrinsic d = 2 only) |
| 3. Experiment P — pooling | done, 13 Aug |
| 4. Axis-aligned threshold recovery with a **calibrated interval** | **done 14 Aug, after two repairs** |
| **new.** screen → select → recover | **not started — now the binding item** |
| **new.** the separability guard | **untested, and now load-bearing** |

The guard moved onto the critical path because the ladder result removed the scaling
exponent from the selection rule. Screening on the calibrated LRT enriches curvature 5×
harder than the signal (fires at 1.000 on resonant against 0.203 on the real target), and
the guard is the only remaining instrument that distinguishes a boundary from curvature —
a real gate's mixture components are linearly separable *in probe space*, curvature's are
spatially interleaved. It has never been tested.

## Scope that should be stated plainly in the paper

At Δ/τ = 1.5 — the regime the housing scaffold sits in — **200 anchors certify the routing
direction only to ≈ 13°**, so a single-coordinate claim is not available there. Naming a
feature needs Δ/τ ≥ 2.5. This is not a defect; it is the honest reach of query-only access
against a router that hides in the noise, and stating it precisely is better than
overclaiming.
