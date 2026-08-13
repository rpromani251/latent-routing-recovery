# DRI Framework — Detection Phase: Consolidated Working Document

**Date:** July 31, 2026
**Status:** Post-review synthesis. Consolidates the independent experimental review of the probe-policy spec with subsequent analysis of power limits, top-1 MoE structure, and the pooling/target-characterization experiments.

---

## 0. Scope and target

**DRI = Detect → Recover → Interpret.** Current work is entirely in the **D** phase.

- **D (now):** per-anchor detection of a local behavioral discontinuity, query-only, label-free.
- **R (next):** recovery of the global partition — estimating global K and the boundary geometry.
- **I (later):** interpretation of what distinguishes the recovered regimes.

**Ultimate target:** the Slack et al. (AIES 2020) scaffolding attack against LIME/SHAP. The attack routes between a biased model `f` and an innocuous model `ψ` via an OOD detector trained to recognize LIME-style perturbations. Detecting and then recovering that routing regime is the paper's objective.

**Standing assumption (settled):** **top-1 hard routing.** Top-k / softmax-weighted routing is out of scope. See §4 for why this matters and what it buys.

---

## 1. What was implemented and tested

Independent reimplementation of the probe-policy spec — Stage 0/1 plus minimal Stage 2/3 — run against trained sklearn MLPs (64,64, early stopping) audited query-only.

Components rebuilt from scratch: GRIDE intrinsic-dimension estimation, plateau detection, tangent-frame probing, density filter, LTS (concentration-step approximation), Hartigan dip test (`diptest` p-values), equal-variance 2-GMM likelihood-ratio test (EM, simulated null threshold).

**Statistical caveat carried throughout:** 48–60 anchor-trials per cell, binomial se ≈ 0.03–0.06, no full-pipeline bootstrap anywhere. **Contrasts between cells are the reliable content; absolute error rates are approximate.** Contrasts of 0.6 vs 0.08 are solid; contrasts of 0.10 vs 0.05 are suggestive only.

---

## 2. Findings

### 2.1 What survives

- **Stage-0-reads-only-A ⇒ selection-free adaptivity.** Clean and correct. Sharpening: independence is required from the *responses*, not from `f`'s training set — `f`'s training data may overlap the anchor set without harm.
- **The abstention taxonomy.** Converts silent failures into declared, pre-query ones. Demonstrably saves budget.
- **The two-senses-of-on-manifold distinction.** Exactly the right diagnosis of the E4 failure.
- **Smooth-isotropic-in-frame probing.** Held honest FP at 0 on clumpy data where E4 reached 0.49.
- **ρ-alongside-π̂ claim scoping**, and **σ = r/√d̂** as the right reconciliation.
- **V1's prediction, confirmed outright** — with a mechanism correction (§2.3).

### 2.2 Mechanical defects in the decision rules

| # | Defect | Fix |
|---|---|---|
| 1 | Noise skirt extends to ~3–4·τ√D, not ~τ. At τ=0.02, D=20, an honest flat d=2 patch has **no plateau anywhere** in the accessible rank range even at k=300. A1 fires for the right outcome, wrong reason. | Split `A1` (noise runs into structure) from `A1n` (ladder never exits the skirt). Diagnose query-free via `r[k_max] < c·τ̂√D`, c ≈ 3. Different remedies: A1 is a property of the anchor, A1n of the budget. |
| 2 | The CI-common-intersection plateau rule is **anti-monotone in data quality**. At τ=0.002 the curve is flat to ±0.08 over four rungs and the rule still returns A1, because tight CIs break the common intersection. Conflates "statistically indistinguishable" with "geometrically flat." | Round-consistency: longest contiguous run of rungs whose CIs intersect (j−½, j+½) for a common integer j. Used for all downstream work here. |
| 3 | W_min in radius-decades is **unachievable by construction** for d≥5, since r ~ k^(1/d). Ranks 5–200 span 1.6 rank-decades → max W ≈ 0.8 at d=2 but ≈ 0.32 at d=5, below W_min=0.35. Confirmed: a clean d=5 manifold in D=50 returns A1. | Specify W in rank-decades (dimension-free), W_min ≈ 0.6, equivalently W_min(radius) = 0.6/d̂. Add a `censored` flag for plateaus running to the last rung. |
| 4 | **The dip test is far weaker than assumed.** Power 0.000 at sep=1.5σ at every π; ≤0.04 at sep=2.5σ (n=750, 250 reps/cell). Earlier pipeline detections only worked because the planted gate was 7.5σ. | Adopt the equal-variance 2-GMM LRT alongside the dip. State plainly that dip-only limits detectable Δ to ≳3σ at m ≈ 10³. |

### 2.3 V1 confirmed, mechanism corrected

Clumpy latent d=2 (six clusters, spacing L=1, s_c ∈ {0.05, 0.10, 0.20}, D=20): **100% of successful anchors got r_hi < L**, median r_hi = 0.05–0.15 — an order of magnitude below the E4 failure radii. Retention corroborates: median ρ at r_hi = 0.67–0.83, at 0.5L = 0.005–0.37, at L = 0.00–0.15. Two independent pre-query guards exclude the FP-producing radii.

But the mechanism is **not** "d̂ drifts up when the ball swallows a second cluster." At deep ranks the curve **collapses toward zero first** (2.05 → 1.65 → 0.47): once a rank exhausts the finite cluster, neighbor distances stop growing with rank, μ → 1, and GRIDE reads d → 0. Contamination arrives later, after the plateau has already ended. The three-regime table's "large scale → d̂ drifts (usually up)" should read **non-monotone: exhaustion collapse, then contamination jump.**

Two costs: at s_c=0.05, τ=0.005 only **5/25 anchors reach "ok"** (14 A1n, 6 A0) — arguably correct abstention, but 20% throughput belongs in any writeup. And d̂ overshoots to 3 on tight clusters, so the frame carries one junk direction (harmless for validity, mildly dilutive for power).

### 2.4 The central §3.3 claim is falsified

§3.3 claims plateau-valid = A11-valid, "the same geometric condition read two different ways." **It is not, in either direction.**

**(a) Extrinsic curvature — the plateau is blind, and here it doesn't matter.** GRIDE estimates *intrinsic* dimension, invariant to smooth extrinsic bending. On a κ=8 paraboloid (tangent-plane departure 0.09–0.13 at r_hi, comparable to the probe radius) the d̂ curve barely moves. Honest FP = 0.00 at r_hi **with the density filter on and off** (n=24–25): off-support excursions land in the ReLU MLP's extrapolation region, which is locally linear — benign inductive bias. The filter's load-bearing role is claim scoping and defense against model classes with wilder off-support behavior, not FP control here. **Do not generalize this beyond piecewise-linear model classes.**

**(b) Response curvature — the real blind spot, with resonance structure.** Flat manifold, MLP trained on y = β·u + 0.5·sin(2πu₁/ℓ). Stage 0 is identical across ℓ (it never queries f), so r_hi ≈ 0.18 everywhere. Honest FP of the dip at r_hi:

| ℓ | 4.0 | 0.6 | **0.3** | 0.15 |
|---|---|---|---|---|
| FP | 0.000 | 0.017 | **0.136–0.169** | 0.000 |
| fraction of sine the MLP learned | 100% | 94% | 80% | **0%** |

A **resonance band**: roughly one oscillation across the probe ball produces residuals that are both marginally bimodal and near-linearly-separable in z — observationally gate-like. The band closes above because LTS+linear absorbs slow curvature, and below because **the model class's spectral bias closes it** — at ℓ=0.15 the black box is genuinely smooth and there is nothing to false-alarm on.

**Reframing:** the relevant "honest surface" is the *trained model's*, and its roughness is bounded below by the training procedure, not by the data-generating truth. The exposure window is **[smallest scale the model class actually learns, probe diameter]**.

**(c) The masking side of the same coin.** With a genuine Δ=0.15 gate *plus* the resonant surface, dip power collapses from 0.68 to **0.085**. Resonant smooth structure both inflates FP and hides real gates. For the adversarial paper this is a new attack surface: the scaffold can compose with a *legal* smooth masking term at the auditor's probe scale, no OOD detector needed. Static, so within declared scope.

### 2.5 Repairs

**Rejected — quadratic lack-of-fit gate with radius shrinkage.** Repairs the resonant FP (0.136 → 0.034) but is **uncalibratable**: effect size η² = 1 − RSS₂/RSS₁ of benign trained-model wiggle (med 0.023, q90 0.151) is statistically indistinguishable from FP-inducing resonant curvature (med 0.036, q90 0.174), and real gates score *highest* (med 0.112), because step + trim-leakage looks quadratic. Fires on 66% of benign anchors; gated power 0.66 → **0.22**. No effect-size floor separates the three populations. **Dead end — recorded so it isn't re-proposed.**

**Adopted — Δ-invariance across the existing ladder.** A gate's fitted gap Δ̂ (equal-variance 2-GMM component separation) is radius-invariant; a smooth confound's apparent gap shrinks once the ball enters its quadratic regime.

- Gates: ratio median **1.01**. Confirmed invariant.
- Gated power 0.58 vs 0.68 naive. The 10-point loss traces to bottom-rung mixture misestimation (ratio q10 = 0.08) — recoverable via the deepest-estimable-rung convention.
- **One halving is not enough.** Honest-resonant ratio at r/2 is 0.94; FP only 0.169 → 0.102. But **Δ̂(r/4)/Δ̂(r) median 0.04.** A third rung makes the discriminator decisive.
- Cost: fixed-offset boundaries lose estimability at depth (π ≈ Φ(−2.7) at r/4 for a boundary at 0.67σ_top). Hence: *evaluate invariance at the deepest rung where minority weight ≥ 0.05.*
- **S should default to 3, not 2.** Query cost +m per anchor.

Status: the invariance **mechanism** is confirmed at both depths. The full 3-rung decision rule's operating characteristics were not run end-to-end (→ Experiment A).

---

## 3. Power: why 0.5 is near the ceiling

The intuition that per-anchor power "should be higher" comes from treating this as a two-sample problem. It is not — the labels are latent, and unsupervised mixture detection is categorically harder. Scaling is in Δ/σ, not primarily in n.

**Δ=1.5σ is worse than it looks.** For an equal-weight, equal-variance two-component Gaussian mixture, the density is bimodal iff Δ > 2σ (Behboodian; higher for unequal weights). **At Δ=1.5σ the mixture density is literally unimodal.** The dip's 0.000 is not a weakness — it tests for bimodality, and there is none. It is answering correctly.

**What is left for the LRT.** The best-fitting single Gaussian matches the mixture's mean and variance exactly, so the first two moments carry zero signal. The LRT works off skewness and excess kurtosis:

| π | moment signal | implied z | predicted power | **observed** |
|---|---|---|---|---|
| 0.25 | skew ≈ 0.19 | ≈ 2.1 | ≈ 0.55 | **0.54** |
| 0.50 | skew = 0, kurt ≈ −0.26 | ≈ 1.45 | ≈ 0.30 | **0.32** |

Two consequences. The EM implementation is **near-efficient** — extracting essentially all available information, not broken. And the π non-monotonicity (peak at 0.25, trough at 0.50) is **forced by moment structure**: skewness carries a factor of (1−2π) and vanishes at π=0.5, leaving only a platykurtic flat-top.

**LRT vs dip, full comparison** (n=750, 250 reps/cell, LRT threshold 6.08 from 800 null sims at the 95th pct; null median 0.80 — nonstandard null as expected; empirical size 0.044):

| | sep=1.5, π=0.10/0.25/0.40/0.50 | sep=2.5 |
|---|---|---|
| dip | 0.000 / 0.000 / 0.000 / 0.000 | ≤ 0.04 |
| LRT | 0.42 / 0.54 / 0.32 / 0.32 | 1.00 across |

**Route to more power.** Moment z-scores scale like √n, so doubling z costs 4× the queries — m ≈ 3000 per anchor to move 1.5σ detection from ~0.5 to ~0.95. Expensive, and compounds with the coverage problem (§5).

**The cheaper route is pooling across anchors.** Boundaries are shared objects: the same hyperplane passes through many balls. Per-anchor power of 0.5 is nearly irrelevant if evidence for a *common* boundary aggregates across 200 anchors, and boundary orientation comes out as a byproduct. **This suggests per-anchor power is the wrong figure of merit.** The right one is power to recover a boundary given N anchors near it. → Experiment P.

### 3.1 Current power ledger

| | Effect on power |
|---|---|
| Δ-invariance guard | −0.10 (bought FP control; partly recoverable) |
| Quadratic repair | −0.44 (rejected) |
| Resonant masking attack | −0.60 (an *attack*, not a design choice) |
| LRT instead of dip | potentially +0.4 or more, **not yet integrated** |

To date, power has been spent to buy validity. The purchase that returns it is the LRT, still unclaimed.

---

## 4. Top-1 routing: structure and consequences

### 4.1 What top-1 buys

Under top-1 hard routing, crossing a boundary swaps expert i for expert j and the output jumps by f_i(x) − f_j(x) — **a genuine additive offset, scale-invariant, Δ̂ ratio ≈ 1.** This is exactly the object the framework assumes. Everything in the spec applies as written.

*(For the record, since it constrains generalization claims: top-2 softmax routing produces a **kink**, not a step — the entering expert arrives with weight → 0, so the output is continuous and the derivative jumps. A kink's apparent Δ̂ scales like r¹, giving ratio ≈ 0.5 — precisely the current threshold. Out of scope here, but the threshold placement is fragile if scope ever widens.)*

### 4.2 Δ-invariance is really a scaling exponent

| Structure | Δ̂ scaling | ratio at r/2 | ratio at r/4 |
|---|---|---|---|
| Step (top-1 gate) | r⁰ | 1.00 | 1.00 |
| Kink (top-2 gate) | r¹ | 0.50 | 0.25 |
| Curvature (confound) | r² | 0.25 | 0.06 |

Observed numbers fit: gates at 1.01; resonant curvature at 0.04 at r/4 (vs r²'s predicted 0.0625). The r/2 anomaly on the resonant surface (0.94) is the spec's own point — one oscillation is not yet the quadratic regime.

**Recommendation:** estimate α by regressing log Δ̂ on log r across rungs rather than thresholding a single ratio. Free, more robust, and yields a three-way classification.

### 4.3 Boundary geometry under a linear gate

Boundaries are where two logits tie — hyperplanes, giving a polyhedral tessellation. **Junctions are codimension-2.** If cells have scale L: fraction of anchors whose ball crosses any boundary ≈ r/L; fraction hitting a junction ≈ (r/L)². At r_hi ≈ 0.18, **local K>2 is rarer by another order of magnitude** — and when it occurs, it is a positive finding (a junction carries more partition information per query than a single boundary).

**Global K needs no bound.** Under top-1 with r_hi ≪ L, a ball generically crosses at most one boundary, so local K=2 regardless of how many experts exist globally. Global K is an R-phase quantity, estimated by *assembling* local evidence, not by fitting a large mixture.

**A deep gate is different:** boundaries are hyperplanes in representation space, and their preimages in covariate space are curved. At probe scale that is response-side curvature — the §2.4b blind spot — so the resonance band directly governs deep-gated FP rates.

---

## 5. Limitations register

Ordered by exposure to a hostile reviewer.

1. **Coverage (highest priority).** All measured power is per-anchor *given* a boundary in the ball. The fraction of randomly placed anchors containing a boundary is ≈ r_hi/L — roughly 15–20% at current scale. The rest return "no structure," locally true and globally uninformative. Fine for a detection claim; **a real problem for constructive recovery**, which is the strongest novelty claim. **Boundary-seeking anchor placement does not exist yet and is probably necessary.**
2. **Silent degeneracy when L < r_hi.** Many experts → small cells → ball spans several → many components at small separations → reads as heteroscedastic noise. **Stage 0 is blind**, because routing density lives in the response while covariate geometry is unchanged. No abstention fires. *This is the only failure mode that produces a confident wrong answer rather than an abstention.*
3. **Detection floor.** LRT reaches ~1.5σ at power 0.42–0.54; below that, nothing, at any π. Not fixable by better decision rules — it is the noise floor at m ≈ 10³ (§3). Trained experts are frequently near-homogeneous, so a real MoE may sit under the floor.
4. **E3 identifiability limit, untouched.** An offset parallel to within-branch covariate variation is absorbed by the linear fit. Genuinely unidentifiable from residuals. Consistent with the spec's open item 6.
5. **π ≈ 0.5.** The LRT fixes *testing* (0.32, graceful). Recovery still breaks — LTS at h=0.75 has no majority branch to trim toward, so the split is detectable but the sides are unlabelable. Open item 1, still open.
6. **Throughput.** 5/25 anchors reach "ok" at τ=0.005 with tight clusters. Correct abstention, but it multiplies against (1).
7. **Adversarial exposure, unmitigated.** Resonant masking drops power 0.68 → 0.085. No defense proposed; whether 3-rung invariance recovers any of it is open (→ Experiment A).
8. **Validation scope.** One model class (ReLU MLP), one D, one τ_y, no full-pipeline bootstrap. The extrinsic-curvature benignity result depends specifically on ReLU extrapolation being locally linear.

**Two that need a paragraph in the paper rather than a fix:** (1), because it separates "we detect a gate" from the constructive claim; and (2), because of the silent-failure character.

---

## 6. Edits to the spec, in order

1. **§3.3:** demote the equivalence to a two-mechanism division of labor — **plateau** = noise floor + dimension + cluster-scale exclusion; **density filter** = support departure + claim scope; **Δ-invariance over a 3-rung ladder** = response-side confound rejection. Move the A11 claim to the response side; define the exposure window as [model-class-learnable scale, probe diameter].
2. **Stage 1/3:** S = 3 rungs. Add the Δ-invariance rule with the deepest-estimable-rung convention. Route shrinking-ratio cases to A5 `structure_unattributed`. Prefer the log-log α regression (§4.2) over a single-ratio threshold.
3. **§3.4/§5:** round-consistency plateau rule replacing common intersection; W_min in rank-decades (≈0.6); add `censored` flag; split A1 vs A1n with the 3τ̂√D diagnostic.
4. **§3.3 table:** large-scale regime is non-monotone on clumpy data (exhaustion collapse before contamination jump).
5. **Stage 3:** adopt the equal-variance LRT alongside the dip, with bootstrap-carried calibration. State plainly that dip-only limits detectable Δ to ≳3σ at m ≈ 10³.
6. **New paper paragraph:** smooth resonant masking as a static attack composable with the scaffold (0.68 → 0.085).
7. **Small:** d̂ overshoot on tight clusters (junk frame direction); Stage-0 independence is from *responses*, not from f's training set.

*Note (§2 of the review):* V4 as written — "impose known curvature on the honest surface" — would have falsified §3.3 on its own. The spec's validation plan was honest enough to contain its own refutation, and that is worth saying in the writeup.

---

## 7. Experiment A — 3-rung Δ-invariance × LRT, end-to-end

Run as **one** experiment: the same EM fit yields both Δ̂ and the LRT statistic. The risk that makes it one experiment is that a more sensitive detector should also false-alarm harder on resonant curvature — the 0.169 FP was measured with a detector having *zero* power at 1.5σ.

```
DESIGN — full factorial, through the actual pipeline
(Stage 0 plateau → tangent frame → density filter → LTS → Stage 3),
not synthetic residuals.

  detector  ∈ {dip, LRT, dip∨LRT}
  rule      ∈ {naive (r_hi only), 2-rung (r/2), 3-rung deepest-estimable}
  surface   ∈ {honest smooth, honest resonant, gated Δ=0.15,
               gated + resonant (coexistence)}

Resonant ℓ set relative to each anchor's realized r_hi (target ~1
oscillation per probe ball), NOT as a fixed absolute — the resonance
band is defined in units of probe diameter. Record per cell the
fraction of the sine the trained model actually learned; discard
cells below 10% and report separately.

n ≥ 200 anchor-trials per cell. Conditional metrics condition on a
subset; 48–60 will not resolve them.

CALIBRATION — do not reuse 6.08.
That threshold came from clean Gaussian draws. LTS at h=0.75 truncates
tails before the LRT sees them, deflating the null. Recalibrate on
trimmed residuals from honest-smooth pipeline anchors; verify empirical
size ≈ 0.05 on a held-out honest-smooth cell before use. Report the
calibrated threshold and null median alongside the synthetic ones.

PRIMARY METRICS
  1. FP repair rate: P(rule rejects | naive fired, honest surface),
     split honest-smooth vs honest-resonant.
  2. Power retention: gated power under each rule / naive, same detector.
  3. shape_abstain rate → A5, split by cause (ratio shrank vs. minority
     weight <0.05 at every rung).
  4. Size on honest-smooth after recalibration.
  5. Coexistence-cell power (the 0.085 result) under each combination —
     does the LRT recover any of it?

PER-ANCHOR LOGGING
  r_hi, plateau width in rank-decades, censored flag, ρ at each rung,
  π̂ and Δ̂ at every rung, selected deepest-estimable rung, LRT statistic
  at every rung, dip p-value at every rung, abstention code.
  Log the FULL fitted offset and weight vectors, not just Δ̂, so
  component matching is reconstructable without a rerun.

PRE-REGISTERED PREDICTIONS (state before running; report hits and misses)
  - LRT raises honest-resonant FP above the dip's 0.169.
  - 3-rung invariance repairs it below 0.05; 2-rung does not (r/2 ratio
    was 0.94 on resonant surfaces).
  - Deepest-estimable-rung selection recovers part of the 0.68→0.58 loss.
  - Gated Δ̂ ratios stay ≈1.0 at whatever depth remains estimable.
  - Log-log α regression separates α≈0 from α≈2 more cleanly than the
    single-ratio threshold.

DELIVERABLE
Results table per cell with binomial CIs, plus an explicit statement of
which detector/rule combination becomes the spec default and its cost.
```

---

## 8. Experiment P — pooled detection across anchors

**Design principle:** false positives and true positives differ *geometrically*, not just in strength. A real boundary produces a consistent normal direction across every anchor touching it; a spurious detection produces a random one. Pooling on geometric agreement kills FP faster than it accumulates TP — exactly what is wanted when per-anchor power is 0.5.

Two corollaries: **do not threshold per anchor before pooling** (at power 0.5 that discards half the true signal), and the headline metric becomes **N₉₅ — anchors needed near a boundary for pooled detection at 0.95** — which is also the first quantity directly meaningful for the R phase.

```
PER-ANCHOR PRIMITIVE (compute for EVERY anchor, including non-firing)
  1. 2-GMM EM on LTS residuals in the tangent frame → posteriors γ_i.
  2. Linear discriminant on (z_i, γ_i) in tangent coords R^d̂
     → normal ν ∈ R^d̂, offset b. (§2.4b already noted residuals are
     near-linearly-separable in z when a gate is present.)
  3. Lift to ambient: n_a = V_a ν / ||V_a ν||, V_a the D×d̂ frame basis.
     Orient so the higher-mean component lies on the + side.
  4. Signed offset c_a = n_a · x_a + b/||ν||.
  5. Weight w_a = LRT statistic (NOT thresholded).
  → each anchor emits (n_a, c_a, w_a): a candidate ambient hyperplane
    with a confidence.

POOLING ESTIMATORS — compare three
  A. Statistic-only: Fisher/Stouffer combination of per-anchor LRT
     p-values. Ignores geometry. Baseline.
  B. Hard geometric: keep anchors above an LRT threshold, cluster (n, c)
     in projective space (antipodal identification), require
     |cos(n_a,n_b)| > 1−ε and |c_a − c_b| < δ.
  C. Soft geometric: w-weighted mode-seeking (mean-shift or Hough
     voting) over ALL anchors' (n, c), no thresholding.

NULL FOR ALL THREE
Permute the residual-to-point assignment within each anchor before
step 2, re-run the full pipeline. Randomizes normals while preserving
marginal residual distributions. ≥500 permutations. Report FP at
nominal 0.05 for each estimator.

GENERATIVE SETTING
Flat d=2 manifold in D=20, single planted hyperplane boundary.
  Δ/σ_resid ∈ {1.0, 1.5, 2.5, 5.0}
  π ∈ {0.10, 0.25, 0.50}
  N anchors ∈ {5, 10, 25, 50, 100, 200}
  anchor distance from boundary / r_hi ∈ {0.2, 0.5, 0.9, 1.5}
    (1.5 is a non-crossing control — must produce null behavior)

PRIMARY METRICS
  1. N₅₀ / N₉₅ per estimator per Δ. HEADLINE.
  2. Estimator ranking. Prediction: C > B > A at every Δ, gap C−B
     largest at Δ=1.5 (where half of true detections fall below any
     per-anchor threshold and B discards them).
  3. Boundary orientation error: angle(n̂, truth) vs N. First R-phase
     quantity; comes free.
  4. FP under permutation null — verify geometric pooling suppresses it
     below the statistic-only baseline, as the design principle claims.

SECONDARY
  - Two parallel boundaries at separation ∈ {0.5, 2, 5}·r_hi: does
    mode-seeking resolve two modes or merge them? Sets the resolution
    limit for regime recovery.
  - Curved boundary (deep-gate preimage): pool locally within anchor
    neighborhoods, then link. Report where the local-hyperplane
    approximation fails as a function of curvature × r_hi.
```

---

## 9. Experiment S — Slack scaffold characterization

**Premise:** every power number to date is against planted synthetic gates. Measure what the actual target induces before optimizing further. If the scaffold's Δ is in the 3–5σ range, the LRT is already at 1.00 and the detection-power question largely dissolves — the binding constraint becomes coverage and the on-manifold tension.

```
BUILD
Reproduce Slack et al. (AIES 2020) on COMPAS, Communities and Crime,
German Credit: biased f (uses sensitive attribute), innocuous ψ (uses
an uncorrelated synthetic feature), OOD detector (RF trained to separate
real data from LIME-perturbed samples).

Audit the score/probability output, NOT the hard label — the framework
presumes a continuous response. If only hard labels are exposed, stop
and note that a discrete-response Stage 3 variant is a prerequisite.

MEASUREMENTS
  S1  Δ/σ_resid: sample point pairs straddling the OOD decision
      boundary at matched covariate distance; measure the jump in
      audited output relative to LTS residual scale.
      THE number that determines whether detection power is a live
      problem at all.
  S2  Boundary accessibility: fraction of real held-out points lying
      within r_hi of the OOD boundary. Coverage for the real target.
  S3  Density-filter tension: for anchors whose ball crosses the OOD
      boundary, record ρ. If ρ < threshold at essentially all of them,
      A3 fires wherever the signal is and the approach abstains on its
      own target. EXPLICIT FALSIFICATION CONDITION — report as such.
  S4  Detector leakage: fraction of real points the OOD detector
      misclassifies. The on-manifold region where crossing is possible
      without leaving support.
  S5  Probe classification: what fraction of the audit's own probe
      points does the OOD detector call OOD? The audit's perturbation
      (isotropic, tangent-frame, small radius) is NOT LIME's (unit
      Gaussian, ambient, all features) — the detector was never trained
      against it.

PRE-REGISTERED TRICHOTOMY — S5 determines the regime
  (i)   Probes stay inside the real region → e(x)=f(x) throughout the
        ball → no gate detected, but attributions reflect the BIASED
        model. Scaffold defeated by evasion rather than detection.
        Still a paper result, framed differently.
  (ii)  Probes straddle → gate detected → DRI works as designed.
  (iii) Probes land in the OOD region → sees ψ only → fooled. The
        density filter should prevent this; verify that it does.

Report which regime each dataset falls into. (i) and (ii) are both wins;
write the paper to accommodate either.
```

**Run S5 and the trichotomy first.** It is cheap, and it determines whether the paper's central claim is "we detect the switch" or "we are immune to it" — which changes the framing of everything upstream.

---

## 10. Open items not addressed anywhere

- **Full-pipeline bootstrap.** Owed everywhere; a separate build. Until it exists, all absolute rates are approximate.
- **Boundary-seeking anchor placement.** Prerequisite for the R phase; does not exist.
- **E3 identifiability** (offset parallel to within-branch variation). Untouched, consistent with spec open item 6.
- **π ≈ 0.5 recovery-side breakdown** of trimming. LRT fixes testing only.
- **Generalization** past one model class, one D, one τ_y.
- **Discrete-response Stage 3**, if the Slack reproduction exposes only hard labels.

### Suggested sequencing

1. **Experiment S, S5 + trichotomy only** — cheap, determines paper framing.
2. **Experiment A** — settles the spec defaults; gated by nothing.
3. **Experiment S, S1–S4** — sizes the real target.
4. **Experiment P** — bridges D → R; the pooling result is the natural workshop-paper centerpiece if the R phase is not yet complete.
