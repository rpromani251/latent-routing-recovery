# How Every Step Got Here: A Derivation Guide for the v2 Routing Audit

This guide walks through the full algorithm in the order a reader meets it in `routing_audit_v2.tex`, and for each step explains what problem it solves, what simpler thing was tried first, why that simpler thing failed, and which round of criticism forced the change. It is deliberately verbose: the point is that nothing in the pipeline is decorative — every component is a repair for a specific, named failure.

The history has three layers. First, the **SPA/profile-mean route** (the original framework): factor a profile matrix M = WH + N and identify simplex vertices. Second, the **v1 note**: abandon profile means, retain individual queries, add a detection stage, recover by local mixtures plus label synchronization. Third, the **v2 revision** (this document's subject): a critique pass that found the calibration story of v1 overstated in specific places — the null was not really K=1, the empirical-null BH was not really FDR-controlled, the ratio statistic was not really stable, propagation through unflagged regions was not really justified — and repaired each.

---

## Part 0: The inheritance — why the SPA route was abandoned

The original method built, for each anchor, a response profile: perturb the input many times, push each response through the feature map ψ, and average into one V-dimensional column of a matrix M. If routing mixes K regimes with regime signatures w_k, then each column is approximately a convex combination — M = WH — and columns live in a simplex whose vertices are the w_k. Separable NMF (SPA) finds the vertices if some columns are pure (an anchor whose whole perturbation cloud stays in one regime), and simplex-constrained least squares then reads off every anchor's composition.

Three structural problems accumulated against this route:

1. **Averaging destroys the evidence.** The first step — collapsing each anchor's response cloud to its mean — erases exactly the signal that proves routing exists: the multimodality of the cloud. A bimodal response cloud whose two modes are regime-1 and regime-2 behavior averages to a point *between* the regimes, a behavior the model never exhibits. The information the auditor most needs is spent before estimation begins.

2. **Separability is an assumption about luck.** SPA needs pure columns: anchors whose entire perturbation neighborhood stays inside one regime. Whether such anchors exist depends on where the gates fall relative to your anchor set and probe scale — a fact the auditor can neither control nor verify from the outside. When separability fails, vertex estimates are biased in ways that are hard to diagnose.

3. **Constant signatures w_k are false in realistic systems.** A regime's response legitimately varies across the input space: an "energy-saving mode" responds differently in July than January. The simplex picture requires each regime to be one fixed point in response space. Realistic within-regime drift smears the vertices and breaks the geometry.

The pivot that defines the current framework: **keep the individual query responses.** Every query j at anchor i has a single latent routing label a_ij = g(x_i + δ_ij). The profile is an admixture, but the *queries are pure* — purity is free at the query level, and it was expensive and unverifiable at the profile level. This one decision converts the recovery problem from "find simplex vertices among averaged profiles" to "cluster responses locally, then stitch local labels into global identities." Everything in Stage B follows from it.

---

## Part 1: Why detection precedes recovery

The v1 note introduced a gate in front of recovery: do not attempt to decompose anything until there is calibrated evidence that one smooth mechanism is insufficient. Three reasons, in increasing order of importance:

- **Budget.** Recovery is query-hungry (fresh samples, mixture fitting, bridge probes). Spending it everywhere is wasteful; spending it only at flagged anchors concentrates power.
- **False-structure risk.** Clustering algorithms return clusters whether or not clusters exist. An unconditional recovery pipeline will happily "find" regimes in a smooth model. A detection gate with an error guarantee is what separates an audit from a Rorschach test.
- **Reporting honesty.** The final audit separates *existence* (statistical claim), *structure* (estimation claim), and *interpretation* (substantive claim). Existence needs its own calibrated test — it cannot be an afterthought of estimation, because estimation assumes what existence is supposed to establish.

---

## Part 2: The detection statistic, step by step

### 2.1 From raw dispersion to scale-normalized dispersion

The primitive observation: if a perturbation cloud straddles a routing boundary, responses come from two mechanisms and the response scatter is inflated relative to what one smooth mechanism would produce. So measure scatter. But raw scatter conflates three sources: observation noise (constant in σ), smooth response variation (grows like σ · slope), and regime mixing (appears only at scales large enough to cross the gate, then saturates). The v1 construction isolates the interesting part:

- Take the top eigenvalue of the response scatter matrix, v_i(σ) — a one-number summary of the widest direction of spread.
- Subtract the noise floor estimated from repeated identical queries.
- Divide by σ, because smooth variation contributes spread proportional to σ. For a locally linear single mechanism, the result r_i(σ) ≈ |local slope|, approximately **flat in σ**.

A regime boundary at distance d from the anchor produces a signature *bump*: r_i is flat below σ ≈ d (cloud doesn't reach the gate), rises when the cloud starts straddling it (a between-regimes jump of fixed size Δ divided by σ is large when σ first covers d), and falls again as σ grows (the fixed jump is divided by an ever larger σ). Scale-dependence of r_i is therefore the detection signal, and the peak scale σ* estimates the distance to the boundary — which Stage B later reuses as the right scale for recovery sampling.

### 2.2 Why the noise floor forced whitening (v2 change)

The v1 formula subtracted a *scalar* noise variance from the top eigenvalue. That silently assumes observation noise is isotropic in ψ-space. If noise is larger in some response coordinates than others (nearly always true — think logits of rare classes), the scalar subtraction under- or over-corrects depending on which direction the top eigenvector points, biasing r_i in a direction-dependent way. The fix is standard: estimate the noise **covariance** from repeats, whiten all features by its inverse square root, and the floor becomes exactly 1 (identity) by construction. The cost is needing a stable covariance estimate — when the response dimension V is large relative to the number of repeats, the raw covariance is singular, so a shrinkage estimator (Ledoit–Wolf) or low-rank model is required. This trade — a cleaner statistic in exchange for one more estimated object — recurs throughout the framework, and the rule adopted each time is: make the estimated object explicit, and put its estimation *inside* the bootstrap so its uncertainty is paid for, not ignored.

### 2.3 Why the ratio statistic died (v2 change)

v1 summarized the curve r_i(σ_1), ..., r_i(σ_T) by the ratio R_i = max_t r_i / min_t r_i. The critique identified a fatal instability: the denominator. At a locally flat anchor, the true spread is near zero, so after noise-floor subtraction the estimate min_t r_i is a small number dominated by estimation error. A slightly *underestimated* denominator makes R_i explode — the anchor is flagged not because any scale shows a bump but because one scale's spread was noise. The statistic confounds "bump present" with "normalization unstable."

Two repairs, layered:

1. **Minimum-signal abstention.** Before computing any scale statistic, test whether the excess eigenvalue is significantly positive at *any* scale. If not, the anchor is reported as **insufficient signal** — a new output category, distinct from "null-consistent." This distinction matters for the audit's meaning: a flat model tells you nothing about boundaries; it does not tell you there are none. Treating no-signal anchors as clean anchors would quietly convert absence of power into evidence of innocence.

2. **Replace the ratio.** Two candidates, in order of preference:
   - *Log-range*: max_t log(r_i + ε) − min_t log(r_i + ε). The log tames the explosion (a ratio of 10 from a noise-dominated denominator becomes an additive 2.3, and ε bounds it), and range-of-logs equals log-of-ratio, so the statistic still measures relative scale-dependence. This is the fallback when no null model is fitted yet.
   - *Studentized deviation from the fitted null scale curve*: the statistic the framework actually wants. Once a null model (Part 3) predicts, for each anchor, the mean and standard deviation of log r_i(σ_t) under smooth single-branch behavior, the statistic becomes: at which scale does the observed curve exceed the null prediction by the most standard deviations? This asks the calibrated question directly — "is this scale-dependence larger than smooth behavior explains *at this anchor*?" — rather than the proxy question "is this curve non-flat?" Non-flatness was never the right question, because curvature makes null curves non-flat too. This is the deepest of the statistic-level repairs and it is what ties the statistic to the null generator.

### 2.4 What the null actually is (the v2 reframing)

v1 said Stage A tests K = 1. The critique's central correction: it does not, and cannot. Consider what a *single* smooth mechanism can produce: curvature (r_i rises with σ), spatially varying derivatives (different anchors have different slopes), heteroskedastic noise (some anchors noisier), anisotropic noise, sharp-but-continuous transitions (a steep sigmoid inside one mechanism is indistinguishable at probed scales from a hard gate), and locally flat regions (normalization instability). Every one of these can trigger a naive dispersion test without any routing. Conversely, real routing can be invisible: branches identical under ψ, probes parallel to the gate, mixtures too imbalanced to show a second mode.

So the honest null hypothesis is not "K = 1." It is:

> H_{0,i}: anchor i follows the calibrated smooth single-branch null — the fitted reference distribution of single-branch smooth behavior.

And the honest claim on rejection is: this anchor's multiscale response geometry is **incompatible with the fitted population of smooth single-branch behaviors** — which is *consistent with* boundary crossing or latent routing, and must be corroborated (shape evidence, recovery structure, mechanism analysis) before routing language enters the report. The claim is triply relative: to ψ, to the probe design, and to the null class. A kink sharper than the null's smoothness class is *outside the null by definition* and will be flagged — correctly, because at the level of observable behavior it *is* a regime boundary. Whether the mechanism behind it is a routing gate or an intrinsic discontinuity is a Stage C question that output-only observation cannot settle. This is why the negative-control suite includes a discontinuous-but-unrouted control: not because the audit should pass it silently, but to document that flagging it is the designed behavior and mechanism attribution is out of Stage A's scope.

This reframing is not a retreat. "Incompatible with calibrated smooth single-branch behavior under a specified probe design" is a meaningful, defensible, publishable auditing claim — arguably the *only* version of the claim that survives review. The reframing also dictates the machinery: if the null is "the fitted reference distribution of smooth behaviors," then something must *fit that distribution*. That something is the null generator.

---

## Part 3: The null generator — what it is, why a GP, how it can fail

### 3.1 What the object is

Every calibrated quantity downstream — the studentization baseline, screening p-values, the selection-aware bootstrap — needs to answer one question: *what would my complete audit dataset have looked like if the model were a single smooth mechanism?* The null generator P̂_0 is the fitted object that answers it. It is a generative model of the **entire audit** — all responses at all anchors, all scales, all probes, plus repeated queries — under the single-branch hypothesis. One draw from P̂_0 is a full synthetic audit dataset, indistinguishable in format from the real one, on which the *entire pipeline* (whitening, dispersion curves, null refits, screening, projections, dip tests) can be rerun.

It must represent four things, because each one, if misrepresented, corrupts calibration in a known direction:

- **Smooth local response surfaces** with realistic slopes *and curvatures* — because curvature is the principal null-side confound; a generator with linear surfaces would flag every curved anchor.
- **Cross-anchor variation** in those slopes and curvatures — because the empirical question is whether an anchor is unusual *relative to the population*, so the generator must reproduce the population's spread, not just its center.
- **Observation noise**, including anchor-level heteroskedasticity — or noisy anchors become false flags.
- **Dependence** between anchors (spatial, batch) — or multiplicity corrections and bootstrap quantiles are computed under a false independence and are anti-conservative.

### 3.2 Why a Gaussian process, specifically

The null hypothesis is, in words, "the local response is one smooth function." A Gaussian process is the canonical probability distribution *over smooth functions*, which makes it the mathematically minimal object that turns the verbal null into a generative one. Concretely, the local surface at anchor i is modeled as a GP with a linear mean (the local slope β_i) and a Matérn covariance with amplitude s_i and lengthscale ℓ_i. Three properties make this the right default rather than a fashionable one:

1. **The hyperparameters are the null's own vocabulary.** β_i is "how steep is smooth behavior here," s_i is "how much smooth nonlinearity is there," ℓ_i is "over what input distance does the response bend," and the Matérn order ν is literally "how many derivatives does smooth mean." The v2 reframing said the null is a *smoothness class*; the GP makes the class an explicit, estimable, reportable choice instead of an implicit consequence of some test statistic. When a reviewer asks "what exactly do you mean by smooth?", the answer is a kernel and its fitted hyperparameters.

2. **Sampling is exactly the operation the bootstrap needs.** The selection-aware bootstrap needs complete synthetic audits. A GP delivers them by construction: draw hyperparameters from their fitted population law, draw the surface's values at the actual probe locations (a finite multivariate normal with the kernel Gram matrix — no function-space machinery needed), add noise. Do this at the original anchors, ladder, and probe counts, jointly by spatial blocks, and you have one bootstrap replicate. The synthetic audits automatically contain curvature-driven scale-dependence — so the bootstrap's null distribution of the scale statistic *includes the confound*, which is precisely what makes the calibration mean something.

3. **It supplies the studentization baseline for free.** The preferred statistic (2.3) needs, per anchor, the null mean and sd of the log dispersion curve. Under the fitted GP these are directly simulable (or in parts analytic). Statistic and null model become two views of one fitted object rather than two separately tuned components.

Alternatives — local polynomials with random coefficients, spline models — are legitimate members of the same design. The GP is the default because it parameterizes exactly the thing the null is about (smoothness) with few interpretable hyperparameters; the framework's validity claim is conditional on *some* adequate generator, not on GPs per se.

### 3.3 The contamination problem and cross-fitting

There is a circularity hazard at the heart of any empirical null: the generator must be fitted from audit data that may *contain the alternatives being hunted*. If flagged-to-be anchors contribute to the fit, the null absorbs some routing signal, inflating its spread and destroying power (or worse, in the tails, miscalibrating size). v1 waved at this with "fit to the central mass, excluding the suspected upper tail." The critique correctly said that is not a procedure — the fitted null is random, contaminated, and dependent on every tested observation, so empirical-null + BH carries no automatic guarantee.

The v2 repairs, in increasing strength: **trimmed fitting** (exclude the top fraction by a preliminary statistic; report sensitivity to the fraction — now a stated procedure rather than a vibe); a **separate calibration anchor set** never tested; and **cross-fitting** as the default — partition anchors into folds, test each fold against a generator fitted on the *other* folds. Cross-fitting breaks the dependence of each anchor's p-value on its own data's contribution to the null, which is the specific pathology BH cannot survive.

### 3.4 Misspecification honesty

The generator is estimated, so it is wrong somewhere, and the direction of wrongness maps to the direction of error: too-rigid smoothness (long ℓ, high ν) → real curvature looks like routing → anti-conservative; too-flexible smoothness (short ℓ) → the generator imitates the mixture bumps themselves → power collapses. Neither failure is detectable from inside the fitted model, which is why the evaluation plan gained a dedicated misspecification suite: run the full pipeline against heavy-tailed noise, unmodeled heteroskedasticity, kinks, and a wrong ν, and *measure* the realized error rates. The published claim is then "calibrated conditional on generator adequacy, with measured degradation under these misspecifications" — the strongest version of the claim that is actually true.

---

## Part 4: Screening, corroboration, confirmation — three jobs, three tools

### 4.1 BH demoted to screening (v2 change)

v1 applied Benjamini–Hochberg to empirical-null p-values and called the result FDR-controlled. Per 3.3, the premises fail: estimated null, contamination, dependence. Rather than pile on conditions until BH is defensible, v2 reassigns its job. BH is a **screening device**: a principled way to concentrate the expensive confirmation budget on a manageable set of anchors. It is tuned for that job (sensitivity analyses via blockwise or BY variants for dependence), and it carries **no final error-rate claim**. The final claims live elsewhere: the omnibus bootstrap for existence, per-anchor selection-aware p-values with dependence-robust corrections for anchor-level statements. Separating screening from confirmation resolves the calibration burden by not asking one tool to do both jobs.

### 4.2 The dip test and the projection problem (v2 change)

A dispersion bump says "more spread than smoothness explains"; it does not say "two populations." The corroborating signature of mixing is *multimodality* of the response cloud at the bump scale. Hartigan's dip test is the standard nonparametric unimodality test — but it is **univariate**, and v1 said "project onto the leading principal directions and dip," which is ambiguous when the projection has more than one dimension and silently favorable when the direction is tuned on the same data being tested.

v2 pins it down: default = dip on the *first* PC score of the whitened confirmation sample; alternative = a small *prespecified* direction set (top two PCs, plus a candidate gating direction when one exists) with Bonferroni across the set; optional = projection pursuit maximizing a bimodality index, permitted **only** if the entire pursuit is re-run inside every bootstrap replicate, so the search pays for itself in the null distribution.

One limitation is documented rather than fixed, because it is not fixable within PCA: principal components maximize *variance*, not *between-regime separation*. A mixture whose separation direction is orthogonal to the dominant within-regime variance direction hides from the top PC. Hence a non-significant dip does not certify unimodality — stated as a known power limitation, consistent with the framework's general rule that negative results are qualified, never absolute.

### 4.3 The selection-aware bootstrap (v1's best idea, kept, with honest labeling)

The anchor was screened from the data; its scale σ* was chosen because it maximized the statistic; the projection was estimated from the flagged responses. Testing the dip *as if* these were fixed in advance is classic selection bias — every choice leaned toward significance. The v1 construction that survives review intact: make selection part of the statistic. Define T_i = 1{i ∈ F} · Dip_i, where F, σ*, and the projection are all re-derived per dataset, and compare against replicates in which the *complete pipeline* — noise estimation, whitening, dispersion curves, null refits (including cross-fitting folds and trimming), screening, projection choice — is rerun from scratch on each synthetic audit from P̂_0. The (1 + #{T^(b) ≥ T^obs})/(B + 1) form is the correct finite-simulation p-value and cannot return zero.

Two v2 honesty edits. First, the word "exact" is retired: exactness would require exchangeability of observed and simulated statistics under the *true* null, but the simulations come from an *estimated* generator — this is an approximate parametric bootstrap, calibrated conditional on generator adequacy (the phrase now appears wherever the p-value does). Second, the **omnibus statistic T_max = max_i T_i is promoted to primary**: the headline claim of the audit is "somewhere, this model's behavior is incompatible with smooth single-branch behavior," and the maximum over anchors absorbs the search across anchors automatically, without pretending to precise anchor-level FDR before the null machinery has earned it. Anchor-level claims remain available, but as the secondary, more heavily caveated product.

### 4.4 Budget splitting (v2 addition)

All of 4.1–4.3 exists because data reuse creates selection effects the analysis must then undo. The cheaper fix, when the query budget allows, is not to reuse: four disjoint samples — S1 detection and scale selection, S2 modality confirmation, S3 recovery, S4 final validation. The bootstrap *can* model reuse (that is what selection-awareness is), but a physical split is easier to explain, harder to attack, and removes the dependence of confirmatory claims on having modeled the reuse correctly. Design rule adopted: prefer independence by design; use selection-aware inference when the budget forces reuse; never silently do neither.

---

## Part 5: Recovery — local mixtures done correctly

### 5.1 Fresh sample at σ*

Recovery begins by re-querying flagged anchors near their peak scale, from S3. Two reasons: σ* is the scale at which the mixture is most visible (that is what the peak *means* — Stage A's byproduct becomes Stage B's tuning parameter), and using fresh data severs recovery estimates from the selection event (an anchor flagged partly by luck would otherwise contribute its lucky fluctuation to the mixture fit — winner's curse).

### 5.2 Soft responsibilities (v2 change)

v1 estimated the local composition by hard-label frequencies: fit a mixture, assign each query to its most probable component, count. The critique confirmed a real statistical error: hard-assignment frequencies are **biased toward uniform** whenever components overlap, because boundary-zone queries are disproportionately captured by the larger/closer component's territory... more precisely, classification counts shrink extreme proportions toward 1/K, and the bias grows with overlap — exactly the regime where auditing operates. The consistent estimator is the mixture's own posterior: π̂_i(ℓ) = (1/m) Σ_j γ_ij(ℓ), where γ_ij(ℓ) is the responsibility of component ℓ for query j. The responsibilities then flow downstream — minimum-prevalence rules use soft mass, gate fits use them as weights — so assignment *uncertainty* is carried forward instead of being rounded away at the first opportunity. The local component count K_loc is selected by held-out likelihood, bootstrap stability, a complexity criterion, and the soft minimum-mass rule; the unperturbed anchor response is posterior-assigned to identify which branch *owns* the anchor (orientation only — v2 keeps v1's refusal to treat the anchor as a pure archetype, which is the surviving lesson of the SPA route's collapse).

---

## Part 6: Synchronization — from problem statement to algorithm

### 6.1 Why the problem exists and why global clustering cannot solve it

Mixture labels are arbitrary per anchor: "component 2" at anchor i and "component 1" at neighbor j may be the same regime. The naive fix — pool all responses everywhere and cluster once — fails for the reason that killed constant-w_k: within one regime, responses drift across the input space, so global clusters organize by magnitude, geography, season — anything but mechanism. The correct object is a **label synchronization problem on the anchor graph**: match components between neighboring anchors (where within-regime drift is small, by continuity assumption B5), then make the matchings globally consistent.

And it is synchronization over **partial** matchings, not permutations — the harder variant — because not every regime appears in every neighborhood (an anchor deep in regime 1's territory near the 1/2 boundary sees {1, 2}; another near the OOD cliff sees {1, 3}). Any method assuming complete correspondences would force spurious matches exactly where the structure is most informative. The Slack-style stack is the worked illustration: local views {1,2}, {1,3}, {2,3} chain into one three-regime system through shared components — global structure assembled from partial views, including a regime (the OOD leaf) that owns no unperturbed anchor at all.

### 6.2 The committed algorithm (v2 change)

v1 said "synchronize the pairwise matchings to be cycle-consistent" — a property, not a procedure; the critique flagged it as unimplementable as stated. The v2 commitment (SYNC1–7): build the anchor graph; compute regularized Mahalanobis matching costs between component pairs across each edge (covariance-weighted, so a match is judged relative to the components' own spreads); solve a rectangular assignment **with a dummy option** at cost χ so components can remain unmatched (this is what makes matchings partial — χ is calibrated as a high quantile of same-branch matching costs under the fitted continuity model, with sensitivity reported); score each edge's **confidence** as the margin between its best and second-best assignment (the operational form of margin assumption B6); take the **maximum-confidence spanning forest** and propagate labels along it (a tree has no cycles, hence no inconsistency — propagation along the most trustworthy tree is exact by construction); then **audit every off-forest edge** by re-matching it independently and checking agreement with the forest labeling. The confidence-weighted agreement rate ρ_cc is the health statistic of the whole synchronization: high ρ_cc means the redundant edges confirm the tree; low ρ_cc means branches are being confused, and the affected component families are **abstained from, not forced**. Finally K̂ is defined operationally: the number of global classes that carry enough aggregate soft mass *and* persist across Stage-B bootstrap replicates — an answer to "how many regimes?" that inherits its error bars from the resampling rather than from optimism.

Design logic worth internalizing: the forest gives *feasibility* (always some consistent labeling), the audit gives *falsifiability* (a measured consistency rate with an abstention trigger). Feasibility without falsifiability was v1's problem; falsifiability without feasibility is a theorem with no algorithm.

---

## Part 7: Gates, uncertainty, and the certification principle

### 7.1 The gate fit and its two pathologies (v2 changes)

Once queries have global labels, each flagged anchor holds a supervised dataset {(δ_ij, label)} and the local boundary is estimated by logistic regression of labels on perturbations; the normalized coefficient vector is the boundary normal, the intercept its offset, and a sparse fit nominates gating features. Two pathologies repaired in v2. First, **perfect separation**: when the regimes separate cleanly in δ-space — the *success* case — the unpenalized logistic MLE diverges; ridge regularization (or Firth's penalty for small local samples) keeps the fit finite while preserving the direction. An algorithm that crashes precisely when the structure is clearest is not an acceptable failure mode. Second, **hard labels overstate information**: the labels are estimated, not observed. The fit therefore minimizes cross-entropy against the soft responsibilities (equivalently, responsibility-weighted logistic regression), so queries the mixture is unsure about pull on the boundary weakly.

### 7.2 Stage-repeating uncertainty (v2 change)

Ordinary logistic standard errors condition on everything upstream being true: the mixture fit, the K selection, the synchronization, the projection, the scale. Five estimated layers, silently treated as known. The critique's demand, adopted wholesale: report uncertainty from a **Stage-B bootstrap** that resamples the recovery queries and repeats *everything* — mixture fitting and K_loc selection, responsibilities, partial matching, synchronization, gate fitting — per replicate. Outputs: normal *cones* (not just standard errors) for boundary normals, intervals for offsets and compositions, persistence frequencies for the global classes (feeding K̂). Same principle as Stage A's bootstrap: any adaptive choice outside the replicate is a source of unearned confidence.

### 7.3 Certified propagation — "unflagged is not interior" (v2's most important Stage-B change)

v1 propagated labels from flagged anchors through unflagged regions, treating unflagged as interior. The critique demolished this: an unflagged anchor may be truly interior — or underpowered, probed in the wrong directions, sitting near a gate with overlapping outputs, hosting a rare regime, or a multiplicity miss. Non-rejection is not evidence of no boundary; propagating through non-rejection **confidently paints exactly the regions where detection was blind**. The audit would be most assertive precisely where it knew least.

v2 inverts the default: no edge carries a label unless it is **certified**. Certification is active: place bridge points along the path between neighboring anchors, probe each with a small battery at a bridge scale, and demand (i) every bridge point passes the minimum-signal and dispersion checks with no modality evidence, and (ii) unperturbed bridge responses interpolate within the predictive tube of a single-branch fit joining the endpoint components. Pass → *certified same-branch*, propagate. An interior bridge point flags → *certified crossing* — which is not a failure but a **discovery**: bisection along the edge localizes the boundary and hands Stage C its geometry. Anything else → *uncertifiable*; labels carried across anyway are stamped EXTRAPOLATED and severed from certified claims. Every regime map now distinguishes certified from extrapolated territory, and propagation abstains in components disconnected from any reliably labeled anchor. The general principle, now applied uniformly: **absence of evidence is never promoted to evidence of absence anywhere in the pipeline** — the same principle that created the insufficient-signal category in Stage A and the qualified negative dip result in A.5.

---

## Part 8: Interpretation — weakened on purpose

Stage C converts structure into a finding, and v2 weakens its language deliberately. Multiple regimes are not misconduct: mixtures of experts, safety routers, and operational controllers are legitimately gated, so Stage C tests whether the recovered routing is *consequential* — associated with protected attributes, geography, energy burden, service quality, error rates — with spatially aware inference, because spatially smooth confounders manufacture spurious associations under iid tests. The probe-support comparison drops v1's near-diagnosis ("ambient-only ⇒ OOD scaffold") for what the evidence supports: ambient-only evidence is *consistent with* an off-manifold scaffold — and is additionally qualified by the support realism score (v2 addition), because ambient probing may have wandered into input regions where behavior is operationally meaningless; on-manifold evidence indicates behavior relevant to the modeled data support — relevance, not wrongdoing; and a claim of misconduct requires a separate interpretation and impact analysis that no clustering output can supply. The reporting contract keeps the three claims — existence, structure, interpretation — separate, so failure at a later stage never erases an earlier result, and success at an earlier stage never inflates a later claim.

---

## Part 9: The operational protections (v2 additions), and why each exists

- **Budget split (S1–S4).** Selection effects are cheaper to prevent than to correct; see 4.4.
- **Negative controls.** Curved single-regime, heteroskedastic single-regime, discontinuous-but-unrouted, benign MoE, irrelevant probe directions — run through the identical pipeline. They answer the referee's first question ("does this fire on ordinary nonlinearity?") with data, calibrate expectations for the observationally-equivalent cases (the discontinuous control *should* flag — documenting that is what keeps the claim honest), and the benign-MoE control rehearses the Stage B/C separation: structure recovered, misconduct not asserted.
- **Support diagnostics.** Every probe family at every anchor reports how realistic its perturbed inputs are (e.g., kNN-distance quantiles against a reference sample). Without this, "ambient-only detection" is uninterpretable — it may mean "hidden scaffold" or "we probed nonsense inputs and the model behaved differently on nonsense."
- **Minimum-signal abstention** (2.3) — the third abstention surface, alongside synchronization abstention and certification abstention. The framework's identity is that declining to answer is a first-class output with its own measured operating characteristics.

---

## Part 10: What Stage B still needs (the parity contract)

Stage A's maturity consists of: a precise null claim, a concrete calibration object (P̂_0), a stabilized statistic, selection-aware inference, and a falsification plan. Stage B reaches parity when it has the six analogues, and this list is the honest to-do list rather than a rhetorical flourish:

1. **A formal estimand and loss.** Stage A has H_0; Stage B needs its target stated at the same precision: the partition restricted to probe-accessible territory up to label permutation, the composition field, the local normals — with matching losses (permutation-minimized partition error *on certified territory*, composition error, angular normal error). Without a named estimand, "recovery worked" is unfalsifiable.
2. **Fully committed procedures.** SYNC1–7 and CERT1–4 close the two biggest gaps; the residual free parameters (χ, confidence thresholds, n_min, bridge densities) need registered defaults with sensitivity analyses. Maturity test: two auditors, same data, same output.
3. **Uncertainty propagating every layer** — the Stage-B bootstrap, with no layer conditioned away.
4. **A theorem-shaped target.** The minimal honest version: under the local distinguishability, continuity, margin, and connectivity assumptions with margin γ, per-edge certified-matching error α_e (into which finite-sample mixture estimation error enters), forest propagation recovers global labels up to permutation with probability ≥ 1 − Σ α_e over forest edges; soft compositions obey a CLT under the local model. Stated as a target, not a result — the same discipline as v1's claim-boundary section, now with the bound's *shape* visible so the proof obligations are enumerable: mixture error bounds, matching-cost concentration, a union bound over the forest.
5. **Abstention operating characteristics.** Measured, not asserted: abstention rate when assumptions fail vs. when they hold. "Knows when it does not know" becomes a reported number.
6. **Inherited selection discipline.** Stage B lives downstream of Stage A's selection; recovery estimates reusing detection data inherit winner's-curse bias. The budget split is the design fix; any unavoidable reuse goes inside the Stage-B bootstrap.

---

## Coda: the one-sentence version of every repair

| Step | Failure it repairs |
|---|---|
| Retain individual queries | Averaging destroyed the multimodality that proves mixing (SPA route) |
| Detection gate before recovery | Clustering finds clusters whether or not they exist |
| Scale normalization r_i(σ) | Raw scatter conflates noise, slope, and mixing |
| Whitening | Scalar floor subtraction assumed isotropic noise |
| Minimum-signal abstention | Flat anchors produced explosive ratios and fake flags |
| Log-range / studentized statistic | The ratio's denominator was noise-dominated; flatness was never the null |
| Reframed H_0 | A single smooth mechanism can mimic every symptom "K = 1" was said to exclude |
| GP null generator | "Calibrated" requires an explicit fitted model of what smooth means |
| Trimming / calibration split / cross-fitting | The null was fitted on contaminated, dependent data |
| BH as screening only | Empirical-null BH carried an FDR claim its premises could not support |
| Specified dip projections | "Leading directions" was ambiguous and silently favorable |
| Omnibus T_max primary | Anchor-level FDR was asserted before the machinery earned it |
| "Conditional on generator adequacy" | The bootstrap was called exact; it is approximate-parametric |
| Budget split S1–S4 | Reuse created selection effects the analysis then had to undo |
| Soft responsibilities | Hard-label frequencies are biased toward uniform under overlap |
| Partial matching with dummy cost | Not every regime appears in every neighborhood |
| Confidence forest + cycle audit | "Synchronize consistently" was a property, not a procedure |
| Ridge/Firth gate fits | The MLE diverged exactly when separation was cleanest |
| Stage-repeating bootstrap | Logistic standard errors ignored five estimated layers |
| Edge certification | Unflagged ≠ interior; propagation filled in where detection was blind |
| Support diagnostics | "Ambient-only" was uninterpretable without probe realism |
| Negative controls | No evidence the statistic responds to switching rather than nonlinearity |
| Softened Stage C language | Ambient-only evidence is consistent with a scaffold, not proof of one |
| Parity contract | Stage B's maturity gap needed an enumerated, falsifiable to-do list |
