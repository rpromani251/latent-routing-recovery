# Status — AISTATS readiness audit

**Date:** 14 August 2026
**Scope:** everything in the `3 Stage Audit` vault folder, the `detect-recover-interpret`
repo at commit `c8d9133` (working tree clean), and `geospatial-xai-attacks` at `5ebf68e`.
Doc claims were spot-checked against the summary CSVs (E1/E3/E4, sim1d noise/boundary/
robustness, sim2d, the baseline table) — every number checked matches its source.
Companion documents: `TODO.md` (the reordered path) and `REPO_REVIEW.md` (refactor plan).

---

## 1. The target

**AISTATS 2027: February 16–23, Montreal (confirmed). Submission deadline not yet
announced.** The 2026 cycle ran abstract Sep 25 / paper Oct 2 / supplementary Oct 9 (AoE),
8 pages + references/checklist/appendix, double-blind. Planning assumption until the CFP
lands: **abstract ~24 Sep, paper ~1 Oct 2026 — six to seven weeks from today.**

The submission is the DRI paper: recovery of a latent routing rule — orientation,
location, threshold, with calibrated uncertainty — from scalar query-only access, under a
bounded-curvature alternative. The positioning note
(`docs/related_work_positioning.md`) already states the defensible framing: **latent
membership** (vs GeoDA / active halfspace learning, where the side is observed),
**calibrated uncertainty**, **bounded curvature as an identifiability boundary rather
than a false-positive nuisance**. Query-efficiency must not be the claimed novelty.

Scope discipline for eight pages: **the paper is the estimator and its guard, not the
historical DRI architecture.** One sentence to write on the wall: *from query-only
perturbations, estimate the direction and location of a hidden hard-routing boundary,
with calibrated uncertainty, while distinguishing genuine boundary structure from smooth
confounders.* That is an AISTATS-shaped statistical problem; the three-phase
detect→recover→interpret framing is the research program, not the submission.

---

## 2. What we have going for us

### 2.1 The validated estimator core (rung 1, geometry supplied)

| Component | Result | Evidence |
|---|---|---|
| Distance from crossing fraction, d̂ = −σΦ⁻¹(π̂) | r = 0.996 vs truth, median rel. error −0.9%, flat over a 10× noise sweep | 13 Aug, `e1d59a1` |
| Boundary normal from separability | no detectable bias in all 20 cells; error is pure variance, pools away (20° at 1 anchor → <3° at 100) | 13 Aug, `b779879` |
| Pooling | geometric (C) > hard-threshold (B) > statistic-only (A); largest gap at Δ/τ = 1.5 — the real target's regime | 13 Aug, `37f54e3` |
| Stage-3 calibration | one parametric MC null through the pipeline: threshold **5.459 at m = 800**; honest surface fires 0.028–0.052 across the entire ladder sweep; the July 6.08 was a sample-size artifact (m = 400) | 14 Aug, `c8d9133` |
| t̂ interval, repaired | coverage **0.945–0.998** and **5× narrower** (0.127 vs 0.691 at Δ/τ = 1.5, N = 200) | 14 Aug |
| Axis dominance, reframed as an equivalence bound | certified upper bound tracks true tilt (0.87°/10.14°/20.13°/30.09°/45.12° vs 0/10/20/30/45); false-claim rate 0.000 beyond tolerance | 14 Aug |

The two scientific links — distance and direction of a hidden boundary are both
recoverable from query-only access with latent membership — are established in
simulation on planted gates. That is the core claim and it is real.

### 2.2 Real-target characterization (Experiment S)

- **The audit is invisible to the scaffold's OOD detector**: 83–98% of our probes are
  called "real", against 0% for the LIME-style family it was trained on. It learned one
  perturbation family and generalizes to none. (Scoped honestly by the Communities &
  Crime counterexample: on all-continuous data our probes are detected too.)
- **The gate is reachable, and our own density filter does not reject the probes that
  reach it** — the single pre-registered result that could have invalidated the approach,
  and it didn't fire.
- **Measured Δ/τ = 1.6–1.95** — at the detector floor. This made detection power a live
  constraint and pooling mandatory; it also fixes the honest scope sentence (below).

### 2.3 Supporting material (routing-audit thread, available as motivation/appendix)

Seattle case study (n = 3,581; precision 1.000 at the registered operating point, 652
flags; Δ̂ = 0.291 vs planted 0.300; honest flag rate 0.0075), the known-regimes
simulation (A11 sharp: FP 0.33–0.81 above the honest lengthscale vs 0.00–0.04 below;
kink control at exactly nominal 0.05), the baseline table including the SBM run with its
own code, and E1–E4 (trimming: clumpy FP 0.49→0.00, trend unmasking 0.14→0.90, power
flat in D at 0.74; OLS absorbs ~64% of the gap in closed form). `consolidated_draft_v3`
is a complete 8-page method note for that thread.

### 2.4 Process assets a reviewer will notice

Pre-registered predictions reported as hits and misses; retractions written down and
committed (`125555a` retracted by the Experiment L run); a provenance-tag discipline
(`[VALIDATED]/[DERIVED]/[PROPOSED]/[OPEN]/[LIMIT]`); a limits register ordered by
reviewer exposure; a clean 12-commit narrative. The failure documentation is unusually
good and is itself submission material.

---

## 3. What stands between this and a submission

### 3.1 The hole: screening has no confounder defence

The 14–15 August runs dismantled every instrument the screen was going to use:

1. **The dip is not the detector.** Its floor is π-dependent: ≈6σ separation needed at
   the π ≤ 0.10 operating point, against a real target at 1.6–1.95σ. Calibrating it buys
   no power and raises resonant FP 0.058 → 0.557.
2. **The calibrated LRT anti-selects the target.** It fires at **1.000 on resonant
   curvature against 0.203 on the real target** — screening on it enriches the confounder
   5× over the signal.
3. **The scaling exponent is out of any selection rule** (Experiment L). α is fittable at
   17% of gate anchors and 100% of curvature anchors, and stops discriminating once the
   ball spans the curvature wavelength (resonant α → 0.00). Structurally: the estimable
   window spans a factor of 2.4 in d/σ while a 3-rung ratio-2 ladder spans 4 — at most
   two rungs are ever simultaneously estimable, and only at ratio ≲1.55. The
   π_top = 0.35 "exact r⁰ recovery" is retracted: an honest surface reproduces it.
4. **The minimum-mass rule is deleted as an estimability gate** — it passes pure noise at
   0.990–0.995 and rejects real small minorities at 0.748.

What remains is the **separability guard** — a real gate's mixture components are
linearly separable in probe space, curvature's are spatially interleaved — and **it has
never been run.** It is now upstream of screen→select→recover, not downstream. This
audit reaches the same conclusion from the record that Robert's reordering states: the
guard test is the next experiment, it is cheap, and if it fails the screening pipeline
has no confounder defence at all — which is far better known before building the
pipeline than after.

### 3.2 Blocking gaps, ranked

1. **Separability guard untested** (above) — gates the paper's shape.
2. **screen→select→recover not started** — the binding end-to-end pipeline result.
3. **Boundary-seeking anchor placement does not exist**, and Experiment P showed
   geometric pooling is dead without it (0.96 by-design vs 0.00 random). A prerequisite
   for the recovery claim, not a deployment nicety.
4. **No end-to-end run at any realism rung.** Housing (rung 2, ~400 distinct response
   values) is the only end-to-end-able target today; the Slack scaffolds (rung 3, the
   genuine held-out test) expose 1–2 discrete values and need a discrete-response
   Stage 3 first.
5. **The pooled level also lacks a curvature defence** (P-2 falsified): long-wavelength
   curvature pools *coherently* — a 100%-confident false boundary with direction
   coherence R ≈ 0.879 on a surface with no boundary. The guard must be tested at this
   level too, not only per-anchor.
6. **Stage 0 has zero lines of code.** Every result is conditional on a supplied, valid
   probe geometry; the frame is supplied and exact throughout Experiment T, and the
   inflation from an estimated (local-PCA) frame is unmeasured.
7. **Coverage is narrow**: t̂ only, θ = 0, Gaussian noise, intrinsic d = 2, by-design
   placement. Coverage for n̂ and ĉ as objects is unmeasured.
8. **`boundary_recovery_v5` is not yet revised.** The change-list
   (`docs/method_revisions_2026-08-15.md`) is written but not applied — the method note
   currently states things known to be wrong (the bootstrap recipe, the min-mass rule,
   Stage 5(a), Stage 7, the S = 3 justification, the offset-agreement check).
9. **No AISTATS-format text exists.** Strong raw material (v5 + experimental report +
   designs + positioning note), zero paper.

### 3.3 Limits to state, not fix

The parallel penalty (invisible at every radius/dimension/projection); π ≈ 0.5 breaks
recovery though testing degrades gracefully; a ball spanning several routing cells lies
rather than abstains (the one silent failure); and the scope sentence: **at Δ/τ = 1.5 —
the housing scaffold's regime — 200 anchors certify the routing direction only to ≈13°,
so naming a feature needs Δ/τ ≥ 2.5.** Rung 4 (a deployed API, no ground truth) is where
credibility is spent, not earned — a deliverable of the paper, not evidence for it.

---

## 4. Housekeeping findings from this audit

- **`geospatial-xai-attacks`: the July 29 staged commit was never made.** The
  routing-audit reorganization is still sitting untracked (`configs/routing_audit.yaml`,
  `data/`, `docs/routing_audit/`, `scripts/routing_audit/`). Since `fd09b19` already
  recovered the thread into the DRI repo, decide deliberately: commit it there as
  archive, or delete the staging and let DRI own the thread.
- **The vault folder lags the repo by two weeks** and diverges silently: duplicated docs
  with different mtimes, four code files carrying dead sandbox paths
  (`build_poster.py`, `seattle_eui.py`, `diag_model_class.py`, `sim_sbm_baseline.py`),
  five superseded σ*-era figures sitting beside current ones, a stray lock file,
  `lu47x0a4c.tmp`, and the 15.5 MB raw CSV at the root. Recommendation: freeze
  `3 Stage Audit` as an archive with a pointer file; the repo is the single source of
  truth.
- **Repo nits** (detail in `REPO_REVIEW.md`): hardcoded `/tmp` paths in
  `exp_a_invariance.py` and `exp_p_pooling.py` break the fresh-checkout promise;
  README's headline still says "the dip test is the discriminator" while the current
  method note demotes the dip and makes the LRT the detector; `COMMIT_MSG*.txt` tracked
  in `sim/`; multi-MB per-anchor row CSVs tracked in git (2.2 MB
  `bootstrap_anchor_stats.csv` and friends) — fine today, a bad growth pattern.
- One supersession to be aware of: `plain_status.md` (morning of 14 Aug) lists
  "re-anchor the ladder" as a next step; the Experiment L run (15 Aug file) falsified
  re-anchoring — **keep κ = 0.78, do not re-anchor.** The plain-status item is stale.

---

## 5. Verdict

The estimator core is publishable-strength and unusually well-documented; the
identifiability framing (latent membership, bounded curvature) is distinctive and the
positioning against GeoDA/halfspace learning is already written. The single gating risk
is the confounder defence: every screening instrument except the never-run separability
guard has been eliminated, so the guard experiment decides which paper gets written —
"detect and recover with a working screen" or "estimator + identifiability boundary,
scoped." Six to seven weeks is enough for guard → screen→select→recover → housing
end-to-end → paper, but only if the scope is frozen by ~1 September and Stage 0, the
discrete-response Stage 3, and rung 3 are explicitly deferred. The guard is first
because it is cheap, it is load-bearing, and both of its outcomes change what gets
built next.

---

## 6. Reconciliation with the second audit (14 Aug)

A parallel audit (run against a zipped snapshot of the repo) independently reached the
same two headline conclusions: the guard is the next experiment, and the paper is the
narrow estimator-plus-guard, not the DRI architecture. Its methodology upgrades for the
guard experiment are adopted into `TODO.md` item 0: held-out AUROC / log-loss
improvement alongside balanced accuracy; a **within-anchor permutation null** for the
spatial-association score (the same null design Experiment P already uses), which turns
"is there halfspace structure" into a calibrated p-value rather than an accuracy
threshold; a pre-declared operating question (at 80–90% gate retention, what fraction
of the resonant confound passes?); and a wider confound panel (quadratic, GP ℓ ≈ σ,
kink, heteroskedastic honest). Its repo rules — exactly one implementation of each
estimator, imported by experiments; command-line entry points for every headline number;
no notebook-dependent results — are folded into `REPO_REVIEW.md` §4.

Where it is stale, because its snapshot predates the 13–14 Aug runs: its proposed steps
2–4 (normal recovery, Experiment P, interval coverage) have largely already run. The
normal estimator is validated unbiased *including* the true-labels-vs-estimated-
responsibilities ablation it asks for ("misassignment costs variance", `b779879`);
Experiment P ran, including the smooth-curvature-in-pooling cell (that is the P-2
falsification); and coverage ran for t̂ (0.00 as specified → 0.945–0.998 repaired). The
live deltas are the ones already in `TODO.md`: n̂/ĉ coverage for whatever the paper
actually reports, the guard at the pooled level, and estimated-frame sensitivity. One
supporting claim of its should not be relied on: "across-rung constancy is already
substantially better on the true gate" — Stage 5(b) is explicitly flagged for
re-examination in `method_revisions_2026-08-15.md` (the CV = 0.098 figure was measured
under the 1-D naive protocol with different geometry).
