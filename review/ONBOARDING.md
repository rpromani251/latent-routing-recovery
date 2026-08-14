# Onboarding — detect-recover-interpret

**For:** anyone joining the project (or returning to it after two weeks away, which in
this project is the same thing). **State as of:** 14 August 2026, repo commit
`c8d9133`, working tree clean. **Time to productive:** ~90 minutes of reading plus one
afternoon of hands-on. Companions: `STATUS.md` (readiness audit), `TODO.md` (the
ordered path), `REPO_REVIEW.md` (repo plan).

---

## 1. The problem

A company deploys a model. Secretly, it routes some inputs onto a different rule — if
you fall in some group, your score is docked by a fixed penalty. You are an outside
auditor with query access only: send inputs, read scalar outputs. No weights, no
gradients, no labels, no protected attribute.

**Can you prove the switch exists, find where it sits, and state the rule it uses —
with honest uncertainty?**

The goal is not "something looks suspicious" (published methods do that). The goal is a
sentence like:

> *It routes on feature 3 at threshold 4.98 ± 0.07, recovered from 100 anchors under
> query-only access, p < 0.05 against a permutation null.*

That is an **inverse problem, not a hypothesis test**, and the ± is what turns a number
into a claim. The motivating adversary is the Slack et al. (AIES 2020) scaffolding
attack — a biased model, an innocuous decoy, and an OOD detector routing between them
precisely so that no post-hoc explanation reveals the bias. That attack *is* hidden
routing, built on purpose.

What makes this different from things that look similar: in adversarial
boundary-finding (GeoDA) and active halfspace learning, every query tells you **which
side you're on** — membership is observed and the only question is where the boundary
is. Here **membership is latent, never observed**. Everything must be inferred from the
shape of a response distribution. That inference chain — scalar responses → latent
mixture responsibilities → boundary geometry → calibrated uncertainty — is the
contribution. (Full positioning: `docs/related_work_positioning.md`.)

## 2. The core trick — the one mental model to internalize

Pick a real data point (an **anchor**). Fire a few hundred slightly-perturbed queries
around it (**probes**). Look at the answers.

- Anchor well inside one region → every probe gets the same rule → one blurry cloud.
- Anchor near the hidden boundary → some probes land on the far side → those answers
  come back shifted by a constant → **two clumps**.

Three independent readings fall out of the split:

| what you look at | what it tells you |
|---|---|
| the **fraction** of probes that crossed (π̂) | **distance** to the boundary: d̂ = −σΦ⁻¹(π̂) |
| **where** the crossers sit in probe space | the boundary's **direction** (its normal) |
| the same at **many anchors**, pooled | one global hyperplane, **with error bars** |

Every other component in the project exists to stop that reasoning from being fooled —
by smooth curvature that fakes two clumps, by trends that eat the jump, by probes that
leave the data manifold, by noise that mimics structure. When you read any piece of the
pipeline and wonder why it exists, the answer is always "because without it, one of
those three readings lies."

## 3. Vocabulary

| Term | Meaning |
|---|---|
| **anchor** | a real data point you probe around; the anchor set A is fixed before any query |
| **probe** | a perturbed query x + δ near an anchor; m ≈ 800–1000 per rung |
| **π** | fraction of probes on the far side of the boundary (the mixture's minor weight) |
| **Δ** | the hidden penalty (jump size); **τ** = observation noise; **Δ/τ** = the one number that decides difficulty. The real scaffold measures **1.6–1.95** — low. |
| **rung / ladder / κ** | the procedure repeated at 3 probe radii (full/half/quarter); κ = σ_top/d positions it. The ladder was meant as a lie detector (a step looks identical at all radii, curvature doesn't) — see §6 for why that's now mostly dead |
| **LTS / trimming** | fit a plane to responses, drop the worst 25%, refit, residualize all points. Forces the fit onto the majority branch so the jump survives (plain OLS eats ~64% of it) |
| **dip test** | "one hump or two?" — assumes little, but π-dependent floor: inert below ≈6σ at π ≤ 0.10. No longer the primary detector |
| **LRT** | "one Gaussian or an equal-variance two-Gaussian mixture?" — the detector. Calibrated threshold **5.459** at m = 800. Sensitive, but fires on curvature too |
| **responsibilities** | EM's soft assignment of each probe to a mixture component — the estimated (never observed) membership labels |
| **separability guard** | the hinge: a real boundary's two components are **linearly separable in probe space**; curvature's are spatially interleaved. Its classifier's decision boundary IS the normal estimate — test and recovery are one computation. **Never yet run.** |
| **pooling** | combining per-anchor (normal, offset, weight) triples into one hyperplane; geometric mode-seeking beats statistic-only combination, most where Δ/τ = 1.5 |
| **axis dominance / t̂** | converting the pooled hyperplane into "routes on feature i above t̂" — an *equivalence* claim (certified upper bound on the off-axis angle), not a significance test |
| **coverage** | does "± 0.07" contain the truth ~95% of the time? Measured for t̂ (after repair: 0.945–0.998) |
| **A11 / A12** | the two checkable preconditions: honest model smooth at every probed scale / probe distribution locally smooth in the probe metric |
| **abstention (A0–A5)** | "I cannot see here" as a first-class output, distinct from "nothing here"; most codes fire before any query is spent |
| **rungs of realism** | 1 = planted gates in sim · 2 = **housing scaffold** (our build, real data, hidden router) · 3 = Slack's published scaffolds (someone else's, built to defeat audits) · 4 = a real API with no ground truth. Everything validated so far is rung 1. Rung 4 spends credibility, never earns it |

## 4. The history in five acts (why things are the way they are)

**Act 1 — the SBM thread (April–June, `geospatial-xai-attacks`).** Regime recovery via
behavioral-fingerprint similarity graphs + stochastic block models. Superseded: the
fingerprint cancels level effects, so the partition aligns with truth at ARI ≈ 0.05.
Kept as a baseline; the repo is now an archive.

**Act 2 — the founding result (24–28 July).** `routing_audit_v2` proposed detecting
via response *dispersion* across scales. The July 28 experiments killed that and
replaced it: **dispersion cannot discriminate a gate from curvature — identifiability,
not a bad statistic; modality (the response distribution) can.** Seattle case study:
n = 3,581 buildings, planted 0.30 penalty on tract demographics; precision 1.000 at the
registered operating point; Δ̂ = 0.291 recovered. Read: `docs/results_2026-07-28.md`.

**Act 3 — consolidation and stress-testing (29–31 July).** The four Peng deliverables
(consolidated note v3, known-regimes sim, baselines incl. SBM, reproducible repo).
E1–E4 extension experiments: the crossing law Φ(−d√D/σ) verified to 3 decimals;
trimming shown to fix three unrelated problems (clumpy-manifold FP 0.49→0.00, trend
unmasking 0.14→0.90, power flat in dimension); the θ = 0 parallel-penalty
identifiability limit. Then the 31 July independent review
(`docs/dri_detection_phase_working_doc.md`) falsified the plateau=A11 equivalence,
found the **resonance band** (curvature at ~1 oscillation per probe ball fakes a gate)
and the **masking attack** (resonant surface hides a real gate, 0.68→0.085), measured
the dip's weakness, adopted the LRT, and designed Experiments A, P, S.

**Act 4 — the estimand reframe and the validation sprint (12–14 Aug).**
`boundary_recovery_v5` reorganized everything around what is being *estimated*: mixture
weight → distance, separating direction → normal, pooling → one hyperplane with error
bars. Then five runs in three days: distance estimator validated (r = 0.996);
normal estimator unbiased (misassignment costs variance, not bias); Experiment P
(geometric pooling wins; **P-2 falsified** — smooth curvature pools into a confident
false boundary; random placement kills pooling, 0.96 vs 0.00); Experiment S (the
audit is invisible to the scaffold's OOD detector, 83–98% of probes called real; the
gate is reachable; Δ/τ = 1.6–1.95); calibration (the prescribed bootstrap destroys the
test; one parametric MC null through the pipeline, threshold 5.459); Experiment T
(axis-dominance rule broken → equivalence bound; t̂ interval coverage 0.00 as
specified → 0.945–0.998 repaired, 5× narrower).

**Act 5 — the ladder result (14–15 Aug). You are here.** Experiment L: the estimable
window (factor 2.4 in d/σ) cannot contain a 3-rung ratio-2 ladder (factor 4); the
scaling exponent α is fittable at 17% of gate anchors and 100% of curvature anchors and
stops discriminating anyway; one Experiment A cell retracted (an honest surface
reproduces the "exact r⁰ recovery"). Net: **the multi-scale defence against curvature
is out of the selection rule, the calibrated LRT anti-selects the real target 5:1
(1.000 on resonant vs 0.203 on the target), and the never-run separability guard is now
the only remaining confounder defence — promoted upstream of screen→select→recover.**
Read: `sim/RESULTS_ladder_2026-08-15.md` and `docs/method_revisions_2026-08-15.md`.

## 5. The pipeline today, with status flags

```
Stage 0  probe geometry (GRIDE, plateau, tangent frame, radius)   [SPECIFIED — ZERO CODE.
         density filter (on-manifold scope)                        All results conditional
                                                                   on supplied geometry]
Stage 1  probe generation (tangent-frame isotropic, per-coord      [VALIDATED design rules;
         scale σ = r/√d̂, ladder κ = 0.78 — do NOT re-anchor]       ladder role reduced by L]
Stage 2  trimmed residualization (LTS, h = 0.75m)                  [VALIDATED, load-bearing]
Stage 3  detection: calibrated LRT (5.459) primary; dip kept       [CALIBRATED 14 Aug;
         on tabled p-value, secondary                              fires on curvature too]
Stage 4  per-anchor estimates: π̂ → d̂ = −σΦ⁻¹(π̂) (distance);       [VALIDATED r = 0.996 /
         separability classifier → normal                          unbiased]
  ⚠      separability GUARD (gate vs curvature discriminator)     [UNTESTED, LOAD-BEARING —
                                                                   next experiment]
Stage 5  consistency checks                                        [MOSTLY RETRACTED:
         (α exponent, min-mass, offset agreement)                  see method_revisions]
Stage 6  pooling → one hyperplane (repaired: project boundary      [VALIDATED; needs
         points onto pooled normal; distance from crossing law)    by-design placement]
Stage 7  statement: equivalence bound on angle to candidate        [VALIDATED as bound;
         axis; t̂ with covering interval                            coverage 0.945–0.998]
```

Two prerequisites sit outside the stages: **boundary-seeking anchor placement** (does
not exist; pooling is dead without it) and the **discrete-response Stage 3** (needed
before the rung-3 Slack targets, which expose only 1–2 distinct values).

## 6. Epistemic state — what is known vs believed

This project's house rule is that the tags matter: `[VALIDATED]`, `[DERIVED]`,
`[PROPOSED]`, `[OPEN]`, `[LIMIT]`. Before citing any claim, know its tag.

**Required reading — the retraction list.** The most important onboarding step is
knowing what was believed and is now dead, because the documents that proposed these
still exist and read persuasively:

1. Plateau = A11 equivalence — falsified both directions (31 Jul).
2. Quadratic lack-of-fit repair — uncalibratable dead end, recorded so it isn't
   re-proposed (31 Jul).
3. The prescribed bootstrap calibration — inflates the null ~100×; replaced by one
   parametric MC null (14 Aug).
4. The minimum-mass estimability rule — passes pure noise at 0.99; deleted (14 Aug).
5. The α scaling-exponent filter and the S = 3 justification — out of any selection
   rule (15 Aug); the Experiment A π_top = 0.35 cell — retracted (15 Aug).
6. Axis-dominance permutation rule and the bootstrap-stability repair — both unusable;
   replaced by the equivalence bound (14 Aug). ("Stability is not validity.")
7. The offset-agreement check — one of its own arms is biased (14 Aug).
8. Re-anchoring the ladder — falsified the day it was proposed (15 Aug); κ stays 0.78.

**Structural limits (not fixable, state them):** a penalty parallel to within-branch
variation is invisible; π ≈ 0.5 breaks recovery (LTS has no majority to commit to); a
ball spanning several routing cells returns a confident wrong answer and nothing
abstains — the one failure that lies rather than declines; and at Δ/τ = 1.5, 200
anchors certify direction only to ≈13°, so feature-naming needs Δ/τ ≥ 2.5.

**Numbers you will hear in every discussion:**

| number | what it is |
|---|---|
| 0.996 / −0.9% | distance estimator correlation / median relative error |
| 5.459 | calibrated LRT threshold at m = 800 (5.62 at m = 400 — calibrate at deployed m) |
| 1.000 vs 0.203 | calibrated LRT fire rate: resonant curvature vs the real target — the crisis |
| 0.96 vs 0.00 | pooling success: by-design vs random anchor placement |
| 1.6–1.95 | the housing scaffold's measured Δ/τ |
| 83–98% | our probes the scaffold's OOD detector calls "real" |
| 0.945–0.998, 5× | repaired t̂ interval: coverage, narrowing |
| 13.5° @ Δ/τ=1.5, N=200 | certifiable angle bound in the target's regime |
| 0.49 → 0.00 | clumpy-manifold honest FP, raw → trimmed |
| ≈6σ | dip detection floor at the π ≤ 0.10 operating point |
| 2.4 vs 4 | estimable window vs 3-rung ladder span — why the ladder can't fit |

## 7. Reading order (~90 minutes)

1. `docs/plain_status.md` — the honest status map in plain language (15 min). Note:
   its "re-anchor the ladder" item is stale — Experiment L killed it hours later.
2. This file's §4–§6 again, now that the names mean something (5 min).
3. `docs/boundary_recovery_v6.pdf` — **the current method note and the theory backbone
   of this onboarding.** v6 (14 Aug) applies the 14–15 Aug change-list in full; retired
   v5 claims stay visible under `[RETIRED]` tags rather than being silently removed, so
   the note doubles as the falsification record. v5 and
   `method_revisions_2026-08-15.md` remain in `docs/` as the paper trail. (40 min)
4. `sim/RESULTS_ladder_2026-08-15.md` — the result that reordered everything (10 min).
5. `docs/related_work_positioning.md` — how to say what this is (10 min).
6. `docs/experimental_report.pdf` — every run with outcomes vs predictions (skim, 10 min).

Deeper, as needed: `docs/dri_detection_phase_working_doc.md` (the 31 July review — the
best single document for *why* the current design is shaped this way);
`docs/results_2026-07-28.md` (the founding Seattle thread);
`docs/routing_audit_probe_geometry_consolidated.md` (E1–E4 + probe geometry, fully
provenance-tagged); `review/STATUS.md` and `review/TODO.md` (where this is all going).

## 8. Hands-on, day one

```bash
git clone <detect-recover-interpret>   # or work in the existing checkout
pip install -r requirements.txt        # numpy/pandas/matplotlib; diptest optional
                                       # (seeded MC fallback built in)
cd sim
python3 exp_distance_estimator.py      # smallest real experiment; minutes
python3 exp_ladder_anchor.py           # ~11 min on 2 cores — reproduces Act 5
```

Gotchas, current as of today: `exp_a_invariance.py` and `exp_p_pooling.py` have
hardcoded `/tmp` paths and won't run from a fresh checkout (fix queued);
`sim_sbm_baseline.py` needs the sibling `geospatial-xai-attacks` repo on the path
(`DRI_GEOXAI_REPO`); Seattle pipeline needs `data/` populated per
`docs/data_dependencies.md`; long runs are resumable via `_parts_*/` directories —
re-invoke the same command after interruption. The vault folder `3 Stage Audit` is a
frozen snapshot of late July — **never work from it; the repo is the source of truth.**

## 9. Where to contribute right now

In priority order (details and pre-registrations in `review/TODO.md`):

1. **Experiment G — the separability guard** (`TODO.md` item 0). The whole project
   currently pivots on an untested component; the harness, threshold, and surfaces all
   exist. This is the highest-leverage week of work available.
2. **Review `boundary_recovery_v6`** — the change-list is applied in draft
   (`docs/boundary_recovery_v6.tex`, 14 Aug); it needs the author's read-through before
   any paper text is drafted from it.
3. **Hygiene**: the `/tmp` paths, README top rewrite, `DECISIONS.md` ledger seeded from
   the retraction list above (`REPO_REVIEW.md` has the plan).
4. **Boundary-seeking anchor placement** — design work; nothing exists; pooling needs it.
5. After the guard verdict: **screen→select→recover end-to-end**, then the housing
   scaffold (rung 2) — the first run of the whole pipeline against a model with a
   router the audit cannot see.

## 10. House rules

- **Pre-register.** Predictions and falsification conditions are written before the
  run, and results are reported against them, hits and misses both.
- **Retract in writing.** A wrong claim gets a dated retraction in a RESULTS file and
  (soon) a `DECISIONS.md` line — never a silent edit.
- **Every number traces.** Claim → RESULTS file → CSV → seed + commit. If you can't
  trace it, don't cite it; if you produce it, make it traceable (the reproduction-block
  format in any `sim/RESULTS_*.md` is the template).
- **Tags travel with claims.** `[PROPOSED]` is not `[VALIDATED]`, and the difference
  has bitten this project more than once.
- **Abstention is an answer.** "I cannot see here" ≠ "nothing here" — in the method,
  and in your own write-ups.
- **The failure record is an asset.** This project's falsifications are its strongest
  evidence of rigor. Write yours up with the same care as the wins.
