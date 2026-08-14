# Plan — poster print deadline (Wed Sep 9, 11:59 PM) and sponsor day (Tue Sep 22)

**Date:** 14 August 2026. This is the operating schedule; `TODO.md` holds the science
ordering it implements. Three deliverables, one pipeline:

1. **MITEI poster** — printed from results frozen by Sep 4, submitted Sep 8, presented
   to energy sponsors Sep 22.
2. **The paper** — a complete, named, arXiv-able draft in hand on Sep 22: the artifact
   sponsors pass to their analysts and statisticians, and the reason you know every
   number cold.
3. **AISTATS submission** (~Oct 1, est.) — the anonymized, formatted version of (2).
   Nothing is written twice; the poster is the paper's figures with an energy-first
   frame, and the sponsor-day prep *is* the writing of the paper.

---

## The gates

| Date | Gate |
|---|---|
| **Fri Aug 21** | Guard verdict (Experiment G) — decides the method claim on poster and paper |
| **Fri Aug 28** | Poster story locked (which panels, which figures) |
| **Fri Sep 4** | **Poster content freeze.** Hard rule: a number not measured by Sep 4 does not print |
| **Tue Sep 8** | Final poster PDF submitted — a full day before the deadline, never the night of |
| **Wed Sep 9, 11:59 PM** | Print deadline (external, immovable) |
| **Tue Sep 15** | Last result that can enter the paper |
| **Fri Sep 18** | **Paper freeze.** Weekend is buffer, not workspace |
| **Mon Sep 21** | Copies printed, QR link live, pitch rehearsed |
| **Tue Sep 22** | Sponsor presentation, paper in hand |
| ~Oct 1 (est.) | AISTATS deadline — polish window Sep 23–Oct 1 |

---

## Week by week

### Week 1 — Fri Aug 14 → Fri Aug 21: the guard, plus poster logistics

Science (unchanged from `TODO.md` item 0): **Experiment G**, the hygiene items, and your
read-through of `boundary_recovery_v6`.

Poster logistics, cheap and done early so Sep 8 has no surprises:

- Confirm print specs and the submission portal/process from the MITEI email now — the
  template is **34 × 24 in, three columns** (`MITEI Poster Template_2026.pptx`); verify
  that's still the 2026 spec and whether they want PDF or PPTX.
- Move the poster assets into the repo (template, `ROMANI_ROBERT.pptx` as the July
  baseline, `POSTER_GUIDE.md`, `build_poster.py`, `make_poster_figures.py`) — the July
  build pipeline exists; make sure it runs from a fresh checkout.
- **Gate, Aug 21: the guard verdict.** Both outcomes are presentable — see
  contingencies — but the poster copy differs, so this lands before the story lock.

### Week 2 — Aug 24–28: screen→select→recover, and the story lock

- Science: build and run **screen→select→recover** on the synthetic suite with the
  guard verdict inside the selection rule; the anchor-placement rule gets built and
  validated within it (`TODO.md` items 1–2).
- **Paper skeleton** (half a day, now): section headers mapped to existing sources —
  intro/related from `related_work_positioning.md`, method from v6, experiments from
  the RESULTS files, limitations from the limits register. From here on, every
  experiment writes its paper subsection the day it finishes, while it's warm.
- **Poster story lock (Fri Aug 28).** Recommended shape — *evolve* the July poster, do
  not restart:
  - **Left column (keep ~as is):** the Seattle energy story — retrofit eligibility, the
    26%-more-efficient-than-reality harm, black-box constraint. It's the hook this
    audience needs, and it's finished, verified material.
  - **Middle column (upgrade):** from *detection* to *recovery*. The two-clumps
    mechanism figure stays; add the estimator story — crossing fraction → distance,
    separating direction → boundary normal, pooling → one rule with error bars — and
    the target sentence ("routes on feature $i$ at threshold $\hat t \pm$ …") as the
    centerpiece claim.
  - **Right column (upgrade):** honest uncertainty. The conformal-coverage figure stays
    (strongest caption on the July poster); add interval coverage 0.945–0.998 and the
    guard result as "how we know two clumps means a boundary and not curvature." Keep
    the sponsor applications box (vendor models, physics-informed hybrids, OOD
    fallbacks) — it is exactly what this audience passes to their analysts. Keep the
    caveats box; sponsors trust posters that state limits.
- New figures needed (draft list, finalize at lock): estimator schematic (the two TikZ
  panels from v6 §2 redrawn for poster scale), guard result panel (from Experiment G),
  pooling/coverage panel (`fig_p_pooling.png` + the coverage table as a graphic).

### Week 3 — Aug 31 → Fri Sep 4: housing attempt, poster draft, content freeze

- Science: **housing scaffold end-to-end** (rung 2) — the stretch item. If it lands by
  Sep 4, it's the poster's strongest panel ("the full pipeline ran against a model with
  a router we couldn't see"). If not, the synthetic end-to-end table prints and housing
  continues for the paper. Decide by the freeze, not at the printer.
- Poster full draft by **Wed Sep 3**; to Peng for async feedback Sep 3 (he may be
  mid-relocation — request comments by Sep 7, proceed regardless).
- **Fri Sep 4: content freeze.** Also the scope freeze from `TODO.md` — the paper's
  defer list is final today.

### Week 4 — Mon Sep 7 → Wed Sep 9: print

- Sep 7: incorporate feedback; slot-in decision on the housing panel; full-size proof
  read (print a quarter-scale test page; check figure legibility at 1 m).
- Sep 8: export final PDF at 34 × 24, **submit**. The deadline is Sep 9 at 11:59 PM;
  submitting a day early costs nothing and absorbs portal failures.
- Sep 9: buffer only.

### Sep 9–18: the paper sprint

The poster is done; everything is the paper now. Suggested order (sections with living
uncertainty first):

- Sep 9–11: experiments section — the confound/null table, screen→select→recover
  operating characteristic, estimator validation, coverage. Every number traces to a
  RESULTS file + commit (house rule).
- Sep 12–14: method section from v6 (it's already written to be lifted); intro +
  related work from the positioning note; the scope sentence (13.5° at Δ/τ = 1.5) goes
  in the claims, not buried.
- **Sep 15: last result in.** Housing, if still running, enters here or is reported
  in-progress — honestly labeled either way.
- Sep 16–17: **red-team pass.** Read as the two reviewers: "how do you know two-regime
  structure isn't smooth misspecification?" (the guard + calibration answer) and "can
  you actually recover the boundary with calibrated uncertainty?" (P + coverage
  answer). Fix what fails; also run the reproducibility check — one command per figure.
- **Fri Sep 18: freeze.** A complete, named PDF — title, abstract, 8-ish pages +
  appendix. This is the hand-out artifact; the AISTATS anonymization happens after
  Sep 22.

### Sep 21–22: sponsor day

- Sep 21: print ~15–20 paper copies; put the paper at a stable link (repo release or
  arXiv) and a **QR code on or beside the poster** — "share with your analysts" works
  much better as a scan than a promise; rehearse the pitch.
- Three tiers, scripted and timed:
  - **30 s (any sponsor):** deployed energy models can carry hidden routing rules;
    aggregate accuracy stays clean while one group is quietly penalized; we recover the
    hidden rule — direction, location, threshold, with error bars — from queries alone.
  - **3 min (interested sponsor):** + the Seattle story and the two-clumps mechanism,
    the target sentence, the applications box, "here's the paper for your team."
  - **10 min (their statistician):** + latent membership vs label-based boundary
    finding, the curvature identifiability problem and the guard, calibration and
    coverage numbers, honest limits. This tier is why the paper must be frozen — it's
    the conversation you most want to be right for.
- Q&A prep sheet (draft Sep 21 from the ONBOARDING cheat-sheet): energy side — "does
  this happen in real vendor models?", "what would this cost to run against ours?",
  "what access do you need?" (answer: query access + a sample of real inputs, nothing
  else); stats side — curvature confound, why coverage is measured not assumed, the
  Δ/τ ≥ 2.5 feature-naming threshold, what's deferred (Stage 0, discrete responses).

---

## Contingencies

| Risk | Response |
|---|---|
| **Guard fails (Aug 21)** | Both artifacts survive with the reframed story (`TODO.md` item 0 decision rule): the poster's method panel becomes estimator + characterized identifiability boundary — for sponsors this is nearly indistinguishable in strength; the paper reframes as planned. Write poster copy from Aug 28 so that either verdict slots in. |
| **Housing misses Sep 4** | Poster prints synthetic end-to-end + Experiment S panels (invisible-to-detector, gate reachable, Δ/τ = 1.6–1.95 measured on a scaffold). Housing continues for the paper (Sep 15 cutoff). |
| **Housing misses Sep 15** | Paper reports it as designed-and-running with the S characterization; not fatal — the estimator + guard + coverage story stands on the synthetic suite. |
| **Peng unavailable** | Proceed on async materials, same as July. Nothing in this plan blocks on a meeting. |
| **Print portal trouble Sep 8** | That's why submission is Sep 8, not Sep 9. Escalate to the MITEI contact confirmed in week 1. |
| **AISTATS CFP lands mid-plan** | Check dates/format the day it posts; a named Sep 22 hand-out is compatible with double-blind review under the 2026-cycle rules (preprints allowed; no social-media promotion during review) — re-verify in the 2027 CFP. |

## Two standing rules

The poster prints only measured numbers — nothing contingent on an experiment still
running, no `[PROPOSED]` content presented as result; the caveats box is a feature.
And the paper-writing *is* the talk prep: if a section was hard to write, that's the
question you'll get on Sep 22 — bring it to the red-team pass, not to the sponsors.
