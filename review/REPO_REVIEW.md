# Repo review — `detect-recover-interpret` refactor plan

**Date:** 14 August 2026. Audited at commit `c8d9133`, working tree clean.
Companion to `STATUS.md` and `TODO.md`.

---

## 1. What is already good — keep it

- **The commit narrative.** Twelve commits that read as a research log, including a
  retraction commit. Do not squash or rewrite history in the refactor.
- **Pre-registration discipline**: `experiment_designs.pdf` states predictions before
  runs; `RESULTS_*.md` files carry reproduction blocks (command, runtime, seed, outputs,
  resumable `_parts_*` convention).
- **Provenance tags** (`[VALIDATED]/[DERIVED]/[PROPOSED]/[OPEN]/[LIMIT]`) and the
  `legacy/` directory with its own README explaining what was superseded and why.
- **`.gitignore` is right**: data and results properly excluded, `data_dependencies.md`
  documents the download path.
- The README's document map (current vs superseded, with one-line reasons).

## 2. The three structural problems

**(a) `src/` implements the superseded method.** The package
(`audit.py`, `dispersion.py`, `probes.py`, …) is the July routing-audit protocol; the
current method — fast EM, calibrated LRT, crossing-law distance, normal estimator,
pooling, equivalence bound — lives as ~20 flat scripts in `sim/`. The README admits the
gap ("nothing in src/ implements Stage 0 … or the LRT") and notes the detection-review
numbers came from a reimplementation that was not preserved. The current
`sim/fast_em.py` etc. re-implement those pieces; nothing should ever cite the
unpreserved code again.

**(b) `sim/` is two generations plus their outputs.** Generation 1 (known-regimes
suite, `sim1d/2d_*`, baselines, `ext1–4`) and generation 2 (the Aug 13–15 DRI
experiments) share one flat directory with their row-level CSVs committed:
`bootstrap_anchor_stats.csv` 2.2 MB, `ladder_anchor_rows.csv` 1.6 MB,
`normal_estimator_probelog.csv.gz` 1.5 MB, `normal_estimator_rows.csv` 0.85 MB,
`distance_estimator_rows.csv` 0.5 MB. Tolerable today; a bad growth pattern with the
guard, screen, and housing runs coming.

**(c) Fresh-checkout reproduction is broken in two places.** Hardcoded `/tmp` paths in
`exp_a_invariance.py` and `exp_p_pooling.py` (`paths.py` exists and is unused there);
`COMMIT_MSG*.txt` drafts tracked in `sim/`; README headline still says the dip is the
discriminator while v5 makes the LRT the detector and the dip secondary.

## 3. Target layout

Evolution, not rewrite — file moves with history intact (`git mv`), no logic changes
mixed into move commits.

```
detect-recover-interpret/
├── README.md                     # rewritten top: v5 estimand framing, status table,
│                                 # quickstart, document map (see §5)
├── pyproject.toml                # replaces requirements.txt; extras: [dip], [sbm]
├── configs/                      # unchanged; plus one YAML per experiment,
│                                 # registered before the confirmatory run
├── src/detect_recover_interpret/
│   ├── core/                     # NEW — promoted from sim/ scripts once the guard/
│   │   ├── em.py                 #   screen experiments settle the API:
│   │   ├── lrt.py                #   fast_em, calibrated LRT + threshold-at-m,
│   │   ├── calibration.py        #   dip (MC fallback), LTS, crossing law,
│   │   ├── guard.py              #   separability guard, pooling (repaired
│   │   ├── lts.py                #   estimator), equivalence bound
│   │   ├── crossing.py
│   │   ├── pooling.py
│   │   └── bounds.py
│   ├── seattle/                  # the gen-1 routing-audit code, moved intact
│   │   ├── audit.py  model.py  probes.py  dispersion.py
│   │   ├── spatial_randomization.py  conformal.py  …
│   └── legacy/                   # unchanged
├── experiments/                  # NEW — one directory per experiment, replacing flat sim/
│   ├── 00_known_regimes/         # sim1d/2d suite + baselines + SBM
│   ├── 01_ext_e1e4/              # ext_core, ext1–4, run_ext
│   ├── 02_distance/  03_normal/  04_s_scaffold/  05_p_pooling/
│   ├── 06_a_invariance/  07_calibration/  08_axis_that/  09_ladder/
│   ├── 10_guard/                 # next (TODO item 0)
│   ├── 11_screen_select_recover/
│   └── _template/                # run.py · config.yaml · PREREG.md · RESULTS.md · outputs/
├── results/                      # row-level CSVs land here (git-ignored, as now);
│                                 # each experiment keeps only summary CSVs (≲100 KB) in git
├── docs/
│   ├── current/                  # boundary_recovery_v5 (REVISED per method_revisions),
│   │                             # experimental_report, experiment_designs,
│   │                             # related_work_positioning, plain_status
│   ├── archive/                  # v2, v3, derivation_guide, probe_policy_spec,
│   │                             # 31 July working doc, July logs, future_work
│   └── DECISIONS.md              # NEW — one line per retraction/repair:
│                                 # date · claim · verdict · commit · evidence file
├── paper/aistats2027/            # NEW — the submission
└── tests/                        # smoke tests: tiny-n run of each experiment entry
                                  # point + a calibration-reproducibility check
```

**Migration map (the non-obvious moves).** `sim/fast_em.py` → `core/em.py`;
`sim/dip.py` → `core/` (keep the MC-fallback); the `exp_*.py` scripts move into their
experiment directories with their `RESULTS_*.md` and summary CSVs; row-level CSVs
(`*_rows.csv`, `*_anchor_stats.csv`, probelogs) move to `results/` and out of git
(keep one archival tag, e.g. `pre-restructure`, so the committed copies stay
retrievable); `COMMIT_MSG*.txt` deleted; gen-1 suite moves wholesale to
`experiments/00_known_regimes/` with `README_SIM.md` as its README.

## 4. Conventions going forward

- **Exactly one implementation of each estimator.** The separability guard, EM/mixture
  fit, calibrated LRT, crossing law, pooling, and equivalence bound live in `core/` and
  experiments import them. An experiment script carrying its own slightly different
  copy is a bug; the flat `sim/` is one commit away from that failure mode today.
- **One query interface for every target.** Synthetic surfaces (gate, resonant,
  quadratic, GP, kink, heteroskedastic), the housing scaffold, and any future rung-3
  model implement the same `query(X) → y` protocol, so the confound suite and the real
  targets run through a single harness.
- **Entry points, not notebooks.** Every headline number regenerates from one command
  (`make e10-guard`, `make fig2`, or `python -m experiments...`); no result depends on
  running notebook cells in order.
- **One experiment = one directory** with `config.yaml` (registered before the run),
  `PREREG.md` (predictions + falsification conditions, written first), `run.py`
  (seeded, resumable), `RESULTS.md` (the current house style is already right: outcome
  vs predictions, reproduction block), and `outputs/` (summaries only).
- **Size discipline**: summary CSVs in git; anything row-level to `results/`
  (git-ignored). If row-level artifacts must be shared, use git-lfs or a release asset.
- **Paths**: everything through `paths.py` env-var convention; a smoke test that runs
  one tiny experiment from a fresh clone is the enforcement mechanism (GitHub Actions,
  numpy-only, minutes).
- **DECISIONS.md is the falsification ledger.** The project's retractions are currently
  legible only by reading four RESULTS files and a change-list; one table makes the
  method's audit trail — a genuine reviewer asset — discoverable.
- **At submission**: tag the repo; produce the anonymized code appendix from the tag
  (AISTATS is double-blind — strip names, the advisor reference, and the sibling-repo
  link from the export, not from the repo).

## 5. README rewrite (top only)

Lead with the v5 claim — the target sentence ("routes on feature 3 at threshold
4.98 ± 0.07 … under query-only access") and the estimand framing (weight → distance,
separating direction → normal, pooling → one hyperplane with error bars). Then a
one-screen status table: validated / falsified-and-repaired / untested-and-load-bearing
(the guard) / deferred (Stage 0). The July 28 "dispersion vs modality" result moves to
a "founding result" paragraph. Everything below the fold (Seattle numbers, sim suite,
baselines) is already accurate — keep.

## 6. Timing — refactor after the guard, hygiene now

Do now (half a day, `TODO.md` item 6): `/tmp` fixes, README top, `COMMIT_MSG` cleanup,
v5 revision, `DECISIONS.md` seeded from the four existing retractions. Do **after** the
guard and screen experiments (weeks 2–3): the `src/core/` promotion and the
`experiments/` restructure — the library API should crystallize around whatever the
selection rule turns out to be, not ahead of it. The restructure is 1–2 days when it
happens and must not compete with Experiment G for this week.

## 7. The sibling repo and the vault

- **`geospatial-xai-attacks`**: still carries the never-committed July 29 staging
  (`configs/routing_audit.yaml`, `data/`, `docs/routing_audit/`,
  `scripts/routing_audit/` — all untracked). Decide once: commit as archive (message
  already drafted in the July 29 status) or delete the staging, since `fd09b19`
  recovered the thread into DRI. Either way, the repo's role is archive: Phases 1–2 +
  SBM thread, README cross-links (already present at `5ebf68e`), no further investment.
- **The `3 Stage Audit` vault folder**: freeze with a `POINTER.md` naming the repo as
  the single source of truth. Its four dead-sandbox-path code files and five
  superseded σ*-era figures are the historical record — leave them, but stop editing
  vault copies of documents that also live in `docs/` (they have already diverged).
