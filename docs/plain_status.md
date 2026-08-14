# Where this project actually stands — in plain language

*Written 14 August 2026. A companion to `boundary_recovery_v5.pdf`, not a replacement:
that document is the method, this one is the honest status map. If the two disagree,
the numbered results in `experimental_report.pdf` and the commit log are authoritative.*

---

## 1. What we are trying to do

A company deploys a model. Secretly it routes some inputs onto a different rule — if
you fall in some group, your score gets docked by a fixed amount. You are an outside
auditor. You can send inputs and read outputs. That is all: no weights, no gradients,
no internals, no labels.

**Can you prove the switch exists, find where it sits, and state the rule it uses?**

The goal is not "something looks suspicious here" — published methods already do that.
The goal is a sentence like:

> *It routes on feature 3 at threshold 4.98 ± 0.07, recovered from 100 anchors under
> query-only access, p < 0.05 against a permutation null.*

That is an **inverse problem**, not a test. It is also why the ± matters so much: it is
what turns a number into a claim.

## 2. The core trick

Pick a real data point — an **anchor**. Fire a few hundred slightly-perturbed queries
around it — **probes**. Look at the answers.

- Anchor well inside one region → every probe gets the same rule → the answers vary
  smoothly. One blurry cloud.
- Anchor near the boundary → some probes land on the far side → those come back shifted
  by a constant, the penalty → the answers split into **two clumps**.

Three things fall out of that split, and each says something different:

| what you look at | what it tells you |
|---|---|
| what **fraction** of probes crossed | how **far away** the boundary is |
| **where** the crossers sit in input space | which **direction** the boundary faces |
| doing this at **many anchors** and combining | one boundary, with error bars |

Everything else in the pipeline exists to stop that reasoning from being fooled.

## 3. The vocabulary, decoded

**π (pi)** — the fraction of probes that landed on the far side. 10% crossed means the
boundary is comparatively far; 40% means it is close.

**Δ (Delta)** — how big the hidden penalty is. **τ (tau)** — the noise level. **Δ/τ**
is signal-to-noise, and it is the single number that decides whether detection is hard.
The real attack measures **1.6–1.95**, which is low.

**Rung / ladder** — the whole procedure is repeated at three probe radii (full, half,
quarter). A real boundary looks identical at every radius; ordinary curvature does not.
The ladder is a lie detector, not extra data.

**Trimming (LTS)** — before testing, fit a plane through the responses to strip out the
ordinary smooth trend. Plain fitting splits the difference between the two clumps and
eats about 64% of the signal. Throwing away the worst-fitting 25% first forces the plane
onto the majority side, and the jump reappears at full size.

**The two tests** — the **dip test** asks "does this have one hump or two?" The **LRT**
asks "is one bell curve or two bell curves the better description?" The LRT is more
sensitive; the dip assumes less and survives curvature better.

**Separability guard** — the important one. Two clumps in the *answers* is not enough;
ordinary curvature produces that too. A real boundary is a **plane**, so the two clumps
must also be separated by a plane in **input** space. Curvature produces clumps that are
spatially interleaved. **And the plane that separates them is the estimate of the
boundary's direction** — the test and the recovery are the same computation. That is the
hinge of the entire method.

**Pooling** — one anchor gives a noisy local guess. Combining many gives one boundary.
The point is that per-anchor power stops being the figure of merit; the number of
anchors needed to recover the boundary takes its place.

**Axis dominance / t̂** — pooling returns the boundary as a weighted mix of all features.
Axis dominance asks whether one feature dominates. If it does, you can say *"it routes
on income above 4.98"*; **t̂** is that number. If not, you report a hyperplane — true,
but far less useful.

**Coverage** — when you write "4.98 ± 0.07", is the truth inside that range 95% of the
time? Never checked. The ± is currently unbacked.

**Anchor placement** — *which* real points you probe around. Turns out not to be a
detail: see §6.

## 4. What is confirmed

Measured against known ground truth, in simulation, on gates we planted:

- **Distance recovery works.** From the crossing fraction alone: r = 0.996 against
  truth, median error −0.9%, flat across a 10× noise sweep.
- **Direction recovery is unbiased.** This matters more than it sounds — bias survives
  averaging, randomness does not. The error is pure randomness in all 20 test cells, so
  20° at one anchor becomes under 3° at 100 anchors.
- **Pooling on geometry beats thresholding**, and by the widest margin at Δ/τ = 1.5 —
  exactly where the real target sits.
- **The audit is invisible to the real attack's detector.** It calls 83–98% of our
  probes "real", against 0% for the perturbation family it was trained on. It learned
  one attack and generalises to none.
- **The gate is reachable on the real target**, and our own on-manifold filter does not
  reject the probes that reach it. This was the single result that could have
  invalidated the approach.
- **Trimming is load-bearing**: honest false positives 0.49 → 0.00 on clumpy data.
- **Dispersion cannot discriminate; modality can.** The founding result.
- **The Stage-3 threshold is calibrated** (14 August): one number, 5.46, computed once —
  not a per-anchor bootstrap and not per-rung.

## 5. What has been proven wrong

The failures have been more productive than the confirmations. Each one changed
something:

- Orientation error was predicted to be best at middling π. It is best at **low** π —
  which inverted a whole experiment's design axis.
- The real attack was predicted at Δ/σ > 2.5. It is **1.6–1.95**, at the detector's
  floor. Detection power became a live constraint and pooling stopped being optional.
- Pooling was predicted to reject smooth curvature on its own. **It does not.**
  Long-wavelength curvature pools *coherently* and yields a 100% confident false
  boundary on a surface containing no boundary at all.
- The permutation null cannot test existence — it is invariant to the statistic by
  construction. It tests *geometry*, which is still useful, but it needed a second null.
- (14 August) The prescribed bootstrap recipe destroys the test rather than calibrating
  it; the minimum-mass rule is an anti-gate; the dip's floor is roughly three times
  worse than stated at the operating point we actually use.

## 6. What is still only on paper

This is the part worth being blunt about.

- **Stage 0 does not exist.** Choosing the probe radius and the local frame
  automatically is fully specified and has **zero lines of code**. Everything measured so
  far hands the method a valid geometry. Deliberately deferred, not dropped — but until
  it exists, every result is conditional on "given a valid local probe geometry."
- **The separability guard has never been tested.** It is the piece covering the LRT's
  blind spot, and the 14 August run showed curvature fires the LRT at up to **1.000** at
  the top rung. The guard is now carrying the load, untested.
- **Coverage has never been measured.** The ± in the headline sentence is unbacked, and
  it is doing most of the work in that sentence.
- **Boundary-seeking anchor placement does not exist**, and Experiment P showed geometric
  pooling *requires* it: **0.96** with anchors placed by design against **0.00** with
  them placed at random. Not slower — dead. This moved from a deployment nicety to a
  prerequisite for the recovery claim.
- **Nothing has been run end-to-end on any scaffold.** Every power figure in the project
  is against gates we planted ourselves.

## 7. What counts as a "real" target — and the trap in that phrase

The documents say "the real target", meaning the housing scaffold. That is too generous,
and the distinction matters. There are four rungs of realism:

| rung | ground truth available? | whose construction? |
|---|---|---|
| 1. planted gates in simulation | yes, by construction | ours |
| 2. **the housing scaffold** | yes — we built the router | **ours** |
| 3. Slack et al.'s published scaffolds | yes — documented attack | **someone else's**, built to defeat audits |
| 4. a deployed API with routing *suspicion* | **none** | someone else's |

Everything validated so far is rung 1. The housing scaffold is rung 2 — more realistic
in that it is a trained model on real data with a genuine internal router the audit
cannot see, but **we chose the construction and we know the answer.**

**The trap:** rung 4 cannot validate the method. With no ground truth there is no power
and no false-positive rate to measure — a null result and a miss are indistinguishable,
and the parallel-penalty limit (§8) guarantees some routing is invisible no matter what.
**Rung 4 is where credibility gets spent, not earned.**

So the genuine held-out test is **rung 3**: Slack's models, because we did not design
them and they were explicitly designed to defeat auditing. There is a hard blocker
there — Experiment S measured that those datasets expose **one or two distinct response
values**, while Stage 3 tests the shape of a continuous residual distribution. A
**discrete-response Stage 3 variant is a prerequisite, not a tweak.** Housing gives 400
distinct values, which is why it is the only end-to-end-able target today.

The practical reading: rung 2 end-to-end is the next real milestone, rung 3 is the
external test and needs a new Stage 3 first, and rung 4 is a deliverable of the paper
rather than evidence for it.

## 8. What is structurally impossible

Not engineering gaps — these do not yield to more work:

- **A penalty aligned with within-branch variation is invisible.** At every radius, in
  every dimension, under every projection. Proven, not conjectured.
- **π ≈ 0.5 breaks recovery.** At balanced mixing there is no majority branch for the
  trim to commit to. Testing degrades gracefully; recovery does not.
- **A probe ball spanning several routing cells returns a confident wrong answer**, and
  nothing abstains. This is the only failure mode in the method that lies rather than
  declines, and it is invisible to Stage 0 because routing density lives in the response
  while the covariate geometry looks unchanged.
- **Dispersion alone can never separate a gate from curvature.** Identifiability, not a
  bad choice of statistic. This is the finding the project is built on.

## 9. Honest one-paragraph summary

The two scientific links — that the **distance** to a hidden boundary and its
**direction** are both recoverable from query-only access — are validated, in
simulation, on planted gates, with the probe geometry handed over. That is real and it
is the core claim. What is missing is everything between "the estimator works in a clean
room" and "here is a defensible audit of a system someone else built": no automatic
geometry, no calibrated interval, no anchor placement, no end-to-end run at any rung,
and the main confounder still fires the per-anchor test at 100%.

## 10. What comes next, in plain terms

1. **Axis-dominance and t̂** — turn the recovered plane into "feature X above 4.98".
   Short, and it now inherits a calibrated per-rung test to define which rungs count.
2. **Interval coverage** — measure whether the ± is honest. The first-class deliverable
   the method note names, and still unmeasured.
3. **Test the separability guard** — check that the thing meant to tell a real boundary
   from ordinary curvature actually does. Never checked, and now load-bearing.
4. **Boundary-seeking anchor placement** — a prerequisite for the recovery claim, not a
   deployment nicety.
5. **Housing scaffold end-to-end** — rung 2, the first time the whole pipeline runs
   against a model rather than a construction we can see into.

Two smaller items created by the 14 August run: **re-anchor the ladder** so it climbs
away from the boundary (a bookkeeping fix that resolves the π tension), and **fix the
hardcoded `/tmp` paths** in `exp_a_invariance.py` and `exp_p_pooling.py`, neither of
which runs from a fresh checkout.
