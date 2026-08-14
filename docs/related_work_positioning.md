# Related work: what the neighbouring literature does, and why this is not that

*Positioning note, 15 August 2026. Written to be folded into the submission's related-work
section. The point is not to list citations but to state the distinction precisely enough
that a reviewer cannot collapse this contribution into an existing one.*

---

## The literature that looks closest

### Black-box boundary-geometry estimation (adversarial attacks)

**GeoDA** — Rahmati, Moosavi-Dezfooli, Frossard & Dai, *GeoDA: a geometric framework for
black-box adversarial attacks*, CVPR 2020 ([arXiv:2003.06468](https://arxiv.org/abs/2003.06468)).

This is the closest-looking prior work and must be addressed head-on. GeoDA operates in
the hard-label black-box setting — each query returns only the classifier's top-1 label —
and estimates the **local geometry of the decision boundary** in order to construct a
minimal-norm adversarial perturbation. It explicitly assumes the boundary has **small mean
curvature near the data point**, and proves ℓ₂ convergence under a bounded-curvature
condition. It also derives an optimal allocation of a query budget across iterations.

Read at the level of a one-line summary — *"black-box queries, estimate a boundary's local
linear geometry, worry about curvature, worry about query budget"* — that is this project.
The distinction has to be made at the level of **what is observable**.

### Active learning of halfspaces

A large classical literature learns a linear separator query-efficiently. Representative
entries: Gonen, Sabato & Shalev-Shwartz, *Efficient Active Learning of Halfspaces: an
Aggressive Approach*, JMLR 2013 ([arXiv:1208.3561](https://arxiv.org/abs/1208.3561));
Yan & Zhang, *Revisiting Perceptron: Efficient and Label-Optimal Learning of Halfspaces*,
NeurIPS 2017; and recent work separating label queries from membership queries, e.g.
Diakonikolas et al., *Active Learning of General Halfspaces: Label Queries vs Membership
Queries* ([arXiv:2501.00508](https://arxiv.org/abs/2501.00508)).

**Consequence: "recover a hyperplane from few queries" cannot be the novelty.** That
problem is solved, repeatedly, with better rates than anything here. Any framing that
foregrounds query-efficiency as the contribution will be correctly rejected.

---

## The distinction, stated precisely

Both literatures above are given **observable membership**. GeoDA is handed the label; an
active halfspace learner is handed the side of the separator each queried point falls on.
The learning problem is *where is the boundary*, and every query is an unbiased, if noisy,
report of which side you are on.

Here **route membership is latent and never observed at all.**

| | GeoDA / active halfspace learning | this work |
|---|---|---|
| what a query returns | the side (a label) | a scalar response |
| membership | **observed** | **latent** — must be inferred |
| the boundary's role | the object being approached | a nuisance discontinuity in a regression surface |
| the goal | an adversarial example / a classifier | the **mechanism**, with uncertainty |
| what can go wrong | curvature slows convergence | curvature is **observationally equivalent** to the target |

The chain that replaces "read the label" is:

```
scalar responses  →  latent mixture responsibilities  →  boundary geometry  →  uncertainty
```

and each arrow is a statistical inference rather than a lookup. That is the contribution:

1. **The mixture's *weight* identifies distance.** The crossing fraction inverts to
   d̂ = −σΦ⁻¹(π̂). Nothing in the label-based literature needs this step, because distance
   to the boundary is obtained by line search on an observable label.
2. **The mixture's *spatial responsibilities* identify orientation.** The separating
   direction of a discriminant fitted on (probe coordinates, responsibilities) recovers
   the boundary normal — and does so with no detectable bias, so corrupted labels
   *attenuate* the estimate rather than rotating it. This is the step that has no analogue
   at all when membership is observed.
3. **Aggregation with calibrated uncertainty.** Pooling noisy local inverse estimates into
   one boundary, with an interval that covers.

## Why the confound is a different problem here, not a harder version of the same one

This is the sharpest point of separation and it should be made explicitly, because it is
also the project's main reviewer vulnerability.

For GeoDA, curvature is a **rate** problem: a curved boundary makes the local linear
approximation worse, so convergence slows. The boundary's *existence* is never in question
— the labels prove it.

Here, curvature is an **identifiability** problem. A smooth surface with sufficient
curvature at the probe scale produces a residual distribution that a two-component mixture
fits better than one Gaussian, with **no discontinuity anywhere**. Measured: resonant
curvature fires the calibrated per-anchor test at up to 1.000, and the pooled geometric
estimator fires at 100% with direction coherence R ≈ 0.879 on a surface containing no
boundary. There is no label to appeal to.

So the null hypothesis cannot be "one Gaussian after a local linear fit". The scientifically
honest pair is

```
H0:  y = h(x) + eps,                              h in C^2,  ||grad^2 h|| <= L
H1:  y = h(x) + Delta * 1{n'x > c} + eps
```

and without the smoothness restriction the alternative is not identified — a sufficiently
sharp but continuous function approximates a step arbitrarily well at finite query
resolution. **That is a genuine identifiability boundary, and it deserves to be stated as
one rather than reported as an inconvenient false-positive rate.**

It also supplies the theoretical rationale for the multi-scale ladder, which is otherwise
a heuristic: over a ball of radius r, a curvature-bounded surface contributes a response
spread of order ½Lr², while a jump contributes Δ regardless of r. So the two separate when
Δ/(Lr²) is large, and the ladder's job is to measure that exponent — r⁰ for a step against
r² for curvature. The regime where it must fail is oscillation below the probe diameter,
which is exactly where the empirical failure sits.

## Why Slack et al. is the right motivation

Slack, Hilgard, Jia, Singh & Lakkaraju, *Fooling LIME and SHAP: Adversarial Attacks on Post
hoc Explanation Methods*, AIES 2020.

Their scaffolding attack is not an analogy for hidden routing — it **is** hidden routing,
built deliberately. A biased model f, an innocuous ψ, and an OOD detector routing between
them on the basis of whether an input looks like a real record or an explainer's
perturbation. The routing rule is latent by construction and the attack's entire purpose is
that no post-hoc explanation reveals it.

That makes it the natural motivating target, and it also yields a result worth reporting in
its own right: **the detector recognises essentially 100% of the LIME-style perturbations it
was trained against and 83–98% of ours as genuine records.** It learned exactly one
perturbation family and generalises to none. The scaffold is defeated by a probe geometry
it never saw — which is a statement about the *attack's* scope, and it is sharper because
Communities & Crime is a counterexample: on all-continuous data, where there is no discrete
lattice to stay on, our probes are detected too.

## How to phrase the contribution so it cannot be collapsed

Avoid: *"a query-efficient method for estimating a hyperplane from black-box access."*
That is the active-learning problem, and it is solved.

Prefer: *"recovery of a latent routing rule — its orientation, location, and threshold,
with calibrated uncertainty — from a scalar black-box response in which route membership is
never observed, under an explicit bounded-curvature alternative."*

The load-bearing words are **latent membership**, **calibrated uncertainty**, and **bounded
curvature**. Each names something the neighbouring literature does not have to do.

---

## Sources

- [GeoDA: a geometric framework for black-box adversarial attacks (arXiv:2003.06468)](https://arxiv.org/abs/2003.06468)
- [Efficient Active Learning of Halfspaces: an Aggressive Approach (arXiv:1208.3561)](https://arxiv.org/abs/1208.3561)
- [Active Learning of General Halfspaces: Label Queries vs Membership Queries (arXiv:2501.00508)](https://arxiv.org/abs/2501.00508)
- [Revisiting Perceptron: Efficient and Label-Optimal Learning of Halfspaces (NeurIPS 2017)](https://papers.nips.cc/paper/2017/file/556f391937dfd4398cbac35e050a2177-Reviews.html)
- [GeoDA, dblp record (CVPR 2020)](https://dblp.org/rec/conf/cvpr/RahmatiMFD20.html)
