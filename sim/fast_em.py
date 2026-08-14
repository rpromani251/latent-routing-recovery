"""
Fast batched equal-variance two-component EM, for the full-pipeline bootstrap.

THE ALGEBRA. For an equal-variance two-component Gaussian mixture the log-odds of
component 2 against component 1 is LINEAR in the observation:

    delta(y) = log(w2/w1) + [(y-mu1)^2 - (y-mu2)^2] / (2 sd^2)
             = a + b y,     b = (mu2-mu1)/sd^2,  a = log(w2/w1) + (mu1^2-mu2^2)/(2 sd^2)

so the responsibility is sigmoid(a + b y) and the per-point log-sum-exp is
L1(y) + softplus(a + b y). Every quantity the M step needs is then a sum of
1, y or y^2 against the responsibility, and sum(y), sum(y^2) are FIXED across
iterations. That removes the (chains x points x components) tensor entirely: each
iteration touches (chains x points) a handful of times instead of (chains x points x 2)
about ten times.

This is an algebraic rewrite, not an approximation: same initialisation, same update
order, same convergence test, same "ll at the pre-update parameters" convention,
same best-of-inits rule, including the reference's `+1e-12` on the component counts
(which makes the mixing weights sum to 1 + 2e-12 -- reproduced rather than tidied,
because the calibration must run the pipeline as it actually is).

Only floating-point association differs, so equality with the reference is checked to
tolerance rather than bitwise; `verify()` reports the observed gap.

The bootstrap consumes the LRT statistic only, so responsibilities and the
discriminant are not returned.
"""
import numpy as np

LOG2PI = float(np.log(2.0 * np.pi))


def _softplus_and_expit(x):
    """(log(1+exp(x)), sigmoid(x)) sharing one exponential, stable in both tails.

    Both quantities are needed every iteration -- softplus for the log-likelihood,
    the sigmoid for the responsibilities -- and both are functions of exp(-|x|), so
    computing them together halves the transcendental count. Writing them as
    exp(-|x|) rather than exp(x) keeps the tails from overflowing.
    """
    ex = np.exp(-np.abs(x))
    den = 1.0 + ex
    sp = np.maximum(x, 0.0) + np.log1p(ex)
    p = np.where(x >= 0.0, 1.0 / den, ex / den)
    return sp, p


# Replicates per block. Measured on this harness at n = 800: B = 6000 costs 140 s
# unblocked, 51 s at 500, 40 s at 250 and no better below that -- the win is cache
# residency plus the fact that a smaller block stops as soon as ITS slowest chain
# converges. Results are identical at every block size.
CHUNK_ROWS = 250


def batched_lrt(Y, MU0, max_iter=150, tol=1e-7, chunk=CHUNK_ROWS):
    """Equal-variance 2-GMM LRT statistic for a batch of samples.

    Y   : (R, n)          R residual vectors of common length n
    MU0 : (R, n_init, 2)  starting means, sorted within each init

    Returns (R,) array of 2 * (ll_mixture - ll_single_gaussian), best init per row.

    Large batches are processed in blocks. The loop runs until the LAST chain in the
    block converges, so one slow chain drags every other chain in its block through
    the full iteration budget; blocking bounds that waste as well as peak memory,
    which at R = 20000 would otherwise be several GB of per-iteration churn. Blocking
    cannot change a result -- chains do not interact.
    """
    Y = np.asarray(Y, float)
    if chunk and Y.shape[0] > chunk:
        return np.concatenate([
            batched_lrt(Y[i:i + chunk], MU0[i:i + chunk], max_iter, tol, chunk=0)
            for i in range(0, Y.shape[0], chunk)])
    R, n = Y.shape
    n_init = MU0.shape[1]
    K = R * n_init

    y = np.repeat(Y, n_init, axis=0)                          # (K, n)
    Sy = y.sum(axis=1)
    Sy2 = (y * y).sum(axis=1)

    mu1 = np.ascontiguousarray(MU0[:, :, 0].reshape(K)).astype(float)
    mu2 = np.ascontiguousarray(MU0[:, :, 1].reshape(K)).astype(float)
    sd = np.maximum(np.repeat(Y.std(axis=1), n_init), 1e-9)
    w1 = np.full(K, 0.5)
    w2 = np.full(K, 0.5)
    ll = np.zeros(K)
    ll_old = np.full(K, -np.inf)
    active = np.ones(K, bool)

    # Converged chains are FROZEN in place rather than sliced out: subsetting would
    # copy the (chains x points) response block every iteration, which costs more
    # than the arithmetic it skips when most chains run to a similar depth.
    for _ in range(max_iter):
        if not active.any():
            break
        inv2s2 = 0.5 / (sd * sd)
        b = (mu2 - mu1) * (2.0 * inv2s2)                      # (mu2-mu1)/sd^2
        a = np.log(w2) - np.log(w1) + (mu1 * mu1 - mu2 * mu2) * inv2s2

        delta = a[:, None] + b[:, None] * y                   # (K, n)
        sp, r2 = _softplus_and_expit(delta)

        # ll at the PRE-update parameters. sum(L1) is closed form in Sy, Sy2, so
        # only the softplus term needs a pass over the data.
        sumL1 = (n * np.log(w1) - 0.5 * n * LOG2PI - n * np.log(sd)
                 - (Sy2 - 2.0 * mu1 * Sy + n * mu1 * mu1) * inv2s2)
        ll_new = sumL1 + sp.sum(axis=1)
        ll = np.where(active, ll_new, ll)

        # M step
        N2 = r2.sum(axis=1)
        S2 = (r2 * y).sum(axis=1)
        N1 = n - N2
        S1 = Sy - S2

        nk1, nk2 = N1 + 1e-12, N2 + 1e-12                     # reference's guard
        new_m1, new_m2 = S1 / nk1, S2 / nk2
        var = (Sy2 - 2.0 * (new_m1 * S1 + new_m2 * S2)
               + new_m1 * new_m1 * N1 + new_m2 * new_m2 * N2) / n
        new_sd = np.maximum(np.sqrt(np.maximum(var, 0.0)), 1e-9)

        w1 = np.where(active, nk1 / n, w1)
        w2 = np.where(active, nk2 / n, w2)
        mu1 = np.where(active, new_m1, mu1)
        mu2 = np.where(active, new_m2, mu2)
        sd = np.where(active, new_sd, sd)

        done = (ll_new - ll_old) < tol * np.maximum(1.0, np.abs(ll_new))
        ll_old = np.where(active & ~done, ll_new, ll_old)
        active = active & ~done

    ll1 = ll.reshape(R, n_init).max(axis=1)
    ybar = Y.mean(axis=1)
    ysd = np.maximum(Y.std(axis=1), 1e-9)
    ll0 = (-0.5 * n * LOG2PI - n * np.log(ysd)
           - ((Y - ybar[:, None]) ** 2).sum(axis=1) / (2.0 * ysd * ysd))
    return 2.0 * (ll1 - ll0)


def make_inits(Y, rng, n_init=3):
    """Reference initialisation: the quartile pair, then n_init-1 sorted random pairs.

    NOTE the reference sorts each random pair but does NOT sort the quartile pair
    (it is already ascending), and takes the pair WITHOUT replacement.
    """
    R, n = Y.shape
    q = np.quantile(Y, [0.25, 0.75], axis=1).T
    out = np.empty((R, n_init, 2))
    out[:, 0, :] = q
    for j in range(1, n_init):
        idx = np.argsort(rng.random((R, n)), axis=1)[:, :2]    # without replacement
        out[:, j, :] = np.sort(np.take_along_axis(Y, idx, axis=1), axis=1)
    return out


# --------------------------------------------------------------------- verify
class _StubRng:
    """Replays fixed pairs through .choice(y, 2, replace=False) so the reference
    starts from exactly the inits this module generated."""

    def __init__(self, pairs):
        self.pairs = list(pairs)
        self.i = 0

    def choice(self, y, k, replace=False):
        v = self.pairs[self.i]
        self.i += 1
        return np.asarray(v, float)


def verify(reference_gmm, n_trials=60, n=800, seed=7, verbose=True):
    """Compare LRT against the reference on identical data AND identical inits."""
    rng = np.random.default_rng(seed)
    abs_d, rel_d, vals = [], [], []
    for t in range(n_trials):
        kind = t % 6
        if kind == 0:
            v = rng.normal(0, 1, n)
        elif kind == 1:
            v = np.concatenate([rng.normal(-0.8, 1, n // 2), rng.normal(0.8, 1, n - n // 2)])
        elif kind == 2:
            v = rng.gamma(2.0, 1.0, n)
        elif kind == 3:
            v = np.concatenate([rng.normal(0, 1, int(0.9 * n)),
                                rng.normal(3.0, 1, n - int(0.9 * n))])
        elif kind == 4:                       # heavily trimmed / platykurtic
            z = rng.normal(0, 1, 4 * n)
            v = z[np.argsort(np.abs(z))[:n]]
        else:                                 # near-degenerate scale
            v = rng.normal(0, 1e-4, n)
        Y = v[None, :]
        MU0 = make_inits(Y, np.random.default_rng(1000 + t), n_init=3)
        got = batched_lrt(Y, MU0)[0]
        want = reference_gmm(v, _StubRng([MU0[0, 1], MU0[0, 2]]))["lrt"]
        abs_d.append(abs(got - want))
        rel_d.append(abs(got - want) / max(abs(want), 1e-12))
        vals.append(want)
    abs_d, rel_d = np.array(abs_d), np.array(rel_d)
    if verbose:
        print(f"  fast EM vs reference over {n_trials} residual shapes "
              f"(LRT range {min(vals):.2f} to {max(vals):.2f}):")
        print(f"    max abs diff {abs_d.max():.3e}   max rel diff {rel_d.max():.3e}")
    return float(abs_d.max()), float(rel_d.max())


if __name__ == "__main__":
    import importlib.util
    import os
    import time
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("p", os.path.join(here, "exp_p_pooling.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    verify(lambda y, r: P.gmm2_equalvar(y, r, n_init=3))

    rng = np.random.default_rng(0)
    Y = rng.normal(0, 1, size=(300, 800))
    MU0 = make_inits(Y, rng)
    t0 = time.time(); batched_lrt(Y, MU0); print(f"  B=300 x n=800: {time.time()-t0:.2f} s")
