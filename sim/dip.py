"""
Hartigan's dip test, self-contained.

Uses the `diptest` package when available (table-interpolated p-values, C speed).
Otherwise falls back to a pure-NumPy dip statistic with Monte-Carlo p-values
calibrated against the uniform null (Hartigan & Hartigan 1985: the uniform is
the asymptotically least favourable unimodal null). The MC table is cached per
sample size and seeded, so p-values are reproducible.

Dip statistic ported (faithfully, modernized) from J. Bauer's Python
implementation of the original algorithm, via the `unidip` package:
  Johannes Bauer, github.com/tatome/dip_test (commit a0e3d44).
"""
import os
import numpy as np

if os.environ.get("DIP_FORCE_MC"):        # test hook: force the MC fallback
    HAVE_DIPTEST = False
else:
    try:
        import diptest as _diptest_pkg
        HAVE_DIPTEST = True
    except ImportError:
        HAVE_DIPTEST = False

_NULL_TABLES = {}          # n -> sorted array of null dips
_MC_B = 4999               # null simulations per sample size
_MC_SEED = 20260729


def _gcm_(cdf, idxs):
    work_cdf = cdf
    work_idxs = idxs
    gcm = [work_cdf[0]]
    touchpoints = [0]
    while len(work_cdf) > 1:
        distances = work_idxs[1:] - work_idxs[0]
        slopes = (work_cdf[1:] - work_cdf[0]) / distances
        minslope = slopes.min()
        minslope_idx = int(np.where(slopes == minslope)[0][0]) + 1
        gcm.extend(work_cdf[0] + distances[:minslope_idx] * minslope)
        touchpoints.append(touchpoints[-1] + minslope_idx)
        work_cdf = work_cdf[minslope_idx:]
        work_idxs = work_idxs[minslope_idx:]
    return np.asarray(gcm), np.asarray(touchpoints, dtype=int)


def _lcm_(cdf, idxs):
    g, t = _gcm_(1 - cdf[::-1], idxs.max() - idxs[::-1])
    return 1 - g[::-1], len(cdf) - 1 - t[::-1]


def _touch_diffs_(part1, part2, touchpoints):
    diff = np.abs(part2[touchpoints] - part1[touchpoints])
    return diff.max(), diff


def dip_stat(dat):
    """Hartigan's dip statistic of a 1-D sample."""
    dat = np.asarray(dat, float)
    idxs, histogram = np.unique(dat, return_counts=True)

    if len(idxs) <= 4 or idxs[0] == idxs[-1]:
        return 0.0

    cdf = np.cumsum(histogram, dtype=float)
    cdf /= cdf[-1]

    work_idxs = idxs.astype(float)
    work_histogram = histogram.astype(float) / histogram.sum()
    work_cdf = cdf

    D = 0.0
    left = [0]
    right = [1]

    while True:
        left_part, left_touchpoints = _gcm_(work_cdf - work_histogram, work_idxs)
        right_part, right_touchpoints = _lcm_(work_cdf, work_idxs)

        d_left, left_diffs = _touch_diffs_(left_part, right_part, left_touchpoints)
        d_right, right_diffs = _touch_diffs_(left_part, right_part, right_touchpoints)

        if d_right > d_left:
            xr = right_touchpoints[d_right == right_diffs][-1]
            xl = left_touchpoints[left_touchpoints <= xr][-1]
            d = d_right
        else:
            xl = left_touchpoints[d_left == left_diffs][0]
            xr = right_touchpoints[right_touchpoints >= xl][0]
            d = d_left

        left_diff = np.abs(left_part[:xl + 1] - work_cdf[:xl + 1]).max()
        right_diff = np.abs(right_part[xr:] - work_cdf[xr:]
                            + work_histogram[xr:]).max()

        if d <= D or xr == 0 or xl == len(work_cdf):
            the_dip = max(np.abs(cdf[:len(left)] - left).max(),
                          np.abs(cdf[-len(right) - 1:-1] - right).max())
            return float(the_dip / 2)
        else:
            D = max(D, float(left_diff), float(right_diff))

        work_cdf = work_cdf[xl:xr + 1]
        work_idxs = work_idxs[xl:xr + 1]
        work_histogram = work_histogram[xl:xr + 1]

        left[len(left):] = list(left_part[1:xl + 1])
        right[:0] = list(right_part[xr:-1])


def _null_table(n):
    if n not in _NULL_TABLES:
        rng = np.random.default_rng(_MC_SEED + n)
        dips = np.empty(_MC_B)
        for b in range(_MC_B):
            dips[b] = dip_stat(rng.uniform(size=n))
        _NULL_TABLES[n] = np.sort(dips)
    return _NULL_TABLES[n]


def preload_null_table(n, path=None):
    """Optionally load/save the MC null table to disk to amortize startup."""
    import os
    if path and os.path.exists(path):
        _NULL_TABLES[n] = np.load(path)["dips"]
        return
    _null_table(n)
    if path:
        np.savez_compressed(path, dips=_NULL_TABLES[n])


def dip_pvalue(x):
    """(dip, p) against the unimodal null.

    With `diptest` installed, defers to its table interpolation. Otherwise
    Monte-Carlo against the uniform null: p = (1 + #{null >= obs}) / (B + 1).
    """
    x = np.asarray(x, float)
    if HAVE_DIPTEST:
        d, p = _diptest_pkg.diptest(np.ascontiguousarray(x))
        return float(d), float(p)
    d = dip_stat(x)
    tab = _null_table(len(x))
    p = (1.0 + float(np.sum(tab >= d))) / (len(tab) + 1.0)
    return d, float(p)


if __name__ == "__main__":
    # sanity checks
    rng = np.random.default_rng(0)
    uni = rng.normal(size=1000)
    bim = np.concatenate([rng.normal(-1.5, 0.3, 500), rng.normal(1.5, 0.3, 500)])
    d1, p1 = dip_pvalue(uni)
    d2, p2 = dip_pvalue(bim)
    print(f"unimodal  dip={d1:.4f} p={p1:.3f}   (want p large)")
    print(f"bimodal   dip={d2:.4f} p={p2:.4f}   (want p ~ 0)")
    # null calibration: p-values under a unimodal null should be ~uniform
    ps = [dip_pvalue(rng.normal(size=500))[1] for _ in range(200)]
    print(f"null p-values: frac<0.05 = {np.mean(np.array(ps) < 0.05):.3f} (want ~0.05)")
    print(f"               frac<0.50 = {np.mean(np.array(ps) < 0.50):.3f} (want ~0.50)")
