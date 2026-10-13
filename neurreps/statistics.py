from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_mean_ci(values, *, confidence: float = 0.95, n_boot: int = 10000, seed: int = 42):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(x, size=(int(n_boot), len(x)), replace=True), axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))

def paired_permutation_test(a, b, *, n_perm: int = 100000, seed: int = 42):
    """Two-sided sign permutation for paired stochastic runs.

    In this framework the usual pairing unit is the random seed. The resulting
    p-value quantifies consistency across optimization runs; it must not be
    interpreted as population-level evidence across independent datasets. For
    n <= 20, all 2^n sign patterns are enumerated.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    diff = a[keep] - b[keep]
    if len(diff) == 0:
        return {"mean_difference": np.nan, "p_value": np.nan, "n": 0, "exact": False}
    observed = abs(float(diff.mean()))
    if len(diff) <= 20:
        exceed, total = 0, 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(diff)):
            total += 1
            exceed += abs(float(np.mean(diff * np.asarray(signs)))) >= observed - 1e-15
        p_value = exceed / total
        exact = True
    else:
        rng = np.random.default_rng(seed)
        exceed = 0
        for _ in range(int(n_perm)):
            signs = rng.choice((-1.0, 1.0), size=len(diff))
            exceed += abs(float(np.mean(diff * signs))) >= observed
        p_value = (exceed + 1) / (int(n_perm) + 1)
        exact = False
    mean_difference = float(diff.mean())
    sd_difference = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    if len(diff) > 1 and np.isfinite(sd_difference) and sd_difference > 0:
        paired_dz = float(mean_difference / sd_difference)
    else:
        paired_dz = float("nan")
    if mean_difference > 0:
        same_sign_fraction = float(np.mean(diff > 0))
    elif mean_difference < 0:
        same_sign_fraction = float(np.mean(diff < 0))
    else:
        same_sign_fraction = float(np.mean(diff == 0))
    return {
        "mean_difference": mean_difference,
        "median_difference": float(np.median(diff)),
        "paired_dz": paired_dz,
        "same_sign_as_mean_fraction": same_sign_fraction,
        "p_value": float(p_value),
        "n": int(len(diff)),
        "exact": bool(exact),
    }

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return adjusted
    order_local = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order_local)
    for rank, idx in enumerate(order_local):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2 or np.allclose(a[keep], b[keep]):
        return {"statistic": np.nan, "p_value": 1.0, "n": int(keep.sum())}
    result = wilcoxon(a[keep], b[keep], zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": int(keep.sum())}
