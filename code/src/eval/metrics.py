"""Evaluation metrics: Clopper-Pearson CIs, AUC."""
from __future__ import annotations

import numpy as np
from scipy import stats


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def auc_from_scores(h0: np.ndarray, h1: np.ndarray) -> float:
    """Mann-Whitney AUC from null and alternative statistic samples."""
    from scipy.stats import rankdata
    allv = np.concatenate([h0, h1])
    r = rankdata(allv)
    n0, n1 = len(h0), len(h1)
    # AUC = P(h1 > h0): rank sum of the h1 block
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))
