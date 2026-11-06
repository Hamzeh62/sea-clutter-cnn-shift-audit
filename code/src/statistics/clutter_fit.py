"""K / Weibull amplitude fitting with method-of-moments + KS model selection.

K pdf (amplitude r):  p_K(r) = (2b/Gamma(nu)) (b r / 2)^nu K_{nu-1}(b r)
Moments: m_k = E[r^k].  From the compound representation r = sqrt(tau)*|g|,
|g| Rayleigh: m2 = 2*nu/b^2 (with tau~Gamma(shape=nu, scale=theta), b^2=4/theta...
we use the spec's MoM formulas:
    nu_hat = 2 / (m4/m2^2 - 2),   b_hat = sqrt(4*nu_hat/m2_bar)
where m2_bar = mean(r^2)/... — spec: b_hat = sqrt(4 nu_hat / m2) with m2 = E[r^2]
checked against E[r^2] = 4 nu / b^2 for this parametrization.
"""
from __future__ import annotations

import numpy as np
from scipy import stats, special


# ----------------------------- K distribution ------------------------------

def k_moments(r: np.ndarray) -> tuple[float, float]:
    """MoM estimators per spec §4.1."""
    r = np.asarray(r, dtype=np.float64)
    m2 = float(np.mean(r**2))
    m4 = float(np.mean(r**4))
    ratio = m4 / (m2**2 + 1e-30)
    nu = 2.0 / max(ratio - 2.0, 1e-6)  # ratio->2 for Rayleigh (nu->inf)
    nu = float(np.clip(nu, 0.05, 500.0))
    b = float(np.sqrt(4.0 * nu / max(m2, 1e-30)))
    return nu, b


def k_pdf(r: np.ndarray, nu: float, b: float) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    z = np.clip(b * r / 2.0, 1e-12, None)
    return (2.0 * b / special.gamma(nu)) * z**nu * special.kv(nu - 1.0, z)


def k_cdf(r: np.ndarray, nu: float, b: float, nodes: int = 96) -> np.ndarray:
    """Exact semianalytic K CDF: r = sqrt(tau)*u, tau~Gamma(nu, 4/b^2), E[u^2]=1.

    CDF(x) = E_tau[1 - exp(-x^2/tau)], Gauss-Legendre nodes in the uniform
    domain mapped through the Gamma quantile function.
    """
    u, w = np.polynomial.legendre.leggauss(nodes)
    u = 0.5 * (u + 1.0)  # map to (0,1)
    w = 0.5 * w
    u = np.clip(u, 1e-9, 1 - 1e-9)
    tau = stats.gamma.ppf(u, a=nu, scale=4.0 / b**2)
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    cdf = (w * (1 - np.exp(-np.outer(r**2, 1.0 / tau)))).sum(axis=1)
    return np.clip(cdf, 0.0, 1.0)


# ---------------------------- Weibull --------------------------------------

def weibull_moments(r: np.ndarray) -> tuple[float, float]:
    """Weibull shape k and scale lambda via MoM on r and r^2."""
    r = np.asarray(r, dtype=np.float64)
    m1 = float(np.mean(r))
    m2 = float(np.mean(r**2))
    ratio = m2 / (m1**2 + 1e-30)  # = Gamma(1+2/k)/Gamma(1+1/k)^2
    ks = np.linspace(0.2, 20.0, 4000)
    g = special.gamma(1 + 2 / ks) / special.gamma(1 + 1 / ks) ** 2
    k = float(ks[np.argmin(np.abs(g - ratio))])
    lam = m1 / special.gamma(1 + 1 / k)
    return k, lam


# ----------------------------- fit + KS ------------------------------------

def fit_cell(r: np.ndarray, max_samples: int = 20000, rng=None) -> dict:
    """Fit K and Weibull to one cell's amplitudes; KS tests; pick winner.

    K parameters: spec MoM formulas (primary). A KS-minimizing polish over nu
    (with b re-derived from m2) is stored alongside for the physics-loss pdf,
    because MoM underfits heavily textured cells (nu < 0.5).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(r) > max_samples:
        idx = rng.choice(len(r), max_samples, replace=False)
        r = r[idx]
    nu, b = k_moments(r)
    wk, wl = weibull_moments(r)
    m2 = float(np.mean(r**2))
    rs = np.sort(r)
    emp = (np.arange(1, len(rs) + 1)) / len(rs)

    def ks_D(nu_, b_, r_full=rs, emp_full=None):
        if emp_full is None:
            n = len(r_full)
            emp_full = np.arange(1, n + 1) / n
        return float(np.max(np.abs(emp_full - k_cdf(r_full, nu_, b_))))

    # polish grid: thinned samples for speed; final D recomputed on full samples
    thin = 10
    r_th = rs[::thin]
    emp_th = np.arange(1, len(r_th) + 1) / len(r_th)

    def ks_D_thin(nu_, b_):
        return float(np.max(np.abs(emp_th - k_cdf(r_th, nu_, b_))))

    grid = np.geomspace(0.05, 50.0, 60)
    best = min(((n_, np.sqrt(4 * n_ / m2)) for n_ in grid), key=lambda p: ks_D_thin(*p))
    out = {
        "n": int(len(r)),
        "K": {"nu": nu, "b": b, "ks_D": ks_D(nu, b)},
        "K_polished": {"nu": float(best[0]), "b": float(best[1]),
                       "ks_D": ks_D(*best)},
        "weibull": {"k": wk, "lambda": wl},
    }
    cdf_w = stats.weibull_min.cdf(rs, wk, scale=wl)
    out["weibull"]["ks_D"] = float(np.max(np.abs(emp - cdf_w)))
    d_k = out["K_polished"]["ks_D"]
    out["best"] = "K" if d_k <= out["weibull"]["ks_D"] else "weibull"
    return out


def local_power(z: np.ndarray) -> float:
    """m2_local: mean |z|^2 (clutter power proxy for SCR scaling)."""
    return float(np.mean(np.abs(z) ** 2))
