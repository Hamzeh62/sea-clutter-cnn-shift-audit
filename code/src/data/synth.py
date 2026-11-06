"""Synthetic clutter + target generation.

- SIRP / compound-Gaussian K clutter: s(t) = sqrt(tau(t)) * g(t),
  tau ~ Gamma(nu, theta) with AR(1)-correlated texture,
  g complex Gaussian whose Doppler PSD is shaped to a template PSD (Welch of real data).
- Swerling 0/1/2 targets with Doppler in a clear region.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.special import erf
from scipy.stats import gamma as _gamma


# --------------------------- texture (AR(1) Gamma) --------------------------

def ar1_gamma_texture(n: int, nu: float, theta: float, rho: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Gamma(shape=nu, scale=theta) marginal with AR(1)-Gaussian-copula correlation.

    rho = one-lag Pearson correlation of the underlying Gaussian (0..1).
    Returns tau of length n (mean = nu*theta).
    """
    z = np.empty(n)
    z[0] = rng.standard_normal()
    innov_sd = np.sqrt(1 - rho**2)
    for i in range(1, n):  # n ~ 1e4 typically; fine
        z[i] = rho * z[i - 1] + innov_sd * rng.standard_normal()
    u = 0.5 * (1 + erf(z / np.sqrt(2)))
    return _gamma.ppf(np.clip(u, 1e-12, 1 - 1e-12), a=nu, scale=theta)


# --------------------------- PSD-shaped speckle -----------------------------

def shape_filter(psd_template: np.ndarray, n: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Complex Gaussian noise, length n, unit mean power, average PSD ~ template."""
    spec = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    tpl = np.clip(np.asarray(psd_template, dtype=np.float64), 1e-12, None)
    # interpolate template onto the n-point FFT bin grid (0..n-1 -> template resampled)
    H = np.sqrt(np.interp(np.linspace(0, len(tpl), n, endpoint=False),
                          np.arange(len(tpl)), tpl))
    H /= np.sqrt(np.mean(H**2))
    y = np.fft.ifft(spec * H)
    return y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-30)


def estimate_psd(z: np.ndarray, prf: float, nperseg: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of a complex series; returns (freqs, psd) with freqs in Hz."""
    f, p = welch(z, fs=prf, nperseg=nperseg, return_onesided=False, detrend=False)
    order = np.argsort(f)
    return f[order], p[order]


def synth_compound_gaussian(n: int, nu: float, b: float,
                            psd_template: np.ndarray | None,
                            rho: float, rng: np.random.Generator) -> np.ndarray:
    """K-distributed complex clutter, length n.  m2 = E|s|^2 = 4 nu / b^2.

    E|s|^2 = E[tau] * E|g|^2 with unit-power speckle; Gamma(shape=nu, scale=theta)
    marginal gives E[tau] = nu*theta = 4*nu/b^2  =>  theta = 4/b^2
    (verified numerically against the closed-form K pdf).
    """
    tau = ar1_gamma_texture(n, nu, 4.0 / b**2, rho, rng)
    g = shape_filter(psd_template if psd_template is not None else np.ones(512), n, rng)
    return np.sqrt(tau) * g


# ------------------------------- targets ------------------------------------

def swerling_target(n: int, f_d: float, prf: float, amp: float,
                    mode: int, rng: np.random.Generator) -> np.ndarray:
    """s_T[k] = amp * A_k * exp(j(2 pi f_d k / prf + phi0)).

    mode 0: A_k = 1 (non-fluctuating)
    mode 1: A constant within CPI, exponential across calls (Rayleigh amp)
    mode 2: exponential pulse-to-pulse (Rayleigh amp each pulse)
    """
    phi0 = rng.uniform(0, 2 * np.pi)
    n_idx = np.arange(n)
    base = np.exp(1j * (2 * np.pi * f_d * n_idx / prf + phi0))
    if mode == 0:
        a = np.ones(n)
    elif mode == 1:
        a = np.sqrt(rng.exponential()) * np.ones(n)
    elif mode == 2:
        a = np.sqrt(rng.exponential(size=n))
    else:
        raise ValueError(f"unknown Swerling mode {mode}")
    return amp * a * base


def clear_doppler_band(psd: np.ndarray, freqs: np.ndarray, guard_bins: int = 8) -> tuple[float, float]:
    """Find the largest PSD valley region away from the clutter peak.

    Returns (f_lo, f_hi) of the clear band, in Hz.
    """
    i_peak = int(np.argmax(psd))
    # search band farthest (in dB) from peak
    db = 10 * np.log10(psd / psd.max() + 1e-20)
    mask = np.ones(len(psd), bool)
    lo = max(0, i_peak - guard_bins)
    hi = min(len(psd), i_peak + guard_bins + 1)
    mask[lo:hi] = False
    if not mask.any():
        return (float(freqs[0]), float(freqs[-1]))
    # contiguous run of minimal mean dB
    runs = []
    i = 0
    while i < len(psd):
        if mask[i]:
            j = i
            while j < len(psd) and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    best = min(runs, key=lambda r: np.mean(db[r[0]:r[1]]))
    return float(freqs[best[0]]), float(freqs[best[1] - 1])


def inject_target(z: np.ndarray, scr_db: float, f_d: float, prf: float,
                  mode: int, rng: np.random.Generator) -> np.ndarray:
    """Add a Swerling target to complex clutter z at given SCR (dB).

    sigma_T = SCR * m2_local with m2_local from z itself.
    """
    m2 = np.mean(np.abs(z) ** 2)
    amp = np.sqrt(10 ** (scr_db / 10.0) * m2)
    return z + swerling_target(len(z), f_d, prf, amp, mode, rng)
