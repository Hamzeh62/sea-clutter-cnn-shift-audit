"""ANMF (texture-invariant adaptive NMF) and GLRT (whitened matched filter).

Per spec §4.3. Secondary data: windows from adjacent cells at the same CPI
index, each normalized by its own norm (texture removal).
SCM: Chat = (1/K) sum_i y_i y_i^H.  With SCM, null law Beta(1, K-Np+1)
(Kraut–Scharf); with known covariance Beta(1, Np-1).
"""
from __future__ import annotations

import numpy as np


def steering(freq_idx: int, Np: int) -> np.ndarray:
    """Unit-norm Doppler steering vector p(f) for FFT bin freq_idx (fftshifted grid)."""
    n = np.arange(Np)
    # bin index in unshifted convention
    p = np.exp(2j * np.pi * freq_idx * n / Np)
    return p / np.linalg.norm(p)


def steering_bank(Np: int) -> np.ndarray:
    """(Np, Np) matrix of all unit steering vectors (columns, unshifted bins)."""
    n = np.arange(Np)
    P = np.exp(2j * np.pi * np.outer(n, np.arange(Np)) / Np)
    return P / np.linalg.norm(P, axis=0)


def scm(secondary: np.ndarray) -> np.ndarray:
    """secondary: (K, Np) rows = normalized secondary windows."""
    Y = secondary / np.sqrt(np.sum(np.abs(secondary) ** 2, axis=1, keepdims=True) + 1e-30)
    return (Y.conj().T @ Y) / Y.shape[0]


def anmf_stat(y: np.ndarray, secondary: np.ndarray, P: np.ndarray | None = None) -> float:
    """t = max_f |p^H C^-1 y|^2 / [(p^H C^-1 p)(y^H C^-1 y)]."""
    Np = len(y)
    P = P if P is not None else steering_bank(Np)
    C = scm(secondary)
    Cinv = np.linalg.pinv(C, rcond=1e-9)
    v = P.conj().T @ (Cinv @ y)          # (Np,)
    a = np.real(np.einsum("ij,ij->j", P.conj().T, Cinv @ P))  # p^H Cinv p per bin
    d = np.real(y.conj() @ Cinv @ y)
    return float(np.max(np.abs(v) ** 2 / (a * d + 1e-30)))


def glrt_stat(y: np.ndarray, secondary: np.ndarray, P: np.ndarray | None = None) -> float:
    """t = max_f |p^H C^-1 y|^2 / (p^H C^-1 p).  Not texture-invariant."""
    Np = len(y)
    P = P if P is not None else steering_bank(Np)
    C = scm(secondary)
    Cinv = np.linalg.pinv(C, rcond=1e-9)
    Cy = Cinv @ y
    v = P.conj().T @ Cy
    a = np.real(np.einsum("ij,ij->j", P.conj().T, Cinv @ P))
    return float(np.max(np.abs(v) ** 2 / (a + 1e-30)))


def prep_anmf_unit(secondary: np.ndarray, P: np.ndarray | None = None,
                   load_jitter: float = 1e-10, normalize_rows: bool = True) -> dict:
    """Precompute the factorized inverse + P-application for a unit's secondary set.

    normalize_rows=True (spec): each secondary window unit-normalized (texture
    removal) — the Beta(1, K-Np+1) law does NOT hold exactly for this variant.
    normalize_rows=False: plain SCM of Gaussian vectors — Kraut-Scharf
    Beta(1, K-Np+1) applies (used by the T4 theory check).
    """
    Np = secondary.shape[1]
    P = P if P is not None else steering_bank(Np)
    if normalize_rows:
        Yn = secondary / np.sqrt(np.sum(np.abs(secondary) ** 2, axis=1, keepdims=True) + 1e-30)
    else:
        Yn = secondary
    K = Yn.shape[0]
    Chat = (Yn.conj().T @ Yn) / K
    Chat += load_jitter * np.trace(Chat).real / Np * np.eye(Np)
    CinvP = np.linalg.solve(Chat, P)              # (Np, Np)
    a = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))  # p^H Cinv p per bin (conj!)
    return {"K": K, "Np": Np, "Chat": Chat, "CinvP": CinvP, "a": a}


def anmf_glrt_from_prep(y: np.ndarray, prep: dict) -> tuple[float, float]:
    """Statistics for a (possibly injected) CUT vector, reusing prep."""
    Cinv_y = np.linalg.solve(prep["Chat"], y)
    v = prep["CinvP"].conj().T @ y           # p^H Cinv y per bin
    d = float(np.real(np.vdot(y, Cinv_y)))
    v2 = np.abs(v) ** 2
    t_anmf = float(np.max(v2 / (prep["a"] * d + 1e-30)))
    t_glrt = float(np.max(v2 / (prep["a"] + 1e-30)))
    return t_anmf, t_glrt


def anmf_glrt_stats(y: np.ndarray, secondary: np.ndarray,
                    P: np.ndarray | None = None) -> tuple[float, float]:
    """One-shot convenience wrapper (prep + evaluate)."""
    return anmf_glrt_from_prep(y, prep_anmf_unit(secondary, P))


def anmf_threshold_beta(pfa: float, K: int, Np: int, scm_used: bool = True,
                        max_bins: int | None = None) -> float:
    """Analytic threshold under Beta(1, m) null: survival (1-t)^m = Pfa
    => t0 = 1 - Pfa^(1/m), with m = Np-1 (known covariance) or K-Np+1 (SCM,
    Kraut–Scharf).

    The statistic is a max over Doppler bins; single-bin Beta thresholds are
    therefore anti-conservative. With max_bins=Np the Bonferroni-corrected
    threshold (1-(Pfa/Np))^(1/m) is returned instead — still an approximation
    because bin correlations are ignored; empirical calibration is authoritative.
    """
    m = (K - Np + 1) if scm_used else (Np - 1)
    m = max(m, 1)
    q = pfa / max_bins if max_bins else pfa
    return float(1.0 - q ** (1.0 / m))
