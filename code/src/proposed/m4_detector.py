"""M4 — detector on the separated target channel (spec §4.4 M4).

NMF statistic with C_sep estimated from clutter-only training windows in the
SEPARATED domain, plus the end-to-end pipeline: network -> warm-started cICA ->
separated channel -> statistic.
"""
from __future__ import annotations

import numpy as np

from ..baselines.anmf import steering_bank
from .m3_cica import anchored_cica, newton_schulz_polar


def separate(x: np.ndarray, W_warm: np.ndarray | None, r: np.ndarray | None,
             xi: float = 0.5, mu: float = 0.1, tol: float = 1e-6,
             max_sweeps: int = 100, seed: int = 0,
             damping: float = 0.0, best_iterate: bool = False,
             perm_invariant: bool = False, stagnation: tuple | None = None,
             pre_whiten: tuple | None = None, polar_done: bool = False) -> dict:
    """x: (M, Np) multi-channel window; W_warm: (d, d) whitened-space warm start
    (polar of V @ A_hat columns); r: (Np,) reference signal or None.
    Returns anchored_cica dict (S rows: row 0 = anchored target channel).

    D4 oscillation fixes (default-off): damping / best_iterate /
    perm_invariant / stagnation pass through to anchored_cica.

    D36 hoists (default-off; legacy behaviour bit-identical when unset):
    pre_whiten=(z, V, d) reuses the caller's whiten instead of recomputing it
    (the caller's and this function's whiten are the same deterministic call
    on the same x), and polar_done=True accepts W_warm already in polar form
    (the SVD polar factor U_ @ Vh_ needs no second Newton-Schulz pass).
    """
    from .m3_cica import whiten
    if pre_whiten is None:
        z, V, d, lam = whiten(x, mdl=True)
    else:
        z, V, d = pre_whiten
    if W_warm is None:
        W0 = None
    elif polar_done:
        W0 = np.asarray(W_warm, dtype=np.complex128)
    else:
        W0 = newton_schulz_polar(np.asarray(W_warm, dtype=np.complex128))
    if r is None:
        # fallback reference: strongest Doppler projection of primary channel
        P = steering_bank(z.shape[1])
        r = (np.abs(P.conj().T @ z[0]) ** 2).astype(np.complex128)
    out = anchored_cica(z, r, W0=W0, xi=xi, mu=mu, tol=tol, max_sweeps=max_sweeps,
                        seed=seed, damping=damping, best_iterate=best_iterate,
                        perm_invariant=perm_invariant, stagnation=stagnation)
    out["V"] = V
    return out


def nmf_statistic(s_T: np.ndarray, prep_sep: dict) -> float:
    """t = max_f |p^H Csep^-1 s|^2 / [(p^H Csep^-1 p)(s^H Csep^-1 s)].

    D36: if prep carries a precomputed inverse `Cinv` (pool-level — Chat is
    fixed across units), apply it by matvec (~0.2 ms) instead of refactoring
    Chat per unit with np.linalg.solve (~165 ms at Np = 512). Preps without
    `Cinv` keep the exact legacy float path (recorded artifacts unchanged).
    """
    if "Cinv" in prep_sep:
        Cinv_s = prep_sep["Cinv"] @ s_T
    else:
        Cinv_s = np.linalg.solve(prep_sep["Chat"], s_T)
    v = prep_sep["CinvP"].conj().T @ s_T
    d = float(np.real(np.vdot(s_T, Cinv_s)))
    v2 = np.abs(v) ** 2
    return float(np.max(v2 / (prep_sep["a"] * d + 1e-30)))


def prep_sep_from_windows(windows: np.ndarray) -> dict:
    """Estimate C_sep from clutter-only separated-domain windows (rows of `windows`).

    Mirrors the ANMF prep (unit normalization per window, diagonal jitter).
    """
    Yn = windows / np.sqrt(np.sum(np.abs(windows) ** 2, axis=1, keepdims=True) + 1e-30)
    Np = Yn.shape[1]
    Chat = Yn.conj().T @ Yn / Yn.shape[0]
    Chat = Chat + 1e-10 * np.trace(Chat).real / Np * np.eye(Np)
    P = steering_bank(Np)
    CinvP = np.linalg.solve(Chat, P)
    a = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))
    # D36: pool-level inverse hoisted out of the per-unit path
    Cinv = np.linalg.inv(Chat)
    return {"Chat": Chat, "CinvP": CinvP, "a": a, "Cinv": Cinv}
