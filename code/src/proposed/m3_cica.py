"""M3 — warm-started constrained complex ICA (spec §4.4 M3).

- whiten(): eigendecomposition whitening + MDL source-order estimate (capped d<=M)
- newton_schulz_polar(): differentiable-friendly polar projection onto unitary group
- complex_fastica(): symmetric complex FastICA, G(u)=log(1+u), g(u)=1/(1+u), g'(u)=-1/(1+u)^2
- anchored_cica(): reference-anchored first unit (augmented Lagrangian) + FastICA rest,
  warm-started from the network's mixing head (or random for cold start)
"""
from __future__ import annotations

import numpy as np


def _symmetric_orthonormalize(W: np.ndarray) -> np.ndarray:
    """W <- (W W^H)^{-1/2} W exactly (eigh-based)."""
    G = W @ W.conj().T
    lam, U = np.linalg.eigh(G)
    lam = np.clip(lam, 1e-30, None)
    Gm12 = U @ np.diag(lam ** -0.5) @ U.conj().T
    return Gm12 @ W


def newton_schulz_polar(A: np.ndarray, iters: int = 10) -> np.ndarray:
    """Polar factor A(A^H A)^{-1/2} via Newton-Schulz iteration."""
    X = A / (np.linalg.norm(A) + 1e-30)
    for _ in range(iters):
        X = 0.5 * (3 * X - X @ (X.conj().T @ X))
    return X


def whiten(X: np.ndarray, d: int | None = None, mdl: bool = True):
    """X: (M, N). Returns z (d, N), V (d, M), d_hat, eigvals (desc).

    MDL: d_hat = argmin_k [ -N(M-k) ln g(k) + 0.5 k(2M-k) ln N ],
    g(k) = (prod_{i>k} l_i)^{1/(M-k)} / ((1/(M-k)) sum_{i>k} l_i).
    """
    M, N = X.shape
    Rx = X @ X.conj().T / N
    lam, E = np.linalg.eigh(Rx)
    order = np.argsort(lam)[::-1]
    lam, E = lam[order], E[:, order]
    lam = np.clip(lam, 1e-12, None)
    if d is None:
        if not mdl or M <= 2:
            d = M
        else:
            best, best_cost = M, np.inf
            for k in range(M):  # k = number of "noise" eigenvalues kept out
                rest = lam[k:]
                gm = len(rest)
                geo = np.exp(np.mean(np.log(rest)))
                ari = np.mean(rest)
                g = geo / (ari + 1e-30)
                cost = -N * (M - k) * np.log(g + 1e-30) + 0.5 * k * (2 * M - k) * np.log(N)
                if cost < best_cost:
                    best, best_cost = k, cost
            d = M - best  # number of signal sources
        d = max(1, min(d, M))
    V = (lam[:d] ** -0.5)[:, None] * E[:, :d].conj().T
    z = V @ X
    return z, V, d, lam


# contrast derivatives for G(u) = log(1+u) with u = |y|^2
_g = lambda u: 1.0 / (1.0 + u)
_gprime = lambda u: -1.0 / (1.0 + u) ** 2

# cold-restart counter (reset by the caller before a run; diagnostics read it)
_restart_count = 0


def pop_restart_count() -> int:
    """Return cold restarts since the last reset and reset the counter."""
    global _restart_count
    n = _restart_count
    _restart_count = 0
    return n


def _contrast(z: np.ndarray, W: np.ndarray) -> float:
    Y = W.conj().T @ z
    u = np.abs(Y) ** 2
    return float(np.mean(np.log1p(u)))


def _perm_inv_diff(W: np.ndarray, W_prev: np.ndarray) -> float:
    """min over signed permutations Pi of ||Pi W - W_prev||_F (D4d).

    Rows of W and W_prev are the source estimates; d <= 4 so the d! permutation
    search is trivial. Per permutation the optimal per-row sign is s = sign of
    Re<w_i, wprev_j>, giving cost sum_j (|w_i|^2 + |wprev_j|^2 - 2|<w_i, wprev_j>|).
    """
    import itertools
    d = W.shape[0]
    G = np.abs(W @ W_prev.conj().T)          # (i_new, j_prev) |<w_i, wprev_j>|
    n2 = (np.abs(W) ** 2).sum(axis=1)        # row norms (unit after orthonorm.)
    n2p = (np.abs(W_prev) ** 2).sum(axis=1)
    best = np.inf
    for perm in itertools.permutations(range(d)):
        c = 0.0
        for i, j in enumerate(perm):
            c += n2[i] + n2p[j] - 2.0 * G[i, j]
        if c < best:
            best = c
    return float(np.sqrt(max(best, 0.0)))


def complex_fastica(X: np.ndarray, W0: np.ndarray | None = None, tol: float = 1e-6,
                    max_sweeps: int = 100, seed: int = 0,
                    return_iters: bool = True, d: int | None = None,
                    damping: float = 0.0, best_iterate: bool = False,
                    perm_invariant: bool = False, stagnation: tuple | None = None,
                    return_extra: bool = False):
    """Symmetric complex FastICA on X (M, N): whitens internally (full rank M
    unless d given), then fixed-point sweeps in the whitened space.

    Update (spec): w_j <- E[z y_j* g(|y_j|^2)] - E[g + |y_j|^2 g'] w_j,
    then symmetric orthonormalization W <- (W W^H)^{-1/2} W.
    Cold restart if contrast decreases 3 consecutive sweeps (only when
    `stagnation` is None — D4e replaces that rule).

    D4 oscillation fixes (all default-off; baseline behaviour unchanged):
      damping: alpha in W <- (1-alpha) W + alpha F(W) (re-orthonormalized after).
      best_iterate: track contrast per sweep, return the argmax-contrast iterate.
      perm_invariant: stopping test uses the signed-permutation-invariant
        difference (D4d) instead of the raw ||W - W_prev||_F.
      stagnation: (window, thresh) -> stop when relative contrast improvement
        over the last `window` sweeps < thresh (D4e).

    Returns S (d, N), W_unmix (d, M) [S = W_unmix @ X], iters — and, when
    return_extra, a 4th dict {j_hist, stop_reason, n_sweeps, best_sweep}.
    """
    z, V, d_hat, lam = whiten(X, d=d if d is not None else X.shape[0], mdl=False)
    d, N = z.shape
    rng = np.random.default_rng(seed)
    if W0 is None:
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        W = np.linalg.qr(A)[0]
    else:
        W = W0.copy()
    W = _symmetric_orthonormalize(W)
    prev_J = _contrast(z, W)
    decr = 0
    iters_used = 0
    j_hist = [prev_J]
    j_best, W_best, sw_best = prev_J, W.copy(), 0
    stop_reason = "max_sweeps"
    for sweep in range(max_sweeps):
        W_prev = W.copy()
        Y = W.conj().T @ z                      # (d, N)
        u = np.abs(Y) ** 2
        Wnew = np.empty_like(W)
        for j in range(d):
            yj = Y[j]
            term1 = z @ (yj.conj() * _g(u[j])) / N
            term2 = np.mean(_g(u[j]) + u[j] * _gprime(u[j]))
            Wnew[:, j] = term1 - term2 * W[:, j]
        if damping > 0.0:
            Wnew = (1.0 - damping) * W_prev + damping * Wnew
        W = _symmetric_orthonormalize(Wnew)
        J = _contrast(z, W)
        iters_used = sweep + 1
        j_hist.append(J)
        if J > j_best + 1e-12:
            j_best, W_best, sw_best = J, W.copy(), sweep + 1
        if stagnation is not None:
            win, thr = stagnation
            if len(j_hist) > win:
                j0 = j_hist[-(win + 1)]
                rel = (J - j0) / (abs(j0) + 1e-30)
                if rel < thr:
                    stop_reason = "stagnation"
                    break
        else:
            # legacy 3-decrease cold-restart rule (never fires in practice, D1)
            if J < prev_J - 1e-12:
                decr += 1
                if decr >= 3:
                    global _restart_count
                    _restart_count += 1
                    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
                    W = np.linalg.qr(A)[0]
                    decr = 0
            else:
                decr = 0
        prev_J = J
        if perm_invariant:
            diff = _perm_inv_diff(W, W_prev)
        else:
            diff = np.linalg.norm(W - W_prev)
        if diff < tol:
            stop_reason = "tol"
            break
    if best_iterate and j_best > j_hist[-1] + 1e-12:
        W = W_best
        stop_reason = f"best_iterate(sw{sw_best})"
    S = W.conj().T @ z
    W_unmix = W.conj().T @ V                     # maps original X to S
    if return_extra:
        return S, W_unmix, iters_used, {"j_hist": j_hist, "stop_reason": stop_reason,
                                         "n_sweeps": iters_used, "best_sweep": sw_best}
    if return_iters:
        return S, W_unmix, iters_used
    return S, W_unmix


def anchored_cica(z: np.ndarray, r: np.ndarray, W0: np.ndarray | None = None,
                  xi: float = 0.5, mu: float = 0.1, tol: float = 1e-6,
                  max_sweeps: int = 100, seed: int = 0,
                  damping: float = 0.0, best_iterate: bool = False,
                  perm_invariant: bool = False, stagnation: tuple | None = None):
    """Reference-anchored constrained complex ICA.

    First unit: maximize J(w1^H z) s.t. |E[(w1^H z) r*]| >= xi, augmented
    Lagrangian with dual update lam <- [lam + mu(xi - eps)]_+. Remaining units:
    symmetric FastICA projected orthogonal to w1. Warm start from W0 (polar of
    network head) if given, else random (cold).

    D4 oscillation fixes (default-off): damping / best_iterate /
    perm_invariant / stagnation pass through to the complement FastICA; the
    anchored unit is fixed either way, so best-iterate snapshot keeps it.

    Returns dict with S (d, N), W, w1_index=0, iters (50 + FastICA sweeps),
    warm (bool), anchor_corr, wall_s, and extra (stop_reason/n_sweeps/best_sweep).
    """
    import time
    t0 = time.perf_counter()
    d, N = z.shape
    rng = np.random.default_rng(seed)
    c = z @ r.conj() / N                        # E[z r*]  (d,)
    warm = W0 is not None
    if W0 is None:
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        W0 = np.linalg.qr(A)[0]
    W0 = newton_schulz_polar(W0)
    # --- anchored unit w1 (gradient ascent on augmented Lagrangian) ---
    w1 = W0[:, 0].copy()
    lam = 0.0
    for it in range(50):
        y1 = w1.conj() @ z
        u = np.abs(y1) ** 2
        dJ = z @ (y1.conj() * _g(u)) / N - np.mean(_g(u) + u * _gprime(u)) * w1
        rho = w1.conj() @ c
        eps = np.abs(rho) + 1e-30
        dEps = c * (rho / eps) * 0.5            # d|rho|/dw1* direction
        grad = dJ + lam * dEps                  # ascent on J + lam*(eps - xi)
        w1 = w1 + 0.5 * grad
        w1 = w1 / (np.linalg.norm(w1) + 1e-30)
        eps_new = np.abs(w1.conj() @ c)
        lam = max(0.0, lam + mu * (xi - eps_new))
        if lam == 0.0 and eps_new >= xi:
            break
    # --- remaining units: symmetric FastICA orthogonal to w1 ---
    if d > 1:
        Q = np.linalg.qr(np.stack([w1], axis=1))[0]  # (d,1)
        # orthonormal basis of complement
        full = np.eye(d) - Q @ Q.conj().T
        U, S_, Vh = np.linalg.svd(full)
        B = U[:, : d - 1]                        # complement basis (d, d-1)
        z_perp = B.conj().T @ z                  # project data off w1 subspace
        W0rest = B.conj().T @ W0[:, 1:] if warm else None
        _, Wb, iters, extra = complex_fastica(
            z_perp, W0rest, tol=tol, max_sweeps=max_sweeps, seed=seed,
            return_iters=True, damping=damping, best_iterate=best_iterate,
            perm_invariant=perm_invariant, stagnation=stagnation, return_extra=True)
        W_rest = B @ Wb                          # back to full space (d, d-1)
        W = np.concatenate([w1[:, None], W_rest], axis=1)
        # re-orthonormalize keeping w1 fixed (Gram-Schmidt)
        for j in range(1, d):
            for i in range(j):
                W[:, j] -= (W[:, i].conj() @ W[:, j]) * W[:, i]
            W[:, j] /= np.linalg.norm(W[:, j]) + 1e-30
    else:
        W = w1[:, None]
        iters = 0
        extra = {"stop_reason": "d1", "n_sweeps": 0, "best_sweep": 0, "j_hist": []}
    S = W.conj().T @ z
    # de-scale to unit power convention
    p = np.sqrt(np.mean(np.abs(S) ** 2, axis=1, keepdims=True)) + 1e-30
    S = S / p
    return {"S": S, "W": W, "iters": iters + 50, "warm": warm,
            "anchor_corr": float(np.abs(W[:, 0].conj() @ c)),
            "wall_s": float(time.perf_counter() - t0), "extra": extra}
