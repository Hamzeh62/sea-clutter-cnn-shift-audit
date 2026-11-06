"""Training losses (spec §4.4): L = L_det + lA L_A + lr L_ref + l1 L_indep +
l2 L_phys + l3 L_recon + l4 L_div, with curriculum ramp on l1, l2."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def g_log1p(u: torch.Tensor) -> torch.Tensor:
    return torch.log1p(u)


def negentropy_approx(y: torch.Tensor) -> torch.Tensor:
    """J(y) approx with G(u)=log(1+u), u=|y|^2: [E G(|y|^2) - E G(|g|^2)]^2 style
    (Gaussian reference cancels in the sum over units; we use the raw E G term
    scaled, consistent across units)."""
    u = (y.real ** 2 + y.imag ** 2)
    return (g_log1p(u).mean()) ** 2


def mixing_loss(Q: torch.Tensor, a_T: torch.Tensor, t_col: int = 0) -> torch.Tensor:
    """L_A^(T) = || q_t - a_T ||^2 on unit-norm columns."""
    q = Q[..., t_col]
    a = a_T / (a_T.abs().norm(dim=-1, keepdim=True) + 1e-12)
    return (q - a).abs().pow(2).sum(dim=-1).mean()


def ref_loss(s_T: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """L_ref = 1 - |E[s r*]| / sqrt(E|s|^2 E|r|^2).

    NOTE (D3 audit): algebraically this is already the DIRECT cosine between r
    and the ground-truth s_T: |E[s r*]|/sqrt(E|s|^2 E|r|^2) = |<r, s_T>|/(||r|| ||s_T||).
    It is applied to ALL windows; on clutter windows s_T = 0 so the term is a
    constant 1 (no gradient, pure dilution of the target-window gradient).
    """
    corr = (s_T * r.conj()).mean(dim=-1).abs()
    es = (s_T.abs() ** 2).mean(dim=-1)
    er = (r.abs() ** 2).mean(dim=-1)
    return (1 - corr / torch.sqrt(es * er + 1e-12)).mean()


def ref_dir_loss(r: torch.Tensor, s_T: torch.Tensor,
                 y: torch.Tensor | None = None) -> torch.Tensor:
    """L_ref_dir = 1 - |<r, s_T>| / (||r|| ||s_T||) per window, weight 1 (D3).

    Masked to TARGET windows (y = 1): on clutter windows s_T = 0 and the term
    is a constant, so it only dilutes the objective.
    """
    ip = (r * s_T.conj()).sum(dim=-1).abs()
    nr = r.abs().pow(2).sum(dim=-1).sqrt()
    ns = s_T.abs().pow(2).sum(dim=-1).sqrt()
    cos = ip / (nr * ns + 1e-12)
    if y is not None:
        mask = y > 0.5
        if mask.any():
            return (1 - cos[mask]).mean()
        return cos.sum() * 0.0
    return (1 - cos).mean()


def indep_loss(S_pre: torch.Tensor) -> torch.Tensor:
    """L_indep = -(1/d) sum_j J(s_j) (negentropy proxy)."""
    d = S_pre.shape[1]
    return -sum(negentropy_approx(S_pre[:, j]) for j in range(d)) / d


_LOGK_CACHE: dict = {}


def _logK_grid(nu: float, b: float, zmin=1e-6, zmax=200.0, n=4000):
    key = (round(nu, 4), round(b, 4))
    if key not in _LOGK_CACHE:
        import scipy.special as sp
        z = np.geomspace(zmin, zmax, n)
        logk = np.log(sp.kv(abs(nu - 1.0), z) + 1e-300)
        _LOGK_CACHE[key] = (np.log(z), logk)
    return _LOGK_CACHE[key]


def phys_loss(s_pre_C: torch.Tensor, nu: float, b: float) -> torch.Tensor:
    """L_phys = -mean ln p_K(|s_pre,C|); differentiable in |s| via piecewise-linear
    interpolation of a precomputed log-Bessel grid (fixed nu, b per run)."""
    import scipy.special as sp
    lz_grid, logk_grid = _logK_grid(nu, b)
    lz_grid_t = torch.from_numpy(lz_grid).to(s_pre_C.real.device)
    logk_grid_t = torch.from_numpy(logk_grid).to(s_pre_C.real.device)
    r = s_pre_C.abs().clamp_min(1e-8)
    z = b * r / 2.0
    lz = torch.log(z)
    h = lz_grid_t[1] - lz_grid_t[0]
    idx = torch.clamp(((lz - lz_grid_t[0]) / h).round().long(), 0, len(lz_grid_t) - 1)
    logk = logk_grid_t[idx]
    lgamma = sp.gammaln(nu)
    log_pdf = float(np.log(2 * b) - lgamma) + nu * torch.log(z) + logk
    return -log_pdf.mean()


def recon_loss(X: torch.Tensor, A: torch.Tensor, S_pre: torch.Tensor) -> torch.Tensor:
    """L_recon = mean ||x - A s_pre||^2 with s_pre = pinv(A) x."""
    return (X - A @ S_pre).abs().pow(2).mean()


def div_loss(S_pre: torch.Tensor) -> torch.Tensor:
    """L_div = || (1/N) sum s s^H - I ||_F^2."""
    C = (S_pre @ S_pre.conj().transpose(-2, -1)) / S_pre.shape[-1]
    eye = torch.eye(C.shape[-1], dtype=C.dtype, device=C.device)
    return ((C - eye).norm(dim=(-2, -1)) ** 2).mean()


import numpy as np  # noqa: E402  (used in phys_loss)


def total_loss(out: dict, X: torch.Tensor, S_pre: torch.Tensor, s_T: torch.Tensor,
               a_T: torch.Tensor, label: torch.Tensor, K_params=(0.5, 1.0),
               lambdas=(1.0, 0.5, 1.0, 0.1, 1.0, 0.1),
               ref_mode: str = "old", lambda_r_dir: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Assemble the total loss. X: (B,M,Np), S_pre: (B,d,Np), s_T: (B,Np).

    ref_mode (D3 ablation):
      "old"  = legacy L_ref only (weight lambda_r)          [S0/D2 behaviour]
      "dir"  = L_ref_dir on target windows, weight 1, old term dropped
      "both" = legacy L_ref (lambda_r) + L_ref_dir (weight 1)
    """
    lA, lr, l1, l2, l3, l4 = lambdas
    l_det = F.binary_cross_entropy_with_logits(out["det"], label)
    if ref_mode == "old":
        l_ref = lr * ref_loss(s_T, out["r"])
    elif ref_mode == "dir":
        l_ref = lambda_r_dir * ref_dir_loss(out["r"], s_T, label)
    elif ref_mode == "both":
        l_ref = lr * ref_loss(s_T, out["r"]) + lambda_r_dir * ref_dir_loss(out["r"], s_T, label)
    else:
        raise ValueError(f"unknown ref_mode: {ref_mode}")
    parts = {
        "det": l_det,
        "A": lA * mixing_loss(out["Q"], a_T),
        "ref": l_ref,
        "indep": l1 * indep_loss(S_pre),
        "phys": l2 * phys_loss(S_pre[:, -1], *K_params),
        "recon": l3 * recon_loss(X, out["A"], S_pre),
        "div": l4 * div_loss(S_pre),
    }
    return sum(parts.values()).real, parts
