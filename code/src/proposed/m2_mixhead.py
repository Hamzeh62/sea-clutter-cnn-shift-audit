"""M2 — mixing-matrix head + reference head + optional detection head (spec §4.4).

Network outputs raw Z in C^{M x d}; A_hat = Q R with Q = Z(Z^H Z)^{-1/2}
(Newton-Schulz polar projection, differentiable) and R upper-triangular with
diagonal e^{u + j phi} (guarantees full column rank).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .m1_network import M1Backbone


def torch_newton_schulz_polar(Z: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Q = Z (Z^H Z)^{-1/2}, differentiable Newton-Schulz (complex)."""
    X = Z / (torch.norm(Z) + 1e-12)
    for _ in range(iters):
        X = 0.5 * (3 * X - X @ (X.conj().transpose(-2, -1) @ X))
    return X


class M2Heads(nn.Module):
    def __init__(self, M: int, d: int, Np: int, backbone: M1Backbone,
                 det_head: bool = False):
        super().__init__()
        self.backbone = backbone
        feat = backbone.out_dim
        self.M, self.d = M, d
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_Z = nn.Linear(feat, 2 * M * d)        # raw Z re+im
        self.fc_diag = nn.Linear(feat, 2 * d)         # u and phi per source
        n_tri = d * (d - 1) // 2
        self.fc_tri = nn.Linear(feat, 2 * n_tri)      # strict upper triangle
        self.fc_ref = nn.Linear(feat, 2 * Np)         # reference r(t)
        self.det_head = det_head
        if det_head:
            self.fc_det = nn.Linear(feat, 1)

    def forward(self, x):
        """x: (B, M, Np) complex -> dict(A, Q, R, r, Z, [det])."""
        feat = self.pool(self.backbone(x)).squeeze(-1)      # (B, F)
        B = feat.shape[0]
        zr, zi = self.fc_Z(feat).chunk(2, dim=-1)
        Z = 0.1 * torch.complex(zr, zi).view(B, self.M, self.d)
        Q = torch_newton_schulz_polar(Z)                    # (B, M, d) isometry
        u, phi = self.fc_diag(feat).chunk(2, dim=-1)
        diag = torch.exp(u + 1j * phi)
        R = torch.zeros(B, self.d, self.d, dtype=diag.dtype)
        idx = torch.arange(self.d)
        R[:, idx, idx] = diag
        tri_r, tri_i = self.fc_tri(feat).chunk(2, dim=-1)
        k = 0
        for i in range(self.d):
            for j in range(i + 1, self.d):
                R[:, i, j] = 0.1 * torch.complex(tri_r[:, k], tri_i[:, k])
                k += 1
        A = Q @ R                                           # full column rank
        rr = self.fc_ref(feat)
        r = torch.complex(rr[:, : rr.shape[1] // 2],
                          rr[:, rr.shape[1] // 2:])
        out = {"A": A, "Q": Q, "R": R, "r": r, "Z": Z}
        if self.det_head:
            out["det"] = self.fc_det(feat).squeeze(-1)
        return out
