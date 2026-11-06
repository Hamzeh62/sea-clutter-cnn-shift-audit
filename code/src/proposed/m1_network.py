"""M1 — complex-valued network backbone (spec §4.4 M1).

Complex ops via equivalent real 2-channel blocks (DeepComplexNets conventions):
- ComplexConv1d: block-matrix weight [Wr -Wi; Wi Wr] on stacked (re, im) channels
- Complex BatchNorm: BatchNorm on real & imag parts
- modReLU: ReLU(|z|+b) * z/|z|
- cross-channel attention: magnitude-based SE gates
- CV-LSTM: real LSTM over 2C-dim stacked representation (equivalent block form)

Input: (B, M, Np) complex -> represented as real (B, 2M, Np).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def c_to_r(x: torch.Tensor) -> torch.Tensor:
    """(B, C, L) complex -> (B, 2C, L) real stacked."""
    return torch.cat([x.real, x.imag], dim=1)


def r_to_c(x: torch.Tensor) -> torch.Tensor:
    B, C2, L = x.shape
    return torch.complex(x[:, : C2 // 2], x[:, C2 // 2:])


class ComplexConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride=1, padding=0):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.conv = nn.Conv1d(2 * cin, 2 * cout, k, stride=stride, padding=padding,
                              bias=False)
        nn.init.normal_(self.conv.weight, std=0.05)

    def forward(self, x):  # x real (B, 2cin, L)
        B, C2, L = x.shape
        W = self.conv.weight  # (2cout, 2cin, k)
        Wo, Wi = W[: self.cout], W[self.cout:]
        # build block weight [Wr -Wi; Wi Wr]
        Wr = torch.cat([Wo[:, : self.cin], -Wi[:, self.cin:]], dim=1)
        Wi_ = torch.cat([Wi[:, : self.cin], Wo[:, self.cin:]], dim=1)
        Wblock = torch.cat([Wr, Wi_], dim=0)
        return F.conv1d(x, Wblock, None, self.conv.stride, self.conv.padding)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(2 * c)

    def forward(self, x):
        return self.bn(x)


class ModReLU(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.b = nn.Parameter(torch.full((c,), -0.05))

    def forward(self, x):  # (B, 2C, L)
        C2 = x.shape[1]
        C = C2 // 2
        re, im = x[:, :C], x[:, C:]
        mag = torch.sqrt(re * re + im * im + 1e-12)
        gated = F.relu(mag + self.b.view(1, C, 1)) / mag
        return torch.cat([re * gated, im * gated], dim=1)


class ChannelAttention(nn.Module):
    """Magnitude squeeze-excitation across (complex) channels."""

    def __init__(self, c: int, red: int = 4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // red), nn.ReLU(),
                                nn.Linear(c // red, c), nn.Sigmoid())

    def forward(self, x):  # (B, 2C, L)
        C2 = x.shape[1]
        C = C2 // 2
        re, im = x[:, :C], x[:, C:]
        mag = torch.sqrt(re * re + im * im + 1e-12).mean(dim=2)  # (B, C)
        g = self.fc(mag).view(B := x.shape[0], C, 1)
        return torch.cat([re * g, im * g], dim=1)


class CVLSTM(nn.Module):
    """Complex LSTM as real LSTM over stacked 2C dims (block-equivalent)."""

    def __init__(self, c: int, h: int):
        super().__init__()
        self.lstm = nn.LSTM(2 * c, 2 * h, batch_first=True)
        self.h = h

    def forward(self, x):  # (B, 2C, L) -> (B, 2h, L)
        seq = x.transpose(1, 2)          # (B, L, 2C)
        out, _ = self.lstm(seq)
        return out.transpose(1, 2)


class M1Backbone(nn.Module):
    """conv64-k7 -> CBN -> modReLU -> conv128-k5-s2 -> CBN -> modReLU ->
    channel attention -> CV-LSTM(128).  Output features (B, 256, L')."""

    def __init__(self, M: int, chans=(64, 128), ks=(7, 5), hidden=128):
        super().__init__()
        self.conv1 = ComplexConv1d(M, chans[0], ks[0], padding=ks[0] // 2)
        self.bn1 = ComplexBatchNorm1d(chans[0])
        self.mod1 = ModReLU(chans[0])
        self.conv2 = ComplexConv1d(chans[0], chans[1], ks[1], stride=2,
                                   padding=ks[1] // 2)
        self.bn2 = ComplexBatchNorm1d(chans[1])
        self.mod2 = ModReLU(chans[1])
        self.attn = ChannelAttention(chans[1])
        self.lstm = CVLSTM(chans[1], hidden)
        self.out_dim = 2 * hidden

    def forward(self, x_complex):  # (B, M, Np) complex
        x = c_to_r(x_complex)
        x = self.mod1(self.bn1(self.conv1(x)))
        x = self.mod2(self.bn2(self.conv2(x)))
        x = self.attn(x)
        x = self.lstm(x)
        return x  # real (B, 2h, L')
