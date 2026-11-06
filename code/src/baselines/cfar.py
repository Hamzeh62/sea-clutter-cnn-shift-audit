"""2-D CA-CFAR and OS-CFAR on the range–Doppler power map (spec §4.3).

Test unit = (range cell, CPI window). Doppler processing: Hann-windowed FFT of
length Np over the CPI. Statistic: t = max_f |X(r,f)|^2 / denom(r,f) with the
denominator estimated per Doppler bin from training cells around the CUT in
range and Doppler (guards excluded; windows clamped at map edges — the 14-cell
Dartmouth files cannot fully fit 8+2 cells per side, so training size varies
with cell position; empirical threshold calibration absorbs the level shift).
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann


def rd_map_2d(win: np.ndarray) -> np.ndarray:
    """win: (Np, ncells) complex CPI block -> power map (Np, ncells) |FFT|^2 (fftshifted)."""
    Np = win.shape[0]
    w = hann(Np, sym=False)[:, None]
    X = np.fft.fftshift(np.fft.fft(win * w, axis=0), axes=0)
    return np.abs(X) ** 2


def _training_tensor(sub: np.ndarray, cut_col: int, guard_r: int, train_r: int,
                     guard_f: int, train_f: int) -> np.ndarray:
    """sub: (n_f, n_cells_local) power map around CUT. Returns (n_f, n_train_max)
    training samples per Doppler bin (padded with NaN)."""
    n_f, n_c = sub.shape
    r_lo = max(0, cut_col - guard_r - train_r)
    r_hi = min(n_c, cut_col + guard_r + train_r + 1)
    g_lo = max(0, cut_col - guard_r)
    g_hi = min(n_c, cut_col + guard_r + 1)
    keep_cols = np.ones(n_c, bool)
    keep_cols[g_lo:g_hi] = False

    out = np.full((n_f, (2 * (guard_f + train_f) + 1) * (r_hi - r_lo)), np.nan)
    for f in range(n_f):
        f_lo = max(0, f - guard_f - train_f)
        f_hi = min(n_f, f + guard_f + train_f + 1)
        block = sub[f_lo:f_hi, r_lo:r_hi]
        mask2d = np.ones(block.shape, bool)
        # guard rows around f (clamped): rows f-gf..f+gf map into block coords
        gf_lo = f - guard_f - f_lo
        gf_hi = f + guard_f + 1 - f_lo
        mask2d[max(0, gf_lo):min(block.shape[0], gf_hi), :] = False
        mask2d[:, max(0, g_lo - r_lo):min(block.shape[1], g_hi - r_lo)] = False
        vals = block[mask2d]
        out[f, : vals.size] = vals
    return out


def cfar_statistic(cells: list[np.ndarray], cut_idx: int, train_r: int,
                   guard_r: int, train_f: int, guard_f: int, mode: str = "ca",
                   os_frac: float = 0.75) -> float:
    """t = max_f |X(r,f)|^2 / denom(f).

    cells: contiguous range block of CPI snapshots (list of (Np,) complex),
    CUT at position cut_idx inside the block (edges clamped).
    """
    win = np.stack(cells, axis=1)          # (Np, ncells)
    pmap = rd_map_2d(win)                  # (Np, ncells)
    tr = _training_tensor(pmap, cut_idx, guard_r, train_r, guard_f, train_f)
    num = pmap[:, cut_idx]
    if mode == "ca":
        denom = np.nanmean(tr, axis=1)
    elif mode == "os":
        n_train = np.sum(~np.isnan(tr), axis=1)
        k = np.maximum(1, np.ceil(os_frac * n_train).astype(int))
        srt = np.sort(tr, axis=1)  # NaNs sort to the end
        rows = np.arange(tr.shape[0])
        last_valid = n_train - 1
        denom = srt[rows, np.minimum(k - 1, last_valid)]
    else:
        raise ValueError(mode)
    return float(np.max(num / (denom + 1e-30)))


def ca_cfar_stat(cells, cut_idx, **kw) -> float:
    kw.setdefault("mode", "ca")
    return cfar_statistic(list(cells), cut_idx, **kw)


def os_cfar_stat(cells, cut_idx, **kw) -> float:
    kw.setdefault("mode", "os")
    return cfar_statistic(list(cells), cut_idx, **kw)
