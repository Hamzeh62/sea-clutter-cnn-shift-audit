"""Test-unit infrastructure shared by calibration and Monte-Carlo evaluation.

A unit = (file, range cell, CPI start index). The unit sampler draws units from
clutter-only cells of given split files, excluding target-bearing cells and
adjacent guards (from results/json/target_cells.json inference).

All five detectors consume the SAME unit context (R5 pairing):
  - detection-channel CPI vector y (Np,) from the primary pol,
  - range-neighbor block for CFAR (contiguous cells around CUT),
  - K secondary windows for ANMF/GLRT (adjacent cells, same CPI index,
    staggered offsets, unit-normalized -> texture removal).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.ipix_loader import campaign_of

ROOT = Path(__file__).resolve().parents[2]


def load_split(seed: int = 42) -> dict[str, list[str]]:
    from src.data.ipix_loader import file_split
    cache = ROOT / "cache"
    files = sorted(set(list(cache.glob("*_cubes.npz"))))
    raw = [str(p) for p in files]
    return file_split(raw, seed=seed)


def target_cells_map() -> dict[str, list[int]]:
    """file stem (no _cubes.npz) -> list of excluded cell indices (targets + 1 guard)."""
    tc = json.loads((ROOT / "results/json/target_cells.json").read_text())
    out = {}
    for key, info in tc.items():
        stem = key.replace(".cdf", "").replace(".CDF", "")
        cands = [i for i, z in enumerate(info["power_zscore"]) if z > 1.0]
        excl = set()
        for c in cands:
            excl.update({c - 1, c, c + 1})  # primary + guard neighbors
        out[stem] = sorted(x for x in excl if 0 <= x)
    return out


class UnitSampler:
    def __init__(self, split_files: list[str], Np: int, hop: int,
                 pol: str = "VV", cross_pol: str | None = "VH",
                 guard_cells: int = 1, seed: int = 0):
        self.Np, self.hop, self.pol = Np, hop, pol
        self.rng = np.random.default_rng(seed)
        tc = target_cells_map()
        self.units = []      # (file_key, cell, win0)
        self.data = {}
        self.data_cross = {}
        for f in split_files:
            key = Path(f).stem
            z = np.load(f)
            cube = z[pol]                     # (nsweep, nrange)
            ncell = cube.shape[1]
            n_win = (cube.shape[0] - Np) // hop + 1
            excl = set(tc.get(key, []))
            for c in range(ncell):
                if c in excl:
                    continue
                for w in range(n_win):
                    self.units.append((key, c, w * hop))
            self.data[key] = cube
            self.data_cross[key] = z[cross_pol] if cross_pol else cube
        self.n_win_per_cell = {k: (v.shape[0] - Np) // hop + 1 for k, v in self.data.items()}

    def __len__(self):
        return len(self.units)

    def draw(self) -> tuple[str, int, int]:
        return self.units[self.rng.integers(len(self.units))]

    def y_cpi(self, key: str, cell: int, w0: int, inject=None) -> np.ndarray:
        """Primary-pol CPI vector; optional injection applied post-hoc."""
        y = self.data[key][w0:w0 + self.Np, cell].astype(np.complex128)
        return inject(y) if inject else y

    def range_block(self, key: str, cell: int, w0: int, half: int,
                    inject_cell: int | None = None, inject=None) -> tuple[list[np.ndarray], int]:
        """Contiguous range block of CPI vectors centered on CUT, CUT index returned.
        `inject` applied only to the injected cell (or CUT if inject_cell None)."""
        cube = self.data[key]
        ncell = cube.shape[1]
        lo = max(0, cell - half)
        hi = min(ncell, cell + half + 1)
        cells = [cube[w0:w0 + self.Np, c].astype(np.complex128) for c in range(lo, hi)]
        tgt = cell if inject_cell is None else inject_cell
        if inject is not None and lo <= tgt < hi:
            i = tgt - lo
            cells[i] = inject(cells[i])
        return cells, cell - lo

    def secondary_windows(self, key: str, cell: int, w0: int, K: int,
                          cells_each_side: int = 4) -> np.ndarray:
        """K unit-norm secondary windows from adjacent cells, same CPI index,
        staggered hop offsets around w0 (texture-removed by normalization)."""
        cube = self.data[key]
        ncell = cube.shape[1]
        cand_cells = [c for c in range(cell - cells_each_side, cell + cells_each_side + 1)
                      if c != cell and 0 <= c < ncell]
        per_cell = int(np.ceil(K / max(1, len(cand_cells))))
        offs = np.arange(per_cell) - per_cell // 2  # staggered around w0
        wins = []
        for c in cand_cells:
            for o in offs:
                s = w0 + o * self.hop
                if s < 0 or s + self.Np > cube.shape[0]:
                    continue
                w = cube[s:s + self.Np, c].astype(np.complex128)
                wins.append(w)
                if len(wins) >= K:
                    return np.stack(wins)
        return np.stack(wins[:K]) if wins else np.empty((0, self.Np), np.complex128)
