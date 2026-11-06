"""IPIX radar data loader.

The files in K:\\All_14p2_Files are NetCDF-3 classic format (magic "CDF\\x01")
despite their .cdf extension (verified: attribute `NetCDF_creation_routine =
make_Dartmouth_ncdf` / `make_Grimsby_ncdf`). Parsed with scipy.io.netcdf_file,
cached to .npz (parse once, per spec R9 engineering notes).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

ATTR_KEYS = [
    "Campaign", "Site", "Data_collection_date", "RF_frequency", "PRF",
    "Pulse_length", "Unambig_velocity", "TX_polarization",
    "Field_dataset_name", "NetCDF_file_name",
]


def campaign_of(path: Path) -> str:
    """Dartmouth 1993 (starea/stareC0000) vs Grimsby 1998 (ANTSTEP)."""
    name = path.name.lower()
    if "antstep" in name:
        return "grimsby98"
    if "stare" in name:
        return "dartmouth93"
    return "unknown"


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("ascii", errors="replace").strip()
    return v


def read_ipix_file(path: str | Path) -> dict:
    """Parse one NetCDF IPIX file.

    Returns dict with:
      data   : dict varname -> np.ndarray
      attrs  : dict attr -> value (global attributes decoded)
      dims   : dict dimname -> int
      campaign, filename
    """
    path = Path(path)
    with netcdf_file(str(path), "r", mmap=False) as f:
        attrs = {k: _decode(v) for k, v in f._attributes.items()}
        dims = dict(f.dimensions)
        data = {}
        for name, var in f.variables.items():
            data[name] = np.array(var.data)  # handles scalars and arrays alike
    return {
        "data": data,
        "attrs": attrs,
        "dims": dims,
        "campaign": campaign_of(path),
        "filename": path.name,
    }


def cache_file(path: str | Path, cache_dir: str | Path) -> Path:
    """Parse file once, store variables + attrs as .npz (attrs as JSON)."""
    path = Path(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / (path.stem + ".npz")
    if out.exists():
        return out
    rec = read_ipix_file(path)
    payload = {f"var:{k}": v for k, v in rec["data"].items()}
    payload["attrs_json"] = np.frombuffer(
        json.dumps({k: str(v) for k, v in rec["attrs"].items()}).encode(), dtype=np.uint8
    )
    payload["dims_json"] = np.frombuffer(
        json.dumps(rec["dims"]).encode(), dtype=np.uint8
    )
    np.savez_compressed(out, **payload)
    return out


def load_cached(path: str | Path, cache_dir: str | Path) -> dict:
    """Load from .npz cache (building it if needed); returns read_ipix_file dict."""
    out = cache_file(path, cache_dir)
    z = np.load(out, allow_pickle=False)
    data = {k[4:]: z[k] for k in z.files if k.startswith("var:")}
    attrs = json.loads(bytes(z["attrs_json"]).decode())
    dims = json.loads(bytes(z["dims_json"]).decode())
    return {
        "data": data,
        "attrs": attrs,
        "dims": dims,
        "campaign": campaign_of(Path(path)),
        "filename": Path(path).name,
    }


# ---------------------------------------------------------------------------
# File-level train/val/test split (R2: file-level, never window-level)
# ---------------------------------------------------------------------------

def file_split(files: list[str | Path], seed: int = 42,
               fractions=(0.6, 0.2, 0.2)) -> dict[str, list[str]]:
    """Stratified-by-campaign file-level split. Returns {'train': [...], 'val': ..., 'test': ...}."""
    rng = np.random.default_rng(seed)
    by_camp: dict[str, list[str]] = {}
    for f in files:
        by_camp.setdefault(campaign_of(Path(f)), []).append(str(f))
    split = {"train": [], "val": [], "test": []}
    for camp, fs in sorted(by_camp.items()):
        fs = sorted(fs)
        rng.shuffle(fs)
        n = len(fs)
        if n >= 5:
            n_tr = int(round(fractions[0] * n))
            n_va = int(round(fractions[1] * n))
        elif n == 4:  # 3/1/0 -> put the odd file in val
            n_tr, n_va = 2, 1
        elif n == 3:
            n_tr, n_va = 1, 1
        elif n == 2:
            n_tr, n_va = 1, 1
        else:
            n_tr, n_va = n, 0
        split["train"] += fs[:n_tr]
        split["val"] += fs[n_tr:n_tr + n_va]
        split["test"] += fs[n_tr + n_va:]
    return split


# ---------------------------------------------------------------------------
# Complex baseband cubes
# ---------------------------------------------------------------------------

def complex_cubes(rec: dict) -> dict[str, np.ndarray]:
    """Build complex baseband cubes from adc_data, ADC-offset corrected.

    Returns dict pol -> complex array (nsweep, nrange) with pol in
    {HH, HV, VH, VV}. TX pol dim 0 = H (like=HH, cross=HV), dim 1 = V
    (like=VV, cross=VH); TX_polarization 'A' = both present.
    """
    d = rec["data"]
    ad = d["adc_data"].astype(np.float32)  # (nsweep, ntxpol, nrange, 4)
    # The adc_*_I/Q variables are small indices (0..3), NOT DC offsets (verified:
    # raw DC is ~-111 counts). Remove per-(txpol, range, adc-channel) temporal mean
    # instead; 1-2% of samples clip at the -128 int8 rail (documented, deviation D9).
    ad = ad - ad.mean(axis=0, keepdims=True)
    like = ad[..., 0] + 1j * ad[..., 1]
    cross = ad[..., 2] + 1j * ad[..., 3]
    out = {}
    if ad.shape[1] >= 2:
        out["HH"], out["HV"] = like[:, 0], cross[:, 0]
        out["VV"], out["VH"] = like[:, 1], cross[:, 1]
    else:
        tx = rec["attrs"].get("TX_polarization", "A").strip().upper()
        if tx == "V":
            out["VV"], out["VH"] = like[:, 0], cross[:, 0]
        else:
            out["HH"], out["HV"] = like[:, 0], cross[:, 0]
    return out


def delay_embed(x: np.ndarray, delta: int = 1) -> np.ndarray:
    """Virtual 2-channel x~(t) = [x(t), x(t-delta)]^T -> (2, nsweep-delta, nrange)."""
    return np.stack([x[delta:], x[:-delta]], axis=0)


def cpi_windows(x: np.ndarray, Np: int, hop: int) -> np.ndarray:
    """(nsweep, nrange) -> (n_win, Np, nrange) sliding windows."""
    n = (x.shape[0] - Np) // hop + 1
    idx = np.arange(Np)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]  # (n_win, Np, nrange)
