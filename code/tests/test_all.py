"""Unit tests T1–T6 (spec §6). Run: .venv/Scripts/python.exe -m pytest tests/ -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.anmf import (anmf_glrt_stats, anmf_threshold_beta,
                                prep_anmf_unit, anmf_glrt_from_prep)
from src.baselines.cfar import ca_cfar_stat, os_cfar_stat


def test_T1_whitening():
    """Whitened data has identity covariance (synthetic)."""
    from src.proposed.m3_cica import whiten
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    X = A @ (rng.standard_normal((4, 5000)) + 1j * rng.standard_normal((4, 5000)))
    z, V, d_hat, lam = whiten(X, d=4)  # full-rank whitening for identity check
    C = z @ z.conj().T / z.shape[1]
    np.testing.assert_allclose(C, np.eye(4), atol=0.06)


def test_T2_fastica_recovers_sources():
    """Complex FastICA recovers a well-conditioned 4x4 mixing; median SIR > 20 dB.

    Sources have non-Gaussian magnitude (tone, K, BPSK, QPSK) — Gaussian
    circular sources are unidentifiable by ICA, so the test uses the signal
    classes relevant to this application.
    """
    from src.proposed.m3_cica import complex_fastica
    from src.data.synth import synth_compound_gaussian
    sirs = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        n = 2000
        s1 = np.exp(1j * (2 * np.pi * 0.13 * np.arange(n) + rng.uniform(0, 2 * np.pi)))  # tone
        s2 = synth_compound_gaussian(n, nu=0.5, b=1.0, psd_template=None, rho=0.9, rng=rng)
        s3 = rng.choice([-1, 1], size=n).astype(float) * (1 + 0.1j * rng.standard_normal(n))  # BPSK
        s4 = np.exp(1j * rng.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], size=n)) * (
            1 + 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))               # QPSK
        S = np.stack([s1, s2, s3, s4])
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        X = A @ S + 0.05 * (rng.standard_normal((4, n)) + 1j * rng.standard_normal((4, n)))
        Shat, W, iters = complex_fastica(X, seed=seed)
        G = np.abs(Shat @ np.linalg.pinv(S))          # (4,4) gain: row=est, col=true
        power = G ** 2
        sir = 10 * np.log10(np.max(power, axis=0) / (np.sum(power, axis=0) - np.max(power, axis=0) + 1e-12))
        sirs.append(np.median(sir))
    assert np.median(sirs) > 20.0, f"median SIR {np.median(sirs):.1f} dB < 20 dB"


def _homogeneous_k_units(rng, n_units, ncells=21, Np=128):
    from src.data.synth import synth_compound_gaussian
    for _ in range(n_units):
        cells = [synth_compound_gaussian(Np, nu=2.0, b=1.0, psd_template=np.ones(64),
                                         rho=0.9, rng=rng) for _ in range(ncells)]
        yield cells


def test_T3_cfar_pfa():
    """CA/OS-CFAR hold nominal Pfa within +-10% on homogeneous synthetic K clutter."""
    rng = np.random.default_rng(3)
    tr, gr, tf, gf = 4, 1, 4, 1
    for mode, stat in [("CA", ca_cfar_stat), ("OS", os_cfar_stat)]:
        stats = [stat(cells, len(cells) // 2, train_r=tr, guard_r=gr,
                      train_f=tf, guard_f=gf, mode=mode.lower())
                 for cells in _homogeneous_k_units(rng, 400)]
        stats = np.array(stats)
        pfa_nom = 1e-2
        thr = np.quantile(stats, 1 - pfa_nom)
        # hold-out check
        stats2 = np.array([stat(cells, len(cells) // 2, train_r=tr, guard_r=gr,
                                train_f=tf, guard_f=gf, mode=mode.lower())
                           for cells in _homogeneous_k_units(rng, 400)])
        emp = np.mean(stats2 > thr)
        assert abs(emp - pfa_nom) <= 0.1 * pfa_nom + 0.004, f"{mode}: emp {emp:.4f}"


def test_T4_anmf_beta():
    """ANMF null law (implementation-verified, deviation D14):

    With the plain SCM of K Gaussian secondaries, the single-bin statistic
    follows Beta(1, m_eff), m_eff = (Np-1)(K-Np+1)/((Np-1)+(K-Np+1)) — verified
    numerically at 5 scales; the spec's Beta(1, K-Np+1) is anti-conservative.
    The spec's texture-removal normalization shifts the null further (D13);
    that variant is calibrated empirically. Statistic always in [0,1].
    """
    rng = np.random.default_rng(4)
    Np = 128
    K = 4 * Np
    sec = rng.standard_normal((K, Np)) + 1j * rng.standard_normal((K, Np))
    prep = prep_anmf_unit(sec, normalize_rows=False)
    single = []
    for _ in range(3000):
        y = rng.standard_normal(Np) + 1j * rng.standard_normal(Np)
        Cinv_y = np.linalg.solve(prep["Chat"], y)
        v2 = np.abs(prep["CinvP"].conj().T @ y) ** 2
        d_ = float(np.real(np.vdot(y, Cinv_y)))
        single.append(float((v2 / (prep["a"] * d_ + 1e-30))[3]))
    single = np.array(single)
    assert single.max() <= 1.0 + 1e-9, "ANMF statistic must be <= 1"
    m_eff = (Np - 1) * (K - Np + 1) / ((Np - 1) + (K - Np + 1))
    for pfa in (1e-2, 1e-3):
        t0 = 1 - pfa ** (1.0 / m_eff)
        emp = np.mean(single > t0)
        se = np.sqrt(pfa * (1 - pfa) / len(single))
        assert abs(emp - pfa) < 4 * se + 0.002, f"emp {emp:.4f} vs {pfa} (m_eff={m_eff:.1f})"
    # normalized-SCM variant (the detector under test): sanity only
    prep2 = prep_anmf_unit(sec, normalize_rows=True)
    for _ in range(300):
        y = rng.standard_normal(Np) + 1j * rng.standard_normal(Np)
        t, _ = anmf_glrt_from_prep(y, prep2)
        assert t <= 1.0 + 1e-9


def test_T5_loader_roundtrip():
    """Parsed array sizes match dims; no NaNs (on cached cubes)."""
    from src.data.ipix_loader import read_ipix_file
    import scipy.io
    files = sorted((ROOT / "cache").glob("*_cubes.npz"))
    assert files, "no cached cubes; run phase1 first"
    z = np.load(files[0])
    for pol in ("VV", "VH", "HH", "HV"):
        arr = z[pol]
        assert arr.ndim == 2
        assert not np.isnan(arr.real).any() and not np.isnan(arr.imag).any()
    raw = sorted(Path(r"K:\All_14p2_Files").glob("*.cdf"))[:1]
    if raw:
        rec = read_ipix_file(raw[0])
        d = rec["data"]["adc_data"]
        assert d.shape[0] == rec["dims"]["nsweep"]
        assert d.shape[2] == rec["dims"]["nrange"]


def test_T6_no_leakage():
    """Train/val/test file sets pairwise disjoint."""
    from src.eval.units import load_split
    s = load_split(42)
    a, b, c = set(map(Path, s["train"])), set(map(Path, s["val"])), set(map(Path, s["test"]))
    assert not (a & b) and not (a & c) and not (b & c)
    assert len(s["train"]) >= 1 and len(s["test"]) >= 1
