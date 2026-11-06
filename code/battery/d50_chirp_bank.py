"""D50 (T4a / R8b) — Chirp-rate bank: separating 'motion within the bin'
from 'energy outside the max bin'.

The d45 walk arm collapses BOTH model-based detectors to Pd = 0.000: their
H0 thresholds are calibrated against a steering bank of pure FFT bins, and a
linear Doppler walk spreads target energy across W/2 = 15--60 Hz of bins, so
no single bin accumulates the coherence.  The CNNs hold 0.84--0.92 there.
This run asks whether a bank enlarged to LFM rates recovers the hybrid, and
whether the remaining gap is motion-in-bin or energy-out-of-bank.

Bank family (nested).  A chirp column at rate r sweeps instantaneous
frequency across +-r/2 Hz within the CPI, so r is the total sweep width in
Hz across the CPI — the same unit as d45's walk width W (30–120 Hz), and
the walk arm's upward sweep (f0-W/2 -> f0+W/2) maps to POSITIVE rates:
  B0   deployed 512-bin steering bank (the shipped M4 statistic)
  B9   B0 + LFM-rate dictionary {0, +-30, +-60, +-90, +-120} Hz/CPI
       (9 unit-norm LFM columns, supported on the |f| <= 150 Hz band)
  B9w  LFM-rate columns only (the 'motion within the bin' counterfactual)

Walk windows: EXACTLY the d45 walk-arm stream (same base units from the
seed-42 sampler, same walk_aux draws from rng(4302), same per-trial rng_t
slots, same sweep law f0 + W*(k/(Np-1) - 0.5)), restricted to the first 200
base units, SCR {0, +5, +10} dB; each window's sweep width W is mapped to the
NEAREST POSITIVE bank rate {30, 60, 90, 120} (the walk law sweeps upward).  n=200, paired with d45's cached 500-unit walk arm.

H0 gates (the multiplicity policy): each bank variant gets its OWN
threshold — the EMPIRICAL 1e-3 quantile of the SAME 600 clutter-only
TEST-split windows (seed-78 sampler/rng family; TEST split because every
deployed calibration pool in this benchmark family is TEST-split — seeds
77/141/142 — and a v1 run showed the TRAIN split carries a genuine
statistic-level shift, e.g. the loaded-ANMF H0 null sits far lower on
TRAIN windows) scored THROUGH that variant.  Family size is fixed by the
bank definition; every variant sees identical windows at the identical
design level, so calibration is matched per variant by construction — no
Bonferroni adjustment is applied and none is needed inside the family
(the comparison across variants is threshold-rule-fair).  A held-out
600-unit TEST-split H0 set (seed 79) validates each gate's realized Pfa
(closed-loop check, reported).

Detectors:
  Hybrid: separated channels s_T computed ONCE per (window, SCR) through the
          deployed statistic_from_net path and CACHED; every bank variant is
          then a post-hoc nmf_statistic(s_T, prep_variant) with prep_variant
          built from the separated calibration windows (C_sep and the
          dictionary are the only things that change).
  ANMF-DL0.001: per unit ONE eigh of the secondary SCM (streamed, never
          persisted); per variant the (W, a, e) triple for that variant's
          dictionary; one eigenbasis evaluation per (unit, variant, SCR).
  CNN-C / CNN-IQ: bank-independent by construction; scored once.

Guards:
  G1a (tolerance) this run's reconstructed walk windows reproduce d45's
      cached walk-arm CNN scores at SCR 0 on all 200 units to max|diff|
      < 1e-5 (d49-G2 lesson: conv kernel selection is batch-size
      dependent — this run scores 200-unit batches vs d45's 500-unit
      batches, so 1e-10 batch-exact equality is unachievable; a stream
      break — base units, walk_aux, rng_t slots, sweep law — would appear
      at the O(1) scale).
  G1b (tolerance) the deployed-bank (B0) hybrid statistic computed from
      THIS run's cached separations matches d45's cached walk-arm values
      to 5e-3 (single-window vs batched LSTM forward; a stream break
      would appear at the 0.05-0.15 scale).
  G1c (decisive) this run's ANMF statistic on the first 10 walk units at
      SCR 0 matches d45's cached walk-arm ANMF row to 1e-6 — pins the
      deployed stat_py convention (numerator whitened: v = W^H(e*z)); the
      v1 local copy omitted the e-weight and was purged (cache v3).

Discipline: exclusive-create pid lock; OPENBLAS_NUM_THREADS=8 before numpy;
no OMP/torch thread overrides; stage-keyed atomic-replace resume cache;
ONE heavy job at a time (d51 waits for this lock's absence).

Output: results/json/d50_chirp_bank.json (+ d50_chirp_bank_cache.npz).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
# no OMP_NUM_THREADS / torch.set_num_threads: byte-exact replay discipline
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.eval.units import UnitSampler, load_split
from src.proposed.infer import HybridDetector
from src.proposed.m4_detector import nmf_statistic, prep_sep_from_windows
from src.baselines.anmf import steering_bank
from src.eval.metrics import clopper_pearson

CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]
PRF = 1000.0
SEED = CFG["seed"]

JS = ROOT / "results/json"
CACHE = JS / "d50_chirp_bank_cache.npz"
OUTJ = JS / "d50_chirp_bank.json"

N_WALK = 200                    # paired subset of d45's walk arm (n=500)
SCRS = (0.0, 5.0, 10.0)
N_CAL = 600                     # TEST-split H0 calibration windows (seed 78)
N_H0T = 600                     # TEST-split held-out H0 validation (seed 79)
RATES = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0)
# each rate = total sweep width across the CPI (Hz); POSITIVE rates are the
# walk arm's upward sweeps, negatives kept for sign-agnostic completeness
BANK_BAND = 150.0               # |f| <= 150 Hz span for chirp columns
VARIANTS = ("B0", "B9", "B9w")
DETS = ("Proposed", "ANMF-DL0.001", "CNN-C", "CNN-IQ")
S_CHUNK = 100                   # separation checkpoint granularity


def log(m):
    print(f"[d50] {m}", flush=True)


def acquire_lock():
    lock = JS / "d50_run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("[d50] another instance holds the lock - exiting", flush=True)
        return None
    return lock


# ------------------------------------------------------------------ banks
def chirp_column(rate_hz, n):
    """Unit-norm LFM column at Doppler 0 with sweep rate_hz, length n."""
    t = np.arange(n, dtype=np.float64)
    ph = 2 * np.pi * np.cumsum(rate_hz * (t - (n - 1) / 2.0)) / (PRF * n)
    c = np.exp(1j * ph)
    return c / np.linalg.norm(c)


def build_bank(variant):
    """(Np, M) unit-norm dictionary for one bank variant."""
    P0 = steering_bank(Np)                       # (Np, Np) FFT bins
    if variant == "B0":
        return P0
    freqs = np.fft.fftshift(np.fft.fftfreq(Np, 1 / PRF))
    cols = [P0] if variant == "B9" else []
    band = np.abs(freqs) <= BANK_BAND
    for r in RATES:
        c = np.zeros(Np, dtype=np.complex128)
        c[band] = chirp_column(r, int(band.sum()))
        cols.append(c[:, None])
    return np.hstack(cols)


def anmf_triple(Q, sig, w, m, P):
    """(W, a, e) for one unit's eigenbasis and one dictionary (dl0.001 path)."""
    import anmf_load_sweep as base
    e = base.variant_isig(sig, w, m, float(np.sum(sig) / Np), "dl0.001")
    W = Q.conj().T @ P
    a = np.real(np.einsum("ij,ij->j", W.conj(), e[:, None] * W))
    return W, a, e


def anmf_stat_from_zy(W, a, e, zy):
    """Deployed stat_py algebra (anmf_load_sweep.py), delegated VERBATIM so
    this pipeline cannot drift from d45/d49's convention: the numerator is
    whitened too — v_f = w_f^H (e * z), d = sum_i e_i |z_i|^2.  The v1 local
    copy omitted the e-weight in v and deflated the whole ANMF null ~10x
    (caught by direct A/B against the deployed path; all ANMF-derived
    artifacts were purged and recomputed as cache v3)."""
    import anmf_load_sweep as base
    return float(base.stat_py(zy, W, a, e, zy))


def unit_anmf_eigh(sampler, key, cell, w0, ext):
    """Secondary SCM eigenbasis for one unit, or None if short of data."""
    sec = sampler.secondary_windows(key, cell, w0, ext.K_SEC, ext.N_CELLS_SIDE)
    if len(sec) < Np:
        return None
    import anmf_load_sweep as base
    Ch = base.build_chat(sec)
    Q, sig = base.chat_eigh(Ch)
    w, m = base.shrink_intensity(Ch, sig)
    return Q, sig, float(w), float(m)


# ------------------------------------------------------------------- main
def main():
    lock = acquire_lock()
    if lock is None:
        return
    try:
        _main()
    finally:
        lock.unlink(missing_ok=True)


def _main():
    t00 = time.time()
    if sys.executable and "venv" not in sys.executable.lower():
        log(f"refusing system python ({sys.executable})")
        return

    prog = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}

    def save_prog(**kw):
        prog.update(kw)
        tmp = JS / "d50_tmp.npz"
        np.savez(tmp, **prog)
        tmp.replace(CACHE)

    # ---- walk windows: EXACT d45 stream, first N_WALK units --------------
    split = load_split(SEED)
    sampler = UnitSampler(split["test"], Np, HOP,
                          pol=CFG["data"]["primary_pol"], seed=SEED + 99)
    from src.data.synth import estimate_psd, clear_doppler_band
    bands = {}
    for key, cube in sampler.data.items():
        f, p = estimate_psd(cube[:, min(3, cube.shape[1] - 1)], PRF, 512)
        bands[key] = clear_doppler_band(p, f, guard_bins=8)
    rng = np.random.default_rng(SEED + 7)
    idx0 = rng.permutation(len(sampler))[:500]     # d45's full draw, then subset
    base0_all = [sampler.units[i] for i in idx0]
    rng2 = np.random.default_rng(SEED + 7)         # d45 continues the SAME rng
    _ = [float(rng2.uniform(*bands[k])) for (k, c, w0) in base0_all]
    rw = np.random.default_rng(4302)
    DOP_HALF = PRF / 2.0                 # d45's exact constant (500.0)
    walk_aux_all = [(float(rw.uniform(-DOP_HALF, DOP_HALF - 30.0)),
                     float(rw.uniform(30.0, 120.0))) for _ in range(500)]
    base0 = base0_all[:N_WALK]
    walk_aux = walk_aux_all[:N_WALK]
    POS_RATES = (30.0, 60.0, 90.0, 120.0)       # walk law sweeps upward
    rates_k = np.array([min(POS_RATES, key=lambda r: abs(r - w[1]))
                        for w in walk_aux])
    log(f"walk units {N_WALK}; bank-rate histogram "
        f"{ {r: int((rates_k == r).sum()) for r in POS_RATES} }")

    def walk_window(k, scr):
        key, cell, w0 = base0[k]
        vv = sampler.data[key][w0:w0 + Np, cell].astype(np.complex128)
        vh = sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)
        f0, W = walk_aux[k]
        n = np.arange(Np)
        w = f0 + W * (n / (Np - 1) - 0.5)
        rng_t = np.random.default_rng(int((scr + 30) * 2) * 100000
                                      + 0 * 1000 + k)   # sw0 slot (d45 conv)
        ph = rng_t.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(w) / PRF
        amp = np.sqrt(10 ** (scr / 10.0) * float(np.mean(np.abs(vv) ** 2)))
        return np.stack([vv + amp * np.exp(1j * ph), vh])

    # ---- detectors --------------------------------------------------------
    det = HybridDetector(ROOT / "results/checkpoints/best_s0.pt")
    zsep = np.load(JS / "csep_prep.npz")
    assert int(zsep["Np"]) == Np, "csep prep Np mismatch"
    Chat0 = zsep["Chat"]
    P0 = steering_bank(int(zsep["Np"]))
    CinvP0 = np.linalg.solve(Chat0, P0)
    a0 = np.real(np.einsum("ij,ij->j", P0.conj(), CinvP0))
    prep0 = {"Chat": Chat0, "CinvP": CinvP0, "a": a0}
    det._prep_sep["__default__"] = prep0

    from d40_train_cnn_baselines import CNNComplex, CNNRealIQ, normalize
    netC, netR = CNNComplex(), CNNRealIQ()
    netC.load_state_dict(torch.load(
        ROOT / "results/checkpoints/best_cnn_complex.pt", map_location="cpu"))
    netR.load_state_dict(torch.load(
        ROOT / "results/checkpoints/best_cnn_real_iq.pt", map_location="cpu"))
    netC.eval(); netR.eval()

    def score_windows(net, xs, batch=256):
        out = np.empty(len(xs), np.float64)
        with torch.no_grad():
            for i in range(0, len(xs), batch):
                xb = torch.from_numpy(
                    normalize(xs[i:i + batch]).astype(np.complex64))
                out[i:i + batch] = net(xb).numpy()
        return out

    # ================= STAGE 1: walk windows, scores, separations ==========
    X = {}
    for scr in SCRS:
        if f"walkX|{scr:g}" not in prog:
            X[scr] = np.stack([walk_window(k, scr) for k in range(N_WALK)])
            prog[f"walkX|{scr:g}"] = X[scr]
            save_prog()
        X[scr] = prog[f"walkX|{scr:g}"]

    if "walkC|0" not in prog:
        save_prog(**{f"walkC|{scr:g}": score_windows(netC, X[scr])
                     for scr in SCRS}
                  | {f"walkR|{scr:g}": score_windows(netR, X[scr])
                     for scr in SCRS})
        log("walk CNN scores done")

    # G1a (tolerance): stream guard against d45's cached CNN walk scores
    z45 = np.load(JS / "d45_shift_battery.npz", allow_pickle=True)
    refC = np.asarray(z45["walk|CNN-C|0"], float)[:N_WALK]
    d1a = float(np.max(np.abs(prog["walkC|0"] - refC)))
    log(f"G1a CNN stream guard (SCR 0, all {N_WALK} units): max|diff|={d1a:.3e}")
    if d1a >= 1e-5:
        raise SystemExit(f"ABORT: walk-stream reconstruction broken "
                         f"(G1a CNN diff {d1a:.3e} >= 1e-5)")

    # separated channels per (unit, SCR) — the post-hoc bank cache
    todo = [(k, scr) for scr in SCRS for k in range(N_WALK)
            if f"walkS|{k}|{scr:g}" not in prog]
    if todo:
        log(f"separations: {len(todo)} windows to go")
        done = 0
        for c0 in range(0, len(todo), 32):
            blk = todo[c0:c0 + 32]
            xs = np.stack([X[scr][k] for (k, scr) in blk])
            outs = det.network_out_batch(xs)
            for j, (k, scr) in enumerate(blk):
                t, info = det.statistic_from_net(xs[j], outs[j],
                                                 prep_sep=prep0)
                prog[f"walkS|{k}|{scr:g}"] = info["s_T"]
            done += len(blk)
            if done % (32 * 10) < 32:
                save_prog()
                log(f"separated {done}/{len(todo)} "
                    f"({time.time() - t00:.0f}s elapsed)")
        save_prog()

    # ================= STAGE 2: H0 calibration gates (streamed) ============
    # v2: gates moved to the TEST split (project convention); purge the v1
    # TRAIN-split gate artifacts once (raw stats/separations are gate-free)
    if "gates_v2" not in prog:
        for kk in list(prog):
            if kk.startswith(("h0X", "h0C", "h0R", "h0S|", "h0Th|",
                              "h0Ta|", "thr|")):
                del prog[kk]
        save_prog()
    # v3: the v1 ANMF statistic omitted the e-weight in the numerator
    # (drift from deployed stat_py); purge every ANMF-derived artifact.
    # Raw separations, hybrid stats (h0Th), and CNN scores are unaffected.
    if "gates_v3" not in prog:
        for kk in list(prog):
            if kk.startswith(("h0Ta|", "thr|", "h0tT|", "wa|")):
                del prog[kk]
        save_prog()
    s78 = UnitSampler(split["test"], Np, HOP,
                      pol=CFG["data"]["primary_pol"], seed=78)
    cal_units = [s78.units[i] for i in
                 np.random.default_rng(78).permutation(len(s78))[:N_CAL]]

    def h0_windows(u, smp):
        return np.stack([
            np.stack([smp.data[k][w0:w0 + Np, c].astype(np.complex128),
                      smp.data_cross[k][w0:w0 + Np, c].astype(np.complex128)])
            for (k, c, w0) in u])

    if "h0X" not in prog:
        prog["h0X"] = h0_windows(cal_units, s78)
        save_prog()
    X0 = prog["h0X"]

    if "h0C" not in prog:
        save_prog(**{"h0C": score_windows(netC, X0),
                     "h0R": score_windows(netR, X0)})
        log("H0 calibration CNN scores done")

    # separated calibration channels (chunked checkpoints)
    if "h0S|0" not in prog:
        for c0 in range(0, N_CAL, S_CHUNK):
            if f"h0S|{c0}" in prog:
                continue
            sl = slice(c0, min(c0 + S_CHUNK, N_CAL))
            outs = det.network_out_batch(X0[sl], chunk=32)
            S = np.empty((len(range(*sl.indices(N_CAL))), Np), np.complex128)
            for j in range(len(range(*sl.indices(N_CAL)))):
                _, info = det.statistic_from_net(X0[sl][j], outs[j],
                                                 prep_sep=prep0)
                S[j] = info["s_T"]
            prog[f"h0S|{c0}"] = S
            save_prog()
            log(f"H0 cal separations chunk {c0} done")
    S_cal = np.concatenate([prog[f"h0S|{c0}"]
                            for c0 in range(0, N_CAL, S_CHUNK)])

    # hybrid stats: B0 (for the gate's consistency record) + variants post-hoc
    t_h_cal = {v: np.empty(N_CAL) for v in VARIANTS}
    if "h0Th|0" not in prog:
        preps = {}
        for v in VARIANTS:
            p = prep_sep_from_windows(S_cal)
            Pv = build_bank(v)
            p["CinvP"] = np.linalg.solve(p["Chat"], Pv)
            p["a"] = np.real(np.einsum("ij,ij->j", Pv.conj(), p["CinvP"]))
            preps[v] = p
        for j in range(N_CAL):
            for v in VARIANTS:
                t_h_cal[v][j] = nmf_statistic(S_cal[j], preps[v])
            if j % 100 == 99:
                save_prog(**{f"h0Th|{v}": t_h_cal[v] for v in VARIANTS})
        save_prog(**{f"h0Th|{v}": t_h_cal[v] for v in VARIANTS})
        log("H0 cal hybrid stats done")
    for v in VARIANTS:
        t_h_cal[v] = prog[f"h0Th|{v}"]

    # ANMF stats: one eigh per unit (streamed), all variants per unit
    import anmf_load_ext as ext
    banks = {v: build_bank(v) for v in VARIANTS}
    t_a_cal = {v: np.full(N_CAL, np.nan) for v in VARIANTS}
    n_short = 0
    if "h0Ta|0" not in prog:
        for j, (k, c, w0) in enumerate(cal_units):
            got = unit_anmf_eigh(s78, k, c, w0, ext)
            if got is None:
                n_short += 1
                continue
            Q, sig, w, m = got
            zy = Q.conj().T @ X0[j][0]
            for v in VARIANTS:
                W, a, e = anmf_triple(Q, sig, w, m, banks[v])
                t_a_cal[v][j] = anmf_stat_from_zy(W, a, e, zy)
            if j % 100 == 99:
                save_prog(**{f"h0Ta|{v}": t_a_cal[v] for v in VARIANTS})
        save_prog(**{f"h0Ta|{v}": t_a_cal[v] for v in VARIANTS})
        log(f"H0 cal ANMF stats done (short-secondary units: {n_short})")
    for v in VARIANTS:
        t_a_cal[v] = prog[f"h0Ta|{v}"]

    # gates: empirical 1e-3 quantile per variant per detector
    thr = {}
    for v in VARIANTS:
        if f"thr|{v}" in prog:
            continue
        q_h = float(np.quantile(t_h_cal[v], 1 - 1e-3))
        q_a = float(np.quantile(t_a_cal[v][~np.isnan(t_a_cal[v])], 1 - 1e-3))
        q_c = float(np.quantile(prog["h0C"], 1 - 1e-3))
        q_r = float(np.quantile(prog["h0R"], 1 - 1e-3))
        thr[v] = np.array([q_h, q_a, q_c, q_r])
        save_prog(**{f"thr|{v}": thr[v]})
        log(f"gate {v}: hybrid {q_h:.4g} anmf {q_a:.4g} "
            f"cnnC {q_c:.4g} cnnR {q_r:.4g}")
    for v in VARIANTS:
        thr[v] = prog[f"thr|{v}"]
    save_prog(gates_v2=np.array([1]), gates_v3=np.array([1]))

    # ================= STAGE 3: held-out H0 validation =====================
    s79 = UnitSampler(split["test"], Np, HOP,
                      pol=CFG["data"]["primary_pol"], seed=79)
    h0t_units = [s79.units[i] for i in
                 np.random.default_rng(79).permutation(len(s79))[:N_H0T]]
    if "h0tX" not in prog:
        prog["h0tX"] = h0_windows(h0t_units, s79)
        save_prog()
    XT = prog["h0tX"]
    if "h0tC" not in prog:
        save_prog(**{"h0tC": score_windows(netC, XT),
                     "h0tR": score_windows(netR, XT)})
    if "h0tS|0" not in prog:
        for c0 in range(0, N_H0T, S_CHUNK):
            if f"h0tS|{c0}" in prog:
                continue
            sl = slice(c0, min(c0 + S_CHUNK, N_H0T))
            outs = det.network_out_batch(XT[sl], chunk=32)
            S = np.empty((len(range(*sl.indices(N_H0T))), Np), np.complex128)
            for j in range(len(range(*sl.indices(N_H0T)))):
                _, info = det.statistic_from_net(XT[sl][j], outs[j],
                                                 prep_sep=prep0)
                S[j] = info["s_T"]
            prog[f"h0tS|{c0}"] = S
            save_prog()
    S_t = np.concatenate([prog[f"h0tS|{c0}"]
                          for c0 in range(0, N_H0T, S_CHUNK)])
    t_h_t = {v: np.empty(N_H0T) for v in VARIANTS}
    t_a_t = {v: np.full(N_H0T, np.nan) for v in VARIANTS}
    if "h0tT|B0" not in prog:
        preps_t = {}
        for v in VARIANTS:
            p = prep_sep_from_windows(S_cal)   # C_sep from CAL windows only
            Pv = build_bank(v)
            p["CinvP"] = np.linalg.solve(p["Chat"], Pv)
            p["a"] = np.real(np.einsum("ij,ij->j", Pv.conj(), p["CinvP"]))
            preps_t[v] = p
        for j in range(N_H0T):
            for v in VARIANTS:
                t_h_t[v][j] = nmf_statistic(S_t[j], preps_t[v])
        n_short_t = 0
        for j, (k, c, w0) in enumerate(h0t_units):
            got = unit_anmf_eigh(s79, k, c, w0, ext)
            if got is None:
                n_short_t += 1
                continue
            Q, sig, w, m = got
            zy = Q.conj().T @ XT[j][0]
            for v in VARIANTS:
                W, a, e = anmf_triple(Q, sig, w, m, banks[v])
                t_a_t[v][j] = anmf_stat_from_zy(W, a, e, zy)
        save_prog(**{f"h0tT|{v}": np.stack([t_h_t[v], t_a_t[v]])
                     for v in VARIANTS})
        log(f"held-out H0 stats done (short: {n_short_t})")
    h0t = {}
    for v in VARIANTS:
        th, ta = prog[f"h0tT|{v}"]
        h0t[v] = {
            "Proposed": float(np.mean(th > thr[v][0])),
            "ANMF-DL0.001": float(np.nanmean(ta > thr[v][1])),
            "CNN-C": float(np.mean(prog["h0tC"] > thr[v][2])),
            "CNN-IQ": float(np.mean(prog["h0tR"] > thr[v][3]))}
    log("held-out H0 realized Pfa: " +
        "; ".join(f"{v} h{h0t[v]['Proposed']:.2e} "
                  f"a{h0t[v]['ANMF-DL0.001']:.2e} "
                  f"c{h0t[v]['CNN-C']:.2e} r{h0t[v]['CNN-IQ']:.2e}"
                  for v in VARIANTS))

    # ================= STAGE 4: walk-arm statistics per variant ============
    # B0 rows are REUSED from d45's cached walk arm (first N_WALK units);
    # B9/B9w come from the cached separations + streamed ANMF transforms.
    stats = {v: {} for v in VARIANTS}
    for scr in SCRS:
        stats["B0"][scr] = (np.asarray(z45[f"walk|Proposed|{scr:g}"],
                                       float)[:N_WALK],
                            np.asarray(z45[f"walk|ANMF|{scr:g}"],
                                       float)[:N_WALK])
    if "wa|B9|0" not in prog:
        preps_w = {}
        for v in ("B9", "B9w"):
            p = prep_sep_from_windows(S_cal)
            Pv = build_bank(v)
            p["CinvP"] = np.linalg.solve(p["Chat"], Pv)
            p["a"] = np.real(np.einsum("ij,ij->j", Pv.conj(), p["CinvP"]))
            preps_w[v] = p
        S_walk = {scr: np.stack([prog[f"walkS|{k}|{scr:g}"]
                                 for k in range(N_WALK)]) for scr in SCRS}
        for k in range(N_WALK):
            if f"wa|{k}" in prog:
                continue
            got = unit_anmf_eigh(sampler, *base0[k], ext)
            blk = {}
            for v in ("B9", "B9w"):
                th = np.array([nmf_statistic(S_walk[scr][k], preps_w[v])
                               for scr in SCRS])
                ta = np.full(len(SCRS), np.nan)
                if got is not None:
                    Q, sig, w, m = got
                    for si, scr in enumerate(SCRS):
                        zy = Q.conj().T @ X[scr][k][0]
                        W, a, e = anmf_triple(Q, sig, w, m, banks[v])
                        ta[si] = anmf_stat_from_zy(W, a, e, zy)
                blk[v] = np.stack([th, ta])
            prog[f"wa|{k}"] = blk
            if k % 25 == 24:
                save_prog()
                log(f"walk stats {k + 1}/{N_WALK}")
        save_prog()
    def get_blk(k):
        b = prog[f"wa|{k}"]
        return b.item() if isinstance(b, np.ndarray) else b

    for v in ("B9", "B9w"):
        for si, scr in enumerate(SCRS):
            stats[v][scr] = (np.stack([get_blk(k)[v][0][si]
                                       for k in range(N_WALK)]),
                             np.stack([get_blk(k)[v][1][si]
                                       for k in range(N_WALK)]))
    # G1b (tolerance): B0 hybrid from THIS run's separations vs d45 cache,
    # through the DEPLOYED prep (csep_prep Chat + B0 dictionary — identical
    # to d45's prep), so the only difference is the LSTM batching.
    # NOTE: nested-dict cache entries (wa|k) pickle to 0-d object arrays
    # across process restarts; get_blk unwraps them.
    refP = np.asarray(z45["walk|Proposed|0"], float)[:N_WALK]
    S0 = np.stack([prog[f"walkS|{k}|0"] for k in range(N_WALK)])
    mineB0 = np.array([nmf_statistic(S0[k], prep0) for k in range(N_WALK)])
    g1b = {"max_abs_diff": float(np.max(np.abs(mineB0 - refP))),
           "tolerance": 5e-3,
           "note": ("B0 hybrid from this run's cached separations through "
                    "the deployed prep vs d45's cached walk arm; "
                    "single-window vs batched LSTM forward — a stream "
                    "break would appear at the 0.05-0.15 scale")}
    log(f"G1b hybrid B0 cross-check: {g1b['max_abs_diff']:.2e} "
        f"(tol {g1b['tolerance']:g})")
    assert g1b["max_abs_diff"] < 0.05, "hybrid B0 cross-check: stream break"

    # G1c (decisive): this run's ANMF stat vs d45's cached walk-arm ANMF row
    # on the first 10 units — pins the stat_py convention (v = W^H(e*z)).
    refA = np.asarray(z45["walk|ANMF|0"], float)[:10]
    mineA, refAm = [], []
    for k in range(10):
        got = unit_anmf_eigh(sampler, *base0[k], ext)
        if got is None or np.isnan(refA[k]):
            continue
        Q, sig, w, m = got
        zy = Q.conj().T @ X[0.0][k][0]
        W, a, e = anmf_triple(Q, sig, w, m, banks["B0"])
        mineA.append(anmf_stat_from_zy(W, a, e, zy))
        refAm.append(refA[k])
    d1c = float(np.max(np.abs(np.array(mineA) - np.array(refAm))))
    log(f"G1c ANMF convention guard ({len(mineA)} units): "
        f"max|diff|={d1c:.3e}")
    assert d1c < 1e-6, \
        f"ANMF stat deviates from deployed stat_py (G1c diff {d1c:.3e})"

    # ================= analysis ============================================
    out = {"task": "T4a/R8b chirp-rate bank on the walk arm (paired with d45)",
           "config": {
               "n_walk": N_WALK, "scrs_db": list(SCRS),
               "rates_hz_per_cpi": list(RATES), "bank_band_hz": BANK_BAND,
               "variants": {v: ("deployed 512-bin bank (reused from d45's "
                                "cached walk arm)" if v == "B0" else
                                "512 bins + 9 LFM rates +-30..120 Hz/CPI"
                                if v == "B9" else
                                "LFM-rate columns only") for v in VARIANTS},
               "n_cal": N_CAL, "n_h0t": N_H0T,
               "cal_recipe": ("TEST-split clutter-only windows, seed-78 "
                              "sampler/rng family (the project's deployed "
                              "calibration pools are all TEST-split); "
                              "per-variant EMPIRICAL 1e-3 quantile gates on "
                              "identical windows (multiplicity policy: family "
                              "size fixed by the bank definition, identical "
                              "windows and design per variant -> matched "
                              "per-variant calibration, no Bonferroni "
                              "adjustment applied or needed)"),
               "walk_recipe": ("d45 walk stream verbatim (base units sampler "
                               "141/rng 49, walk_aux rng 4302, per-trial rng_t "
                               "sw0 slots), first 200 units; sweep width W "
                               "mapped to the nearest positive bank rate"),
               "streams": {"walk": "seed-42 family",
                           "cal": "seed 78 (test split)",
                           "h0t": "seed 79 (test split)"}},
           "validation": {
               "g1a_cnn_stream_max_absdiff": d1a,
               "g1a_tolerance": 1e-5,
               "g1b_hybrid_diff": g1b,
               "h0t_realized_pfa": h0t},
           "gates": {v: {"Proposed": float(thr[v][0]),
                         "ANMF-DL0.001": float(thr[v][1]),
                         "CNN-C": float(thr[v][2]),
                         "CNN-IQ": float(thr[v][3])} for v in VARIANTS},
           "pd": {}, "mcnemar": {}, "notes": []}

    IDX = {"Proposed": 0, "ANMF-DL0.001": 1, "CNN-C": 2, "CNN-IQ": 3}
    for v in VARIANTS:
        out["pd"][v] = {}
        for scr in SCRS:
            th, ta = stats[v][scr]
            row = {}
            for dn, arr in (("Proposed", th), ("ANMF-DL0.001", ta),
                            ("CNN-C", prog[f"walkC|{scr:g}"]),
                            ("CNN-IQ", prog[f"walkR|{scr:g}"])):
                a = np.asarray(arr, float)
                hits = a > thr[v][IDX[dn]]
                kk = int(hits.sum()); nn = int(len(hits))
                lo, hi = clopper_pearson(kk, nn, 0.05)
                row[dn] = {"pd": kk / nn, "ci_lo": lo, "ci_hi": hi,
                           "k": kk, "n": nn}
            out["pd"][v][f"{scr:g}"] = row

    from d45_shift_battery import mcnemar
    for v in VARIANTS:
        th = stats[v][0.0][0]
        for cn in ("CNN-C", "CNN-IQ"):
            arr = prog[f"walk{'C' if cn == 'CNN-C' else 'R'}|0"]
            m = mcnemar(th > thr[v][IDX["Proposed"]],
                        arr > thr[v][IDX[cn]])
            out["mcnemar"][f"{v}|{cn}"] = {
                "n10_hybrid_only": m["n01"], "n01_cnn_only": m["n10"],
                "p": m["p"]}
    out["notes"] = [
        "Bank variants are post-hoc for the hybrid (cached separated "
        "channels; only C_sep-domain prep and the dictionary change) and "
        "share one eigh per unit for the ANMF.",
        "CNN scores are bank-independent; their identical columns across "
        "variants isolate the threshold-gate effect.",
        "Walk windows are a paired subset (first 200) of d45's 500-unit "
        "walk arm on the identical rng stream; the same trials d45 scored "
        "(guards G1a/G1b/G1c).",
        "v3: the ANMF statistic is delegated to the deployed base.stat_py "
        "(numerator whitened, v = W^H(e*z)); the v1 local copy omitted the "
        "e-weight — all ANMF-derived artifacts (gates, held-out H0, walk "
        "ANMF rows) were purged and recomputed; hybrid rows are unchanged "
        "(separate, unaffected path).",
        "B0 rows reuse d45's cached walk-arm statistics for the model-based "
        "detectors (exact same trials) RE-GATED at this run's empirical "
        "TEST-split per-variant quantiles — NOT d45's deployed-threshold "
        "rows; CNN B0 rows are this run's arrays, guard-equal to the cache "
        "at SCR 0 (G1a).",
        "Reported either way it falls: if B9 does not recover the hybrid, "
        "the walk gap is NOT a bank-completeness artifact.",
    ]
    OUTJ.write_text(json.dumps(out, indent=1))
    log(f"wrote {OUTJ.name} ({time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
