"""D45 (task R8) — Supervised-baseline shift battery + mechanism probes.

Question: does the D40 in-distribution CNN dominance survive distribution
shift of the H1 target model?  All arms run the SAME paired seed-42 sw0-grid
MC (sampler 141, rng 49, per-trial rng_t; the d38/d40-validated stream) with
H0 thresholds UNCHANGED (shifts are H1-side only):

  indist   Swerling-0, f_D in the file's clear-Doppler band  (= D40 anchor)
  sw1/sw2  Swerling-1 / Swerling-2 fluctuation (phase-4 rng_t slot convention)
  fullband f_D ~ U(-250, +250) Hz incl. the clutter ridge
  walk     linear Doppler walk (chirp), sweep width W ~ U(30,120) Hz over CPI
  gauss5/10/20  finite spectral width: carrier + Gaussian linewidth sigma_f
  logn     lognormal amplitude fluctuation across CPI (sigma_ln = 1.0)
  spike    H1 trials on the spikiest nu-hat tercile units
  flat     H1 trials on the smoothest nu-hat tercile units

Detectors: CNN-C, CNN-IQ (d40 checkpoints + thresholds), Proposed hybrid
(deployed phase-4 thresholds from prop_h0_stats), ANMF-DL0.001 (deployed
thresholds from anmf_load_ext.json).  Saved byte-identical-trial statistics
are REUSED: hybrid sw0 <- v1_repro_sw0.json, ANMF sw0/sw1/sw2 <-
anmf_load_ext_progress.npz stats_sw{0,1,2}[:, :, 1] (dl0.001 eigen path).
Everything else is computed here.

Mechanism probes:
  occlusion   zero the |f_D| +-3-bin Doppler region (VV) of in-distribution
              H1 windows -> tone-driven evidence vs elsewhere
  power       clutter-only windows with a coherent-free +3 dB gain: CNN score
              shift vs the NMF statistic (scale invariance = the control)
  noise       white complex noise injected at matched power (no spectral line)

D40 presentation fixes recorded in the analysis (defects found during recon):
  (a) the d40 "phase2_anmf4000" CNN subsets were indexed by unit index
      instead of pool position (misaligned subset -> corrected here);
  (b) CNN-R 1e-4 rows re-thresholded at the realized quantile of the full
      21,462-unit phase-2 pool (empirical, n*pfa >= 10).

Outputs: results/json/d45_shift_battery.json (+ npz stage cache).  Resume-safe
at arm x detector x SCR granularity; stage keys are "arm|det" / "mech|...".
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
# NOTE: no OMP_NUM_THREADS and no torch.set_num_threads here — the byte-exact
# replay scripts (phase4, d38, d43) run with torch's default intra-op thread
# count; overriding it changes conv/LAPACK reduction order and breaks the
# 1e-10 v1 replay guard (observed: 81/500 mismatches).
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch
from tqdm import tqdm
from scipy.stats import binomtest, lognorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")

from src.eval.units import UnitSampler, load_split
from src.eval.metrics import clopper_pearson
from src.data.synth import swerling_target, estimate_psd, clear_doppler_band
from src.proposed.infer import HybridDetector

CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]
PRF = 1000.0
SEED = CFG["seed"]
SCR_GRID = CFG["eval"]["scr_grid_db"]
N_MC = CFG["eval"]["n_mc"]
CI = CFG["eval"]["ci"]

JS = ROOT / "results/json"
CACHE = JS / "d45_shift_battery.npz"
OUTJ = JS / "d45_shift_battery.json"

sys.path.insert(0, str(ROOT / "scripts"))
from d40_train_cnn_baselines import CNNComplex, CNNRealIQ, normalize  # noqa: E402

PFA_KEYS = ("0.001", "0.0001")
DOP_HALF = PRF / 2.0                     # 250 Hz full band
ARMS = ["indist", "sw1", "sw2", "fullband", "walk", "gauss5", "gauss10",
        "gauss20", "logn", "spike", "flat"]
DETS = ("Proposed", "ANMF-DL0.001", "CNN-C", "CNN-IQ")
KEYD = {"Proposed": "Proposed", "ANMF-DL0.001": "ANMF",
        "CNN-C": "CNN-C", "CNN-IQ": "CNN-IQ"}


# ------------------------------------------------------------------ helpers
def score_windows(net, xs, batch=256, desc=""):
    out = np.empty(len(xs), np.float64)
    with torch.no_grad():
        for i in tqdm(range(0, len(xs), batch), desc=desc, leave=False,
                      total=(len(xs) + batch - 1) // batch):
            xb = torch.from_numpy(normalize(xs[i:i + batch]).astype(np.complex64))
            out[i:i + batch] = net(xb).numpy()
    return out


def load_nets():
    netC, netR = CNNComplex(), CNNRealIQ()
    netC.load_state_dict(torch.load(ROOT / "results/checkpoints/best_cnn_complex.pt",
                                    map_location="cpu"))
    netR.load_state_dict(torch.load(ROOT / "results/checkpoints/best_cnn_real_iq.pt",
                                    map_location="cpu"))
    netC.eval(); netR.eval()
    return netC, netR


def pd_rows(arr, thr):
    hits = arr > thr
    kd = int(hits.sum())
    lo, hi = clopper_pearson(kd, len(hits), 1 - CI)
    return {"pd": kd / len(hits), "ci_lo": lo, "ci_hi": hi, "k": kd, "n": int(len(hits))}


def mcnemar(a, b):
    """Exact two-sided McNemar on paired boolean hit arrays."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    p = float(binomtest(min(b01, b10), n, 0.5).pvalue) if n else 1.0
    return {"n01": b01, "n10": b10, "n_pairs": int(len(a)), "p": p}


def load_nu_hat_map():
    st_ = json.loads((JS / "clutter_stats.json").read_text())
    nmap = {}
    for fkey, fval in st_.items():
        stem = fkey[:-4] if fkey.endswith(".npz") else fkey
        for ci, c in enumerate(fval["cells"]):
            kk = c.get("K") or c.get("K_polished") or {}
            v = kk.get("nu")
            if v is not None and 0.05 < v < 100:
                nmap[(stem, ci)] = v
    return nmap


def mk_target(arm, scr_db, vv, rng, aux):
    """Target waveform s_T for one shifted trial (phase-4 per-trial rng).

    aux = shared per-unit arm-level draw (carrier / (f0, W) / None), made
    once per arm so all four detectors see identical targets.
    """
    m2 = float(np.mean(np.abs(vv) ** 2))
    amp = np.sqrt(10 ** (scr_db / 10.0) * m2)
    N = len(vv)
    k = np.arange(N)
    if arm in ("indist", "spike", "flat"):
        return swerling_target(N, aux, PRF, amp, 0, rng)          # aux = f_d
    if arm in ("sw1", "sw2"):
        return swerling_target(N, aux, PRF, amp, int(arm[2]), rng)
    if arm == "fullband" or arm.startswith("gauss"):
        return swerling_target(N, aux, PRF, amp, 0, rng)          # aux = carrier
    if arm == "walk":
        f0, W = aux
        w = f0 + W * (k / (N - 1) - 0.5)                          # +-W/2 sweep
        ph = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(w) / PRF
        return amp * np.exp(1j * ph)
    if arm == "logn":
        a = float(lognorm(s=1.0).rvs(random_state=rng)) / np.sqrt(np.e)
        ph = rng.uniform(0, 2 * np.pi) + 2 * np.pi * aux * k / PRF
        return amp * a * np.exp(1j * ph)          # E[a^2] = 1 (unit mean power)
    raise ValueError(f"unknown arm {arm}")


def gauss_line_target(scr_db, vv, rng, f_center, sigma_f):
    """Carrier + Gaussian spectral linewidth sigma_f (finite-width target)."""
    m2 = float(np.mean(np.abs(vv) ** 2))
    amp = np.sqrt(10 ** (scr_db / 10.0) * m2)
    w = rng.normal(0.0, sigma_f, len(vv))
    ph = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(w + f_center) / PRF
    return amp * np.exp(1j * ph)


# --------------------------------------------------------------------- main
def main():
    t00 = time.time()
    # environment guard: the phantom second launch runs system python, which
    # has different BLAS/threads than the project venv -> refuse to run there
    if sys.executable and "venv" not in sys.executable.lower():
        print(f"[d45] refusing system python ({sys.executable}); "
              f"run with .venv/Scripts/python.exe", flush=True)
        return
    # single-job guard: second harness launch exits immediately
    # (exclusive-create so simultaneous double launches cannot both win)
    lock = JS / "d45_run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("[d45] another instance holds the lock - exiting", flush=True)
        return
    try:
        _main(t00)
    finally:
        lock.unlink(missing_ok=True)


def _main(t00):
    netC, netR = load_nets()
    det = HybridDetector(ROOT / "results/checkpoints/best_s0.pt")
    zsep = np.load(JS / "csep_prep.npz")
    from src.baselines.anmf import steering_bank
    import anmf_load_sweep as base
    import anmf_load_ext as ext
    P = steering_bank(int(zsep["Np"]))
    Chat = zsep["Chat"]
    CinvP = np.linalg.solve(Chat, P)
    a = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))
    prep_sep = {"Chat": Chat, "CinvP": CinvP, "a": a}
    det._prep_sep["__default__"] = prep_sep

    # ---- deployed thresholds (unchanged H0 calibration) --------------------
    thrP = {p: float(np.quantile(np.load(JS / "prop_h0_stats.npz")["stats"],
                                 1 - float(p)))
            for p in PFA_KEYS}
    extj = json.loads((JS / "anmf_load_ext.json").read_text())
    thrA = {k.split("|")[1]: float(v["threshold"])
            for k, v in extj["h0_thresholds"].items() if k.startswith("dl0.001|")}
    d40j = json.loads((JS / "d40_cnn_baselines.json").read_text())
    thrC = {n: {p: float(d40j["thresholds"][n][p]["threshold"]) for p in PFA_KEYS}
            for n in ("CNN-C", "CNN-IQ")}
    print(f"[d45] deployed thresholds: hybrid {thrP}, anmf {thrA}, "
          f"cnnC {thrC['CNN-C']}, cnnR {thrC['CNN-IQ']}", flush=True)

    # ---- base units + f_d (rng 49 stream, identical to phase-4/d38/d40) ----
    split = load_split(SEED)
    sampler = UnitSampler(split["test"], Np, HOP, pol=CFG["data"]["primary_pol"],
                          seed=SEED + 99)
    bands = {}
    for key, cube in sampler.data.items():
        f, p = estimate_psd(cube[:, min(3, cube.shape[1] - 1)], PRF, 512)
        bands[key] = clear_doppler_band(p, f, guard_bins=8)
    rng = np.random.default_rng(SEED + 7)
    idx0 = rng.permutation(len(sampler))[:N_MC]
    base0 = [sampler.units[i] for i in idx0]
    fds0 = [float(rng.uniform(*bands[key])) for (key, cell, w0) in base0]
    print(f"[d45] {len(base0)} base units; clear-band f_d mean "
          f"{np.mean(fds0):+.1f} Hz", flush=True)

    # ---- nu-hat terciles on the base units ---------------------------------
    nu_map = load_nu_hat_map()
    cell_nu = np.array([nu_map.get((key, cell), np.nan) for key, cell, w0 in base0])
    valid = ~np.isnan(cell_nu)
    t1, t2 = np.percentile(cell_nu[valid], [100 / 3, 200 / 3])
    m_spike = valid & (cell_nu <= t1)
    m_flat = valid & (cell_nu > t2)
    print(f"[d45] nu-hat matched {valid.sum()}/{N_MC}; terciles {t1:.3f}/{t2:.3f}; "
          f"spike n={m_spike.sum()}, flat n={m_flat.sum()}", flush=True)

    # ---- per-unit ANMF transforms (cached) ----------------------------------
    anmf_cache = {}

    def anmf_stat(key, cell, w0, y):
        ck = (key, cell, w0)
        if ck not in anmf_cache:
            sec = sampler.secondary_windows(key, cell, w0, ext.K_SEC,
                                            ext.N_CELLS_SIDE)
            anmf_cache[ck] = ext.unit_transforms_sel(base.build_chat(sec)) \
                if len(sec) >= Np else (None, None)
        ut, Q_u = anmf_cache[ck]
        if ut is None:
            return np.nan
        W, aa, e = ut["dl0.001"]
        zy = Q_u.conj().T @ y
        return float(base.stat_py(y, W, aa, e, zy))

    # ---- stage cache ---------------------------------------------------------
    prog = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}

    def save_prog(**kw):
        prog.update(kw)
        np.savez(CACHE, **prog)

    # ---- per-arm shared draws ------------------------------------------------
    fds_full = np.random.default_rng(4301).uniform(-DOP_HALF, DOP_HALF, N_MC)
    rw = np.random.default_rng(4302)
    walk_aux = [(float(rw.uniform(-DOP_HALF, DOP_HALF - 30.0)),
                 float(rw.uniform(30.0, 120.0))) for _ in range(N_MC)]

    def arm_window(arm, k, scr, fd_override=None, sigma=None):
        """(2, Np) H1 window for unit k, or None if the unit is out of arm scope."""
        key, cell, w0 = base0[k]
        if arm == "spike" and not m_spike[k]:
            return None
        if arm == "flat" and not m_flat[k]:
            return None
        vv = sampler.data[key][w0:w0 + Np, cell].astype(np.complex128)
        vh = sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)
        sw_slot = int(arm[2]) if arm in ("sw1", "sw2") else 0
        rng_t = np.random.default_rng(int((scr + 30) * 2) * 100000
                                      + sw_slot * 1000 + k)
        if arm.startswith("gauss"):
            s_T = gauss_line_target(scr, vv, rng_t, float(fds_full[k]),
                                    float(arm[5:]))
        else:
            if arm in ("indist", "spike", "flat", "logn", "sw1", "sw2"):
                aux = fds0[k]            # clear-band carrier (phase-4 convention)
            elif arm == "walk":
                aux = walk_aux[k]        # (f0, W) per-unit chirp draws
            else:                        # fullband
                aux = fds_full[k]
            s_T = mk_target(arm, scr, vv, rng_t, aux)
        return np.stack([vv + s_T, vh])

    zext = np.load(JS / "anmf_load_ext_progress.npz")
    v1 = json.loads((JS / "v1_repro_sw0.json").read_text())
    z40 = np.load(JS / "d40_scores_trials.npz")

    def run_arm(arm):
        """Compute raw stats for all four detectors on one arm.

        Saved byte-identical sources are reused inside the per-SCR branches:
        hybrid {0,+10} dB indist validated against v1 (replay guard), CNN indist
        grids taken from d40_scores_trials, ANMF indist/sw1/sw2 from the ext pool.
        """
        for k_scr, scr in enumerate(tqdm(SCR_GRID, desc=f"{arm} grid",
                                         leave=False)):
            need = [d for d in DETS
                    if f"{arm}|{KEYD[d]}|{scr:g}" not in prog]
            if not need:
                continue
            xs, idx = [], []
            for k in range(N_MC):
                x = arm_window(arm, k, scr)
                if x is not None:
                    idx.append(k)
                    xs.append(x)
            if not idx:
                raise RuntimeError(f"arm {arm}: no valid units")
            xs_v = np.stack(xs)
            if "Proposed" in need:
                outs = det.network_out_batch(xs_v)
                st = np.full(N_MC, np.nan)
                for j, i in enumerate(idx):
                    tp, _ = det.statistic_from_net(xs_v[j], outs[j],
                                                   prep_sep=prep_sep)
                    st[i] = tp
                if arm == "indist" and f"{scr:g}" in ("0", "10"):
                    ref = np.asarray(v1["stats"][f"{scr:g}"]["Proposed"], float)
                    mism = int(np.sum(~np.isclose(st, ref, rtol=1e-10, atol=1e-12)))
                    print(f"[d45] guard indist hybrid @{scr:g} dB: "
                          f"{mism} mismatches vs v1", flush=True)
                    if mism:
                        raise SystemExit(
                            f"ABORT: indist hybrid replay @{scr:g} dB mismatches v1")
                save_prog(**{f"{arm}|Proposed|{scr:g}": st})
            if "CNN-C" in need:
                if arm == "indist":
                    sc = z40["c"][k_scr]
                else:
                    sc = np.full(N_MC, np.nan)
                    sc[idx] = score_windows(netC, xs_v, desc=f"{arm} {scr:+g}dB C")
                save_prog(**{f"{arm}|CNN-C|{scr:g}": sc})
            if "CNN-IQ" in need:
                if arm == "indist":
                    sr = z40["r"][k_scr]
                else:
                    sr = np.full(N_MC, np.nan)
                    sr[idx] = score_windows(netR, xs_v, desc=f"{arm} {scr:+g}dB R")
                save_prog(**{f"{arm}|CNN-IQ|{scr:g}": sr})
            if "ANMF-DL0.001" in need:
                if arm in ("indist", "sw1", "sw2"):
                    swk = {"indist": "sw0"}.get(arm, arm)
                    sa = zext[f"stats_{swk}"][1, k_scr, :]   # (mode, SCR, trial)
                else:
                    sa = np.full(N_MC, np.nan)
                    for j, i in enumerate(idx):
                        key, cell, w0 = base0[i]
                        sa[i] = anmf_stat(key, cell, w0, xs_v[j][0])
                save_prog(**{f"{arm}|ANMF|{scr:g}": sa})

    # ---- run all arms ----------------------------------------------------------
    for arm in ARMS:
        t0 = time.time()
        run_arm(arm)
        done = sum(f"{arm}|{KEYD[d]}|{s:g}" in prog
                   for d in DETS for s in SCR_GRID)
        print(f"[d45] {arm}: {done}/4 det stat arrays cached "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- mechanism probe 1: occlusion of the tone region ------------------------
    for scr in (0.0, 5.0, 10.0):
        if f"mech|occ|hyb|{scr:g}" in prog:
            continue
        xs = np.stack([arm_window("indist", k, scr) for k in range(N_MC)])
        fd = np.array(fds0)
        bins = np.fft.fftshift(np.fft.fftfreq(Np, 1 / PRF))
        mask = np.abs(bins[None, :] - fd[:, None]) <= 3.0     # +-3 bins around f_D
        xs_occ = xs.copy()
        xs_occ[:, 0, :][mask] = 0.0
        outs, outs_occ = det.network_out_batch(xs), det.network_out_batch(xs_occ)
        dh = np.zeros((2, N_MC))
        for k in range(N_MC):
            ta, _ = det.statistic_from_net(xs[k], outs[k], prep_sep=prep_sep)
            tb, _ = det.statistic_from_net(xs_occ[k], outs_occ[k], prep_sep=prep_sep)
            dh[0, k], dh[1, k] = ta, tb
        save_prog(**{f"mech|occ|hyb|{scr:g}": dh,
                     f"mech|occ|cnnC|{scr:g}": score_windows(netC, xs,
                                                             desc=f"occ{scr:g} C"),
                     f"mech|occ|cnnC_occ|{scr:g}": score_windows(netC, xs_occ,
                                                                 desc=f"occ{scr:g} Cocc"),
                     f"mech|occ|cnnR|{scr:g}": score_windows(netR, xs,
                                                             desc=f"occ{scr:g} R"),
                     f"mech|occ|cnnR_occ|{scr:g}": score_windows(netR, xs_occ,
                                                                 desc=f"occ{scr:g} Rocc")})
        print(f"[d45] occlusion {scr:+g} dB done", flush=True)

    # ---- mechanism probe 2: coherent-free +3 dB power boost (clutter-only) ------
    if "mech|pow|hyb" not in prog:
        xs0 = np.stack([np.stack([
            sampler.data[key][w0:w0 + Np, cell].astype(np.complex128),
            sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)])
            for (key, cell, w0) in base0])
        xsb = xs0.copy()
        xsb[:, 0, :] *= 10 ** (3 / 20.0)                       # +3 dB, no coherence
        outs0, outsb = det.network_out_batch(xs0), det.network_out_batch(xsb)
        ph = np.zeros((2, N_MC))
        for k in range(N_MC):
            ta, _ = det.statistic_from_net(xs0[k], outs0[k], prep_sep=prep_sep)
            tb, _ = det.statistic_from_net(xsb[k], outsb[k], prep_sep=prep_sep)
            ph[0, k], ph[1, k] = ta, tb
        save_prog(**{"mech|pow|hyb": ph,
                     "mech|pow|cnnC": np.stack([score_windows(netC, xs0, desc="pow C0"),
                                               score_windows(netC, xsb, desc="pow C+3")]),
                     "mech|pow|cnnR": np.stack([score_windows(netR, xs0, desc="pow R0"),
                                               score_windows(netR, xsb, desc="pow R+3")])})
        print("[d45] power probe done", flush=True)

    # ---- mechanism probe 3: white complex noise at matched power ----------------
    for scr in (0.0, 10.0):
        if f"mech|noise|hyb|{scr:g}" in prog:
            continue
        xs = []
        for k in range(N_MC):
            key, cell, w0 = base0[k]
            vv = sampler.data[key][w0:w0 + Np, cell].astype(np.complex128)
            vh = sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)
            rng_t = np.random.default_rng(88000 + int(scr) * 1000 + k)
            m2 = float(np.mean(np.abs(vv) ** 2))
            amp = np.sqrt(10 ** (scr / 10.0) * m2)
            noise = amp * (rng_t.standard_normal(Np)
                           + 1j * rng_t.standard_normal(Np)) / np.sqrt(2)
            xs.append(np.stack([vv + noise, vh]))
        xs = np.stack(xs)
        outs = det.network_out_batch(xs)
        nh = np.array([det.statistic_from_net(xs[k], outs[k], prep_sep=prep_sep)[0]
                       for k in range(N_MC)])
        save_prog(**{f"mech|noise|hyb|{scr:g}": nh,
                     f"mech|noise|cnnC|{scr:g}": score_windows(netC, xs,
                                                               desc=f"noi{scr:g} C"),
                     f"mech|noise|cnnR|{scr:g}": score_windows(netR, xs,
                                                               desc=f"noi{scr:g} R")})
        print(f"[d45] noise probe {scr:+g} dB done", flush=True)

    # ---- per-tercile realized Pfa: clutter-only windows from the tercile units --
    # (the spike/flat H1 arms carry the tercile power comparison; this stage
    # measures the false-alarm side on the SAME units, no injection)
    for tname, mask in (("spike", m_spike), ("flat", m_flat)):
        if f"tercpfa|{tname}" in prog:
            continue
        idx_t = np.flatnonzero(mask)
        xs = np.stack([
            np.stack([
                sampler.data[key][w0:w0 + Np, cell].astype(np.complex128),
                sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)])
            for (key, cell, w0) in (base0[i] for i in idx_t)])
        outs = det.network_out_batch(xs)
        dh = np.array([det.statistic_from_net(xs[j], outs[j],
                                              prep_sep=prep_sep)[0]
                       for j in range(len(idx_t))])
        da = np.array([anmf_stat(*base0[i], xs[j][0])
                       for j, i in enumerate(idx_t)])
        save_prog(**{f"tercpfa|{tname}": np.stack([dh, da]),
                     f"tercpfa|{tname}|cnnC": score_windows(netC, xs,
                                                            desc=f"tcp {tname} C"),
                     f"tercpfa|{tname}|cnnR": score_windows(netR, xs,
                                                            desc=f"tcp {tname} R")})
        print(f"[d45] tercile-H0 {tname} (n={len(idx_t)}) done", flush=True)

    # ================= analysis ==================================================
    out = {"config": {
        "n_mc": N_MC, "scr_grid": SCR_GRID, "seed": SEED,
        "recipe": ("phase-4 seed-42 stream (sampler 141, rng 49, per-trial rng_t "
                   "slot convention); base units + clear-band f_d shared with "
                   "v1_repro_sw0 / anmf_load_ext / d40; H0 thresholds unchanged "
                   "(shifts are H1-side only)"),
        "thresholds": {
            "Proposed": thrP,
            "ANMF-DL0.001": thrA,
            "CNN-C": thrC["CNN-C"], "CNN-IQ": thrC["CNN-IQ"]},
        "threshold_provenance": {
            "Proposed": "empirical quantiles of the 1500-unit phase-4 H0 pool",
            "ANMF-DL0.001": "deployed extrapolated thresholds (anmf_load_ext.json)",
            "CNNs": "d40 phase-4-rule thresholds (seed-77 pool; 1e-4 extrapolated)"},
        "arms": {a: {"sw_slot": int(a[2]) if a in ("sw1", "sw2") else 0}
                 for a in ARMS},
        "nu_terciles": {"t1": float(t1), "t2": float(t2),
                        "spike_n": int(m_spike.sum()), "flat_n": int(m_flat.sum())},
        "walk_width_hz": [30.0, 120.0], "sigma_f_hz": [5.0, 10.0, 20.0],
        "doppler_half_band_hz": DOP_HALF},
        "validation": {}, "pd": {}, "robustness": {}, "realized_pfa": {},
        "mechanism": {}, "mcnemar": {}, "d40_fixes": {}, "runtime_s": {}}

    # guards: in-distribution arm must reproduce the saved references exactly
    for name, ref in (("CNN-C", z40["c"]), ("CNN-IQ", z40["r"])):
        diffs = [float(np.nanmax(np.abs(prog[f"indist|{name}|{s:g}"] - ref[i])))
                 for i, s in enumerate(SCR_GRID)
                 if f"indist|{name}|{s:g}" in prog]
        out["validation"][f"indist_{name}_max_absdiff"] = float(max(diffs))
        assert out["validation"][f"indist_{name}_max_absdiff"] == 0.0, \
            f"{name} in-dist drift"
    for scr in ("0", "10"):
        if f"indist|Proposed|{scr}" in prog:
            d = float(np.nanmax(np.abs(
                prog[f"indist|Proposed|{scr}"]
                - np.asarray(v1["stats"][scr]["Proposed"], float))))
            out["validation"][f"indist_Proposed_{scr}dB_max_absdiff"] = d
            assert d < 1e-10, "indist hybrid drift"
    for arm in ("sw0", "sw1", "sw2"):
        ka = "indist" if arm == "sw0" else arm
        diffs = [float(np.nanmax(np.abs(prog[f"{ka}|ANMF|{s:g}"]
                                        - zext[f"stats_{arm}"][1, i, :])))
                 for i, s in enumerate(SCR_GRID)
                 if f"{ka}|ANMF|{s:g}" in prog]
        if diffs:
            out["validation"][f"{ka}_ANMF_max_absdiff"] = float(max(diffs))
            assert out["validation"][f"{ka}_ANMF_max_absdiff"] == 0.0, \
                f"{ka} ANMF drift"
    print(f"[d45] guards passed: {out['validation']}", flush=True)

    def stats_arr(arm, detn):
        """Assemble (17, N_MC) stats from per-SCR cache keys."""
        return np.stack([np.asarray(prog[f"{arm}|{KEYD[detn]}|{s:g}"], float)
                         for s in SCR_GRID])

    # Pd rows + paired McNemar (each detector vs the hybrid) per arm
    for arm in ARMS:
        out["pd"][arm] = {}
        for detn in DETS:
            arr = stats_arr(arm, detn)
            thr = (thrP["0.001"] if detn == "Proposed"
                   else thrA["0.001"] if detn == "ANMF-DL0.001"
                   else thrC[detn]["0.001"])
            rows = {}
            for k_scr, scr in enumerate(SCR_GRID):
                col = arr[k_scr]
                rows[f"{scr:g}"] = pd_rows(col[~np.isnan(col)], thr)
            out["pd"][arm][detn] = rows
        out["mcnemar"][arm] = {}
        for detn in ("ANMF-DL0.001", "CNN-C", "CNN-IQ"):
            hd_arr, hp_arr = stats_arr(arm, detn), stats_arr(arm, "Proposed")
            thr_d = thrA["0.001"] if detn == "ANMF-DL0.001" else thrC[detn]["0.001"]
            vm = ~(np.isnan(hd_arr) | np.isnan(hp_arr))
            out["mcnemar"][arm][detn] = {}
            for i in range(len(SCR_GRID)):
                out["mcnemar"][arm][detn][f"{SCR_GRID[i]:g}"] = mcnemar(
                    hp_arr[i][vm[i]], (hd_arr[i] > thr_d)[vm[i]])

    # robustness profile: mean Pd over the operating region (SCR >= -10 dB)
    idxs = [i for i, s in enumerate(SCR_GRID) if s >= -10]
    for detn in DETS:
        prof = {}
        for arm in ARMS:
            a_i = float(np.mean([out["pd"]["indist"][detn][f"{SCR_GRID[i]:g}"]["pd"]
                                 for i in idxs]))
            a_s = float(np.mean([out["pd"][arm][detn][f"{SCR_GRID[i]:g}"]["pd"]
                                 for i in idxs]))
            prof[arm] = {"mean_pd_indist": a_i, "mean_pd_shift": a_s,
                         "rel_degradation": float((a_i - a_s) / max(a_i, 1e-12))}
        out["robustness"][detn] = prof

    # realized Pfa per nu-hat tercile: clutter-only windows from the tercile
    # units (the tercpfa stage above), 1e-3 threshold, Clopper-Pearson CIs
    for tname in ("spike", "flat"):
        for detn, col in (("Proposed", 0), ("ANMF-DL0.001", 1)):
            arr = np.asarray(prog[f"tercpfa|{tname}"][col], float)
            thr = (thrP["0.001"] if detn == "Proposed" else thrA["0.001"])
            out["realized_pfa"][f"{tname}|{detn}"] = pd_rows(arr, thr)
        out["realized_pfa"][f"{tname}|CNN-C"] = pd_rows(
            np.asarray(prog[f"tercpfa|{tname}|cnnC"], float),
            thrC["CNN-C"]["0.001"])
        out["realized_pfa"][f"{tname}|CNN-IQ"] = pd_rows(
            np.asarray(prog[f"tercpfa|{tname}|cnnR"], float),
            thrC["CNN-IQ"]["0.001"])

    # mechanism: occlusion
    out["mechanism"]["occlusion"] = {}
    for scr in (0.0, 5.0, 10.0):
        dh = prog[f"mech|occ|hyb|{scr:g}"]
        for name, key in (("CNN-C", "cnnC"), ("CNN-IQ", "cnnR")):
            s0 = prog[f"mech|occ|{key}|{scr:g}"]
            s1 = prog[f"mech|occ|{key}_occ|{scr:g}"]
            out["mechanism"]["occlusion"][f"{name}@{scr:g}"] = {
                "hybrid_mean": float(np.nanmean(dh[0])),
                "hybrid_occ_mean": float(np.nanmean(dh[1])),
                "hybrid_rel_drop": float(np.nanmean(dh[0] - dh[1])
                                         / max(np.nanmean(dh[0]), 1e-30)),
                "cnn_mean": float(np.mean(s0)),
                "cnn_occ_mean": float(np.mean(s1)),
                "cnn_rel_drop": float(np.mean(s0 - s1) / max(np.mean(s0), 1e-30)),
                "cnn_hits_thr": int(np.sum(s0 > thrC[name]["0.001"])),
                "cnn_occ_hits_thr": int(np.sum(s1 > thrC[name]["0.001"]))}
    # mechanism: power boost
    ph = prog["mech|pow|hyb"]
    out["mechanism"]["power_boost_3db"] = {
        "nmf_invariance_max_absdiff": float(np.nanmax(np.abs(ph[1] - ph[0]))),
        "nmf_invariance_median_absdiff": float(np.median(np.abs(ph[1] - ph[0]))),
        "nmf_hits_base_vs_boost": [int((ph[0] > thrP["0.001"]).sum()),
                                   int((ph[1] > thrP["0.001"]).sum())],
        "cnnC_score_mean": [float(np.mean(prog["mech|pow|cnnC"][0])),
                            float(np.mean(prog["mech|pow|cnnC"][1]))],
        "cnnR_score_mean": [float(np.mean(prog["mech|pow|cnnR"][0])),
                            float(np.mean(prog["mech|pow|cnnR"][1]))],
        "cnnC_rate_above_thr": [float(np.mean(prog["mech|pow|cnnC"][0]
                                              > thrC["CNN-C"]["0.001"])),
                                float(np.mean(prog["mech|pow|cnnC"][1]
                                              > thrC["CNN-C"]["0.001"]))],
        "cnnR_rate_above_thr": [float(np.mean(prog["mech|pow|cnnR"][0]
                                              > thrC["CNN-IQ"]["0.001"])),
                                float(np.mean(prog["mech|pow|cnnR"][1]
                                              > thrC["CNN-IQ"]["0.001"]))]}
    # mechanism: noise surrogate
    out["mechanism"]["noise_surrogate"] = {}
    for scr in (0.0, 10.0):
        nh = prog[f"mech|noise|hyb|{scr:g}"]
        out["mechanism"]["noise_surrogate"][f"{scr:g}db"] = {
            "hybrid_hits": int(np.sum(nh > thrP["0.001"])),
            "cnnC_hits": int(np.sum(prog[f"mech|noise|cnnC|{scr:g}"]
                                    > thrC["CNN-C"]["0.001"])),
            "cnnR_hits": int(np.sum(prog[f"mech|noise|cnnR|{scr:g}"]
                                    > thrC["CNN-IQ"]["0.001"])),
            "n": int(len(nh))}

    # D40 presentation fixes (computed authoritatively by d45b; cite here)
    d45b = json.loads((JS / "d45b_d40_pfa_fixes.json").read_text())
    out["d40_fixes"] = {
        "anmf4000_subset_misaligned": True,
        "note": ("d40 stage_phase2 indexed full_c/full_r by unit index instead "
                 "of pool position; the subset is statistically similar but "
                 "mislabeled. Corrected values in d45b_d40_pfa_fixes.json "
                 "(inverse-permutation pool-position mapping)."),
        "cnnI_rethreshold_1e-4": d45b["d40_defect_b"]["rethreshold"]["CNN-IQ@0.0001"],
        "cnnC_1e-4_check": {
            "threshold_deployed": thrC["CNN-C"]["0.0001"],
            "rate_full_pool": d45b["d40_defect_a"]["realized_pfa_phase2_full"]["CNN-C@0.0001"]},
        "realized_pfa_phase2_full": d45b["d40_defect_a"]["realized_pfa_phase2_full"]}

    out["runtime_s"]["total"] = time.time() - t00
    OUTJ.write_text(json.dumps(out, indent=2))
    print(f"[d45] wrote {OUTJ.name} ({(time.time()-t00)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    import traceback
    # own append-mode log with pid attribution: the harness launches the job
    # twice (venv + system python) and both inherit the same redirect files,
    # which silently clobber each other's output
    LOGD = ROOT / "results" / "d45_run.log"
    import atexit
    _lf = open(LOGD, "a", buffering=1)
    atexit.register(_lf.close)
    os.dup2(_lf.fileno(), 1)
    os.dup2(_lf.fileno(), 2)
    print(f"[d45] === launch pid {os.getpid()} exe {sys.executable} ===", flush=True)
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        print("[d45] FATAL: see traceback above", flush=True)
        raise
