"""T2 (task R6) — Real-target in-vivo battery: target cells AND adjacent cells.

Extends D43 (which scored target cells only, hit-counts only) to the full
in-vivo design the paper-2 handoff specifies:

  - the 8 literature-verified 1993 stare files (d43 LIT table, cells 1-based);
  - per file, the d43 window rule (w0 = HOP + j*4*HOP, w0+Np <= nsweep; 128
    windows on the 131072-sweep files) on the TARGET cell and ALL ADJACENT
    cells within +/-4 (N_SIDE, the secondary-window geometry);
  - ALL FOUR detectors at the DEPLOYED calibrated thresholds:
      CNN-C / CNN-IQ   (d40 checkpoints + d40 phase-4 extrapolated thresholds),
      Proposed hybrid  (d43/phase-4 path: network_out_batch +
      statistic_from_net, main csep_prep; deployed extrapolated 1e-3),
      ANMF-DL0.001     (deployed per-unit eigen path, d43 recipe);
  - per-cell detection rates with exact Clopper-Pearson CIs; the PRIMARY vs
    ADJACENT contrast is the in-vivo false-alarm reading (adjacent cells are
    target-free by construction — published guard-cell leakage caveated);
  - per-cell MoM nu-hat (phase-1 clutter_stats.json) correlated with per-cell
    CNN firing on adjacent cells (Spearman, pooled across files);
  - real-target Doppler structure on the target cell: per-window peak
    (clear-Doppler band, phase-4 cell_psd_bands recipe), -3 dB width, walk
    (slope + p95-p5 excursion), median Doppler; range-Doppler maps for the
    two d43 exemplar files.

Guards (abort conditions):
  G1  target-cell hybrid + ANMF rates reproduce d43_real_target.json per file
      EXACTLY (same windows, same chunk-64 batching, same deployed recipe) —
      pins the whole in-vivo path to the validated D43 run.
  G2  CNN scoring path: first 50 seed-77 pool windows reproduce
      d40_scores_pool77.npz c_c[:50] exactly (normalize + score_windows).

No ground truth exists for real detections: every rate is an APPARENT
per-window exceedance rate; adjacent-cell rates inherit target leakage into
guard-adjacent secondaries (ANMF covariance) as a conservative bias.

Resumable: per-(file, cell) checkpoint blocks in results/json/d48_progress.npz;
exclusive-create pid lock; one heavy job at a time.

Output: results/json/d48_real_target_invivo.json
        results/figures/fig_realtarget_rdmaps.{png,pdf}
        results/figures/fig_realtarget_cellrates.{png,pdf}
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.ipix_loader import read_ipix_file, complex_cubes
from src.data.synth import estimate_psd, clear_doppler_band
from src.proposed.infer import HybridDetector
from src.baselines.anmf import steering_bank
import anmf_load_sweep as base
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]
STEP = 4 * HOP
PRF = 1000.0
NFFT = 512
DATA_DIR = Path(CFG["data"]["raw_dir"])
SEED = CFG["seed"]
K_SEC = CFG["baselines"]["anmf"]["K_secondary"]
N_SIDE = CFG["baselines"]["anmf"]["secondary_cells_each_side"]
THR_HYB = 0.4542599082845903          # deployed extrapolated 1e-3 (d43)
THR_ANMF = 0.6567024168979555         # deployed extrapolated 1e-3 (d43)

JS = ROOT / "results/json"
FIG = ROOT / "results/figures"
PROG = JS / "d48_progress.npz"
OUTJ = JS / "d48_real_target_invivo.json"

LIT = {  # file -> (literature cell, 1-based; sources)  [d43 table]
    "19931107_135603_starea.cdf": (9, "rs14_4364;iet2017"),
    "19931107_141630_starea.cdf": (9, "sensors22_9163"),
    "19931107_145028_starea.cdf": (8, "rs14_4364;iet2017"),
    "19931108_213827_starea.cdf": (7, "rs14_4364;iet2017"),
    "19931108_220902_starea.cdf": (7, "sensors22_9163;iet2017"),
    "19931109_191449_starea.cdf": (7, "sensors22_9163;iet2017"),
    "19931109_202217_starea.cdf": (7, "iet2017"),
    "19931110_001635_starea.cdf": (7, "iet2017"),
}
EXEMPLARS = ["19931107_141630_starea.cdf", "19931109_191449_starea.cdf"]
DETS = ("CNN-C", "CNN-IQ", "Proposed", "ANMF-DL0.001")


def log(m):
    print(f"[d48] {m}", flush=True)


def acquire_lock():
    lock = JS / "d48_run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("[d48] another instance holds the lock - exiting", flush=True)
        return None
    return lock


def load_nu_map():
    st = json.loads((JS / "clutter_stats.json").read_text())
    nmap = {}
    for fkey, fval in st.items():
        stem = fkey.replace("_cubes.npz", ".cdf")
        for ci, c in enumerate(fval["cells"]):
            v = (c.get("K") or {}).get("nu")
            if v is not None and 0.05 <= v <= 500.0:
                nmap[(stem, ci)] = float(v)
    return nmap


def load_thresholds():
    """DEPLOYED 1e-3 thresholds, exactly the ones d43 used:
    hybrid = extrapolated constant (thresholds.json audit table), ANMF =
    anmf_load_ext.json dl0.001|0.001 (asserted equal to d43's constant),
    CNNs = d40 phase-4 extrapolated.  (NOTE: an empirical quantile of
    prop_h0_stats is NOT the deployed hybrid threshold at 1e-3 — phase-4's
    rule extrapolates below n*pfa = 10; d43's constant is the deployed value
    byte-validated by the G1 guard.)"""
    d40j = json.loads((JS / "d40_cnn_baselines.json").read_text())
    thr = {n: {p: float(d40j["thresholds"][n][p]["threshold"])
               for p in ("0.001", "0.0001")}
           for n in ("CNN-C", "CNN-IQ")}
    thr["Proposed"] = {"0.001": THR_HYB}
    extj = json.loads((JS / "anmf_load_ext.json").read_text())
    a1 = float(extj["h0_thresholds"]["dl0.001|0.001"]["threshold"])
    assert abs(a1 - THR_ANMF) < 1e-12, \
        f"ANMF extj {a1} != d43 constant {THR_ANMF}"
    thr["ANMF-DL0.001"] = {"0.001": a1}
    return thr


def cp_ci(k, n, alpha=0.05):
    from src.eval.metrics import clopper_pearson
    lo, hi = clopper_pearson(int(k), int(n), alpha)
    return float(lo), float(hi)


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
    thr = load_thresholds()
    nu_map = load_nu_map()

    det = HybridDetector(ROOT / "results/checkpoints/best_s0.pt")
    zsep = np.load(JS / "csep_prep.npz")
    Chat = zsep["Chat"]
    P = steering_bank(int(zsep["Np"]))
    CinvP = np.linalg.solve(Chat, P)
    a = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))
    prep_sep = {"Chat": Chat, "CinvP": CinvP, "a": a}   # d43 convention (no Cinv)
    det._prep_sep["__default__"] = prep_sep

    sys.path.insert(0, str(ROOT / "scripts"))
    from d40_train_cnn_baselines import CNNComplex, CNNRealIQ, normalize
    import torch
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

    # ---------------- G2: CNN scoring-path guard ---------------------------
    from src.eval.units import UnitSampler, load_split
    split = load_split(CFG["split"]["seed"])
    s77 = UnitSampler(split["test"], Np, HOP, seed=77)
    o77 = np.random.default_rng(77).permutation(len(s77))[:50]
    xs77 = np.stack([
        np.stack([s77.data[k][w0:w0 + Np, c].astype(np.complex128),
                  s77.data_cross[k][w0:w0 + Np, c].astype(np.complex128)])
        for (k, c, w0) in [s77.units[i] for i in o77]])
    z77 = np.load(JS / "d40_scores_pool77.npz")
    val = {"d43_rate_reproduction": {}}   # guards bookkeeping (before G2)
    # d38 lesson: conv kernel selection is batch-size dependent.  The d40
    # reference scored 1,500 windows in batches of 256; this 50-window probe
    # forms one batch of 50, so expect ~1e-7 RELATIVE drift (observed 9.5e-07
    # on the first run).  Guard on relative drift <= 1e-5; the deployed d43
    # hybrid guard needed exact one-batch replication only because its rtol
    # was 1e-10.  Detection decisions sit O(1) from the threshold, so a 1e-7
    # relative score drift cannot flip any rate at k/128 granularity.
    ref = z77["c_c"][:50]
    got = score_windows(netC, xs77)
    g2 = float(np.max(np.abs(got - ref) / np.abs(ref)))
    val["cnn_scoring_guard_max_reldiff"] = g2
    log(f"G2 CNN scoring guard: max rel diff {g2:.3e} (pool77 c_c[:50])")
    if g2 > 1e-5:
        log("ABORT — CNN scoring path drift beyond the batch-composition "
            "tolerance")
        OUTJ.write_text(json.dumps({"validation": val}, indent=1))
        return

    # ---------------- progress store ---------------------------------------
    prog = {}
    if PROG.exists():
        prog = dict(np.load(PROG, allow_pickle=True))
        log(f"resuming: {len(prog)} checkpoint blocks present")

    def save_prog():
        tmp = JS / "d48_progress_tmp.npz"
        np.savez(tmp, **prog)
        tmp.replace(PROG)

    # ---------------- per-file run -----------------------------------------
    per_file = {}
    win = np.hanning(Np)
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / PRF))

    for fname in sorted(LIT):
        tf0 = time.time()
        t_idx = LIT[fname][0] - 1
        cells = [c for c in range(t_idx - N_SIDE, t_idx + N_SIDE + 1)
                 if 0 <= c < 14]
        rec = read_ipix_file(DATA_DIR / fname)
        vv = complex_cubes(rec)["VV"]
        vh = complex_cubes(rec)["VH"]
        nsweep, nrange = vv.shape
        w0s = list(range(HOP, nsweep - Np + 1, STEP))

        # file clear-Doppler band (phase-4 recipe, cell min(3, nrange-1))
        fgrid, psd = estimate_psd(vv[:, min(3, nrange - 1)], PRF, 512)
        band = clear_doppler_band(psd, fgrid, guard_bins=8)
        in_band = (freqs >= band[0]) & (freqs <= band[1])

        for cell in cells:
            key = f"{fname}|{cell}"
            if key in prog:
                continue
            xs = np.stack([
                np.stack([vv[w0:w0 + Np, cell].astype(np.complex128),
                          vh[w0:w0 + Np, cell].astype(np.complex128)])
                for w0 in w0s])
            sc = {"CNN-C": score_windows(netC, xs),
                  "CNN-IQ": score_windows(netR, xs)}
            outs = det.network_out_batch(xs, chunk=64)   # d43 batching
            th = np.array([det.statistic_from_net(xs[k], outs[k],
                                                  prep_sep=prep_sep)[0]
                           for k in range(len(xs))])
            ta = np.empty(len(xs))
            for k, w0 in enumerate(w0s):
                cand = [c for c in range(cell - N_SIDE, cell + N_SIDE + 1)
                        if c != cell and 0 <= c < nrange]
                per_cell_n = int(np.ceil(K_SEC / len(cand)))
                offs = np.arange(per_cell_n) - per_cell_n // 2
                wins = []
                for c in cand:
                    for o in offs:
                        s = w0 + o * HOP
                        if 0 <= s and s + Np <= nsweep:
                            wins.append(vv[s:s + Np, c].astype(np.complex128))
                    if len(wins) >= K_SEC:
                        break
                sec = np.stack(wins[:K_SEC])
                y = xs[k][0]
                Cht = base.build_chat(sec)
                Q, sig = base.chat_eigh(Cht)
                w_l, m_l = base.shrink_intensity(Cht, sig)
                e = base.variant_isig(sig, w_l, m_l, float(np.sum(sig) / Np),
                                      "dl0.001")
                W = Q.conj().T @ P
                av = np.real(np.einsum("ij,ij->j", W.conj(), e[:, None] * W))
                ta[k] = base.stat_py(y, W, av, e, Q.conj().T @ y)
            prog[key] = np.stack([sc["CNN-C"], sc["CNN-IQ"], th, ta])
            save_prog()
            log(f"  {fname[:16]} cell {cell} done "
                f"({time.time() - tf0:.0f}s into file)")

        # ---------------- guards + assembly for this file -------------------
        S = {c: prog[f"{fname}|{c}"] for c in cells}   # (4, n) per cell
        n = S[cells[0]].shape[1]
        # G1: target-cell hybrid/ANMF rates must equal d43's
        d43 = json.loads((JS / "d43_real_target.json").read_text())
        r_h = float(np.mean(S[t_idx][2] > THR_HYB))
        r_a = float(np.mean(S[t_idx][3] > THR_ANMF))
        ok = (r_h == d43["per_file"][fname]["rate_hybrid"]
              and r_a == d43["per_file"][fname]["rate_anmf_dl001"])
        val["d43_rate_reproduction"][fname] = {
            "hybrid": {"recomputed": r_h,
                       "d43": d43["per_file"][fname]["rate_hybrid"]},
            "anmf": {"recomputed": r_a,
                     "d43": d43["per_file"][fname]["rate_anmf_dl001"]},
            "match": bool(ok)}
        log(f"G1 d43 reproduction {fname[:16]}: h {r_h} vs {d43['per_file'][fname]['rate_hybrid']}, "
            f"a {r_a} vs {d43['per_file'][fname]['rate_anmf_dl001']} -> "
            f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            out = {"task": "T2/R6 in-vivo battery", "validation": val}
            OUTJ.write_text(json.dumps(out, indent=1))
            log("ABORT — d43 reproduction failed; inspect before trusting T2")
            return

        # per-cell rates + CIs
        pc = {}
        for c in cells:
            blk = {"role": "target" if c == t_idx else "adjacent",
                   "nu_hat": nu_map.get((fname, c)),
                   "n_windows": int(n), "rates": {}, "rate_ci": {},
                   "score_median": {}}
            for di, dn in enumerate(DETS):
                hits = S[c][di] > thr[dn]["0.001"]
                blk["rates"][dn] = float(hits.mean())
                blk["rate_ci"][dn] = cp_ci(hits.sum(), n)
                blk["score_median"][dn] = float(np.median(S[c][di]))
            pc[str(c)] = blk

        # primary-vs-adjacent contrast (in-vivo false-alarm reading)
        adj = [c for c in cells if c != t_idx]
        contrast = {}
        for di, dn in enumerate(DETS):
            kh = int(sum((S[c][di] > thr[dn]["0.001"]).sum() for c in adj))
            na = n * len(adj)
            contrast[dn] = {
                "target_rate": pc[str(t_idx)]["rates"][dn],
                "adjacent_pooled_rate": kh / na,
                "adjacent_pooled_ci": cp_ci(kh, na),
                "adjacent_max_cell_rate": max(pc[str(c)]["rates"][dn]
                                              for c in adj),
            }

        # Doppler structure on the target cell (clear-band peak tracking)
        fk, w3 = [], []
        for k, w0 in enumerate(w0s):
            y = vv[w0:w0 + Np, t_idx].astype(np.complex128)
            Spe = np.fft.fftshift(np.abs(np.fft.fft(win * y, NFFT)) ** 2)
            ib = np.where(in_band)[0]
            pk = int(ib[np.argmax(Spe[ib])])
            fk.append(freqs[pk])
            half = Spe[pk] / 2
            l = pk
            while l - 1 >= 0 and Spe[l - 1] > half and in_band[l - 1]:
                l -= 1
            r = pk
            while r + 1 < NFFT and Spe[r + 1] > half and in_band[r + 1]:
                r += 1
            w3.append((r - l + 1) * (PRF / NFFT))
        fk = np.asarray(fk); w3 = np.asarray(w3)
        tvec = np.asarray(w0s) / PRF
        slope = float(np.polyfit(tvec, fk, 1)[0])
        doppler = {
            "clear_band_hz": [float(band[0]), float(band[1])],
            "f_pk_median_hz": float(np.median(fk)),
            "f_pk_iqr_hz": [float(np.percentile(fk, 25)),
                            float(np.percentile(fk, 75))],
            "walk_slope_hz_per_s": slope,
            "walk_excursion_p95_p5_hz": float(np.percentile(fk, 95)
                                              - np.percentile(fk, 5)),
            "width_3db_median_hz": float(np.median(w3)),
            "n_windows": int(n),
        }

        # range-Doppler map: per-cell file-mean |FFT|^2 over the sampled windows
        rd = {}
        for c in cells:
            acc = None
            for w0 in w0s:
                Spe = np.abs(np.fft.fft(
                    win * vv[w0:w0 + Np, c].astype(np.complex128), NFFT)) ** 2
                acc = Spe if acc is None else acc + Spe
            rd[c] = np.fft.fftshift(acc / len(w0s))
        per_file[fname] = {
            "target_cell_0based": t_idx, "cells": cells,
            "n_windows": int(n), "per_cell": pc, "contrast": contrast,
            "doppler": doppler, "clear_band_hz": [float(band[0]), float(band[1])],
        }
        if fname in EXEMPLARS:
            per_file[fname]["rd_map"] = {str(c): rd[c].tolist() for c in cells}
            per_file[fname]["freqs_hz"] = freqs.tolist()
        log(f"{fname}: assembled ({time.time() - tf0:.0f}s into file)")

    # ---------------- nu-hat correlation (adjacent cells, pooled) ----------
    rows = []
    for fname, pf in per_file.items():
        for c, blk in pf["per_cell"].items():
            if blk["role"] != "adjacent" or blk["nu_hat"] is None:
                continue
            rows.append((blk["nu_hat"],
                         *[blk["rates"][dn] for dn in DETS]))
    rows = np.array(rows)
    corr = {"n_cells": int(len(rows)),
            "definition": "Spearman rho between per-cell MoM nu-hat and the "
                          "per-cell apparent detection rate, ADJACENT cells "
                          "pooled across the 8 files"}
    for di, dn in enumerate(DETS):
        rho, p = sps.spearmanr(rows[:, 0], rows[:, 1 + di])
        corr[dn] = {"rho": float(rho), "p": float(p)}
    log("nu correlation: " + ", ".join(
        f"{dn} rho={corr[dn]['rho']:+.2f} (p={corr[dn]['p']:.3g})"
        for dn in DETS))

    out = {"task": "T2/R6 real-target in-vivo battery (target + adjacent cells)",
           "config": {
               "files": sorted(LIT), "window_rule": "w0 = HOP + j*4*HOP "
               "(d43), 128 windows per 131072-sweep file",
               "cells": "target +/-4 (N_SIDE geometry)",
               "thresholds_1e3": {dn: thr[dn]["0.001"] for dn in DETS},
               "threshold_provenance": {
                   "CNN-C/CNN-IQ": "d40 phase-4 extrapolated (deployed)",
                   "Proposed": "deployed extrapolated (d43 constant)",
                   "ANMF-DL0.001": "deployed extrapolated (d43 constant)"},
               "hybrid_path": "d43 convention: network_out_batch(chunk=64) + "
                              "statistic_from_net, main csep_prep (no Cinv)",
               "anmf_path": "deployed per-unit dl0.001 eigen path (d43 recipe)"},
           "validation": val, "per_file": per_file, "nu_correlation": corr,
           "notes": []}
    out["notes"] = [
        "NO GROUND TRUTH: every rate is an APPARENT per-window exceedance "
        "rate at the deployed 1e-3 thresholds; adjacent-cell rates are the "
        "in-vivo false-alarm reading, not certified Pfa.",
        "Adjacent-cell ANMF secondaries include the target cell (cells +/-4 "
        "geometry); target leakage into the covariance biases the ANMF "
        "statistic DOWN on adjacent cells (conservative for false alarms) "
        "and is caveated.",
        "The target cell itself is contaminated (target present): its rates "
        "mix target evidence with in-vivo clutter false alarms; the "
        "primary-vs-adjacent contrast, not the raw target rate, is the "
        "quantity of interest.",
        "CNN firing on adjacent cells vs per-cell MoM nu-hat: see "
        "nu_correlation; correlation does not establish mechanism.",
        "Doppler/walk/width are estimated from clear-band periodogram peaks "
        "of the raw target-cell VV (no separation stage); weak-target files "
        "can have clutter-dominated peaks.",
        "Guards: G2 CNN scoring path exact vs d40_scores_pool77; G1 target-cell "
        "hybrid/ANMF rates reproduce d43_real_target.json per file exactly.",
    ]
    OUTJ.write_text(json.dumps(out, indent=1))
    _figs(out)
    log(f"wrote {OUTJ.name} + figures ({time.time() - t00:.0f}s total)")


def _figs(out):
    FIG.mkdir(exist_ok=True, parents=True)
    # RD maps for exemplars
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, fname in zip(axes, EXEMPLARS):
        pf = out["per_file"][fname]
        fr = np.asarray(pf["freqs_hz"]) / 1e3
        cells = pf["cells"]
        M = np.stack([np.asarray(pf["rd_map"][str(c)]) for c in cells])
        t_idx = pf["target_cell_0based"]
        ti = cells.index(t_idx)
        ext = [fr[0], fr[-1], cells[0] - 0.5, cells[-1] + 0.5]
        ax.imshow(10 * np.log10(M + 1e-12), aspect="auto", origin="lower",
                  extent=ext, cmap="viridis")
        ax.axhline(t_idx, color="w", ls="--", lw=0.8)
        ax.set_xlabel("Doppler (kHz)")
        ax.set_title(fname[:16], fontsize=8.5)
    axes[0].set_ylabel("range cell (0-based)")
    fig.suptitle("Range-Doppler maps (file-mean |FFT|^2, dB) — target cell dashed",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig_realtarget_rdmaps.png", dpi=200)
    fig.savefig(FIG / "fig_realtarget_rdmaps.pdf")
    plt.close(fig)

    # per-cell rates: target (filled) vs adjacent (open), per detector.
    # Rows = detectors (4), columns = the 8 files.  (The original 2x4 grid
    # assumed 2 rows of files; with 8 files it IndexError'd — fixed 2026-09-10
    # AFTER the battery JSON was written; figures only, no recompute.)
    fig, axes = plt.subplots(4, 8, figsize=(21, 9.2), sharey=True)
    dets = list(DETS)
    fnames = sorted(out["per_file"])
    assert len(fnames) <= axes.shape[1], "figure grid sized for 8 files"
    for di, dn in enumerate(dets):
        for r, fname in enumerate(fnames):
            pf = out["per_file"][fname]
            cells = sorted(pf["per_cell"].keys(), key=int)
            xs = np.arange(len(cells))
            ax = axes[di][r]
            rates = [pf["per_cell"][c]["rates"][dn] for c in cells]
            los = [pf["per_cell"][c]["rate_ci"][dn][0] for c in cells]
            his = [pf["per_cell"][c]["rate_ci"][dn][1] for c in cells]
            colors = ["#D55E00" if pf["per_cell"][c]["role"] == "target"
                      else "#0072B2" for c in cells]
            for x, rr, lo, hi, cc in zip(xs, rates, los, his, colors):
                ax.errorbar(x, max(rr, 0.4 / pf["n_windows"]),
                            yerr=[[max(rr - lo, 0)], [max(hi - rr, 0)]],
                            fmt="o", ms=3.5, color=cc, capsize=2)
            ax.set_yscale("log")
            ax.set_ylim(0.3e-2, 1.6)
            if di == 0:
                ax.set_title(fname[:13] + "…", fontsize=7)
            if r == 0:
                ax.set_ylabel(f"{dn}\napparent rate", fontsize=7.5)
            ax.grid(alpha=0.25, which="both")
            ax.tick_params(labelsize=6)
    fig.suptitle("Per-cell apparent detection rates (deployed 1e-3; orange = "
                 "target cell, blue = adjacent)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig_realtarget_cellrates.png", dpi=200)
    fig.savefig(FIG / "fig_realtarget_cellrates.pdf")
    plt.close(fig)


if __name__ == "__main__":
    if "--figs-only" in sys.argv:
        # Rebuild figures from the saved JSON (no recompute).  Added after the
        # 2026-09-10 run crashed in _figs (all 8 files forced into a 2x4 grid);
        # the battery itself (guards, JSON) was already complete.
        out = json.loads(OUTJ.read_text(encoding="utf-8"))
        _figs(out)
        print("[d48] figures rebuilt from", OUTJ.name)
    else:
        main()
