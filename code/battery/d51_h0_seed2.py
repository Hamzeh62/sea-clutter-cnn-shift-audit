"""D51 (T4b) — Second-seed replication of d49's headline cells.

The d49 H0-side headline (flat-calibrated threshold met by spike clutter)
rests on one pool draw (seed 141).  This run re-draws the battery's
headline cells with a fresh pool (seed 142, same recipe) for the two arms
that carry the story — the flat-calibrated CNN-C (the ~152x-nominal cell)
and its hybrid comparator (the ~16x-nominal graceful cell) — plus the
random-pool CONTROL rows, on a fresh 6,000-unit clutter-only pool drawn
with d49's exact recipe but pool seed 142:

  Pool P: UnitSampler(test, seed=142), rng(142).permutation order, 6,000 units
  nu-hat: phase-1 clutter_stats.json MoM (d46/d49 convention, keys = cube
          stems); terciles t1/t2 on P's nu distribution
  Calibration pools: C_flat (nu > t2), C_rand (first 2,000 permutation
          positions, nu-matched only)
  Test pool: T_spike (spike units NOT in C_rand — disjoint by construction)
  Thresholds: phase-4 M3 extrapolated rule (n*1e-3 <= 2 in every pool)
  Rates: exact Clopper-Pearson CIs; k/n reported per cell

Detectors: CNN-C (d40 checkpoint) and Proposed hybrid (d43 in-vivo path).
The paired design scores both detectors on IDENTICAL units; the CNN-C and
hybrid random-pool controls share the same C_rand units and the same test
units as their flat-calibrated arms, so the seed-2 replication tests the
ORDERING (CNN-C inflation >> hybrid inflation at the point estimates)
rather than any single cell's value.

Discipline: exclusive-create pid lock; OPENBLAS_NUM_THREADS=8 before numpy;
no OMP/torch thread overrides; chunked (250-unit) resume checkpoints.

Output: results/json/d51_h0_seed2.json (+ d51_progress.npz).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
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
from src.baselines.anmf import steering_bank
from src.eval.metrics import clopper_pearson

CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]

JS = ROOT / "results/json"
PROG = JS / "d51_progress.npz"
OUTJ = JS / "d51_h0_seed2.json"

N_POOL = 6000
CHUNK = 250
POOL_SEED = 142
DETS = ("CNN-C", "Proposed")


def log(m):
    print(f"[d51] {m}", flush=True)


def acquire_lock():
    lock = JS / "d51_run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("[d51] another instance holds the lock - exiting", flush=True)
        return None
    return lock


def load_nu_map():
    """(file_stem, cell) -> MoM nu — d46/d49 EXACT convention."""
    st = json.loads((JS / "clutter_stats.json").read_text())
    nmap = {}
    for fkey, fval in st.items():
        stem = fkey[:-4] if fkey.endswith(".npz") else fkey
        for ci, c in enumerate(fval["cells"]):
            v = (c.get("K") or {}).get("nu")
            if v is not None and 0.05 < v < 100:
                nmap[(stem, ci)] = float(v)
    return nmap


def extrapolate_threshold(arr, pfa):
    """phase-4 M3 exponential-tail rule (tail above the 90th percentile)."""
    t0 = float(np.quantile(arr, 0.90))
    exc = arr[arr > t0]
    lam = 1.0 / max(float(np.mean(exc - t0)), 1e-12)
    c = len(exc) / len(arr)
    return float(t0 + np.log(c / pfa) / lam)


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
    nu_map = load_nu_map()

    # ---------------- pool P (units + windows), seed 142 --------------------
    split = load_split(CFG["split"]["seed"])
    sampler = UnitSampler(split["test"], Np, HOP,
                          pol=CFG["data"]["primary_pol"], seed=POOL_SEED)
    n_total = len(sampler)
    order = np.random.default_rng(POOL_SEED).permutation(n_total)[:N_POOL]
    units = [sampler.units[i] for i in order]
    nu = np.array([nu_map.get((k, c), np.nan) for (k, c, _w) in units])
    n_match = int(np.sum(~np.isnan(nu)))
    log(f"pool P: {N_POOL} units, nu matched {n_match} ({n_match / N_POOL:.1%})")
    assert n_match >= int(0.55 * N_POOL), (
        f"nu lookup matched only {n_match}/{N_POOL} (expected ~62%; a much "
        "lower rate means a unit-key format mismatch vs clutter_stats.json)")

    # ---------------- detectors ---------------------------------------------
    det = HybridDetector(ROOT / "results/checkpoints/best_s0.pt")
    zsep = np.load(JS / "csep_prep.npz")
    assert int(zsep["Np"]) == Np, "csep prep Np mismatch"
    Chat = zsep["Chat"]
    P = steering_bank(int(zsep["Np"]))
    CinvP = np.linalg.solve(Chat, P)
    a = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))
    prep_sep = {"Chat": Chat, "CinvP": CinvP, "a": a}
    det._prep_sep["__default__"] = prep_sep

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

    # ---------------- chunked scoring with resume ----------------------------
    prog = {}
    if PROG.exists():
        prog = dict(np.load(PROG, allow_pickle=True))
        log(f"resuming: {len(prog)} keys present")

    def save_prog():
        tmp = JS / "d51_tmp.npz"
        np.savez(tmp, **prog)
        tmp.replace(PROG)

    S = {dn: np.empty(N_POOL) for dn in DETS}
    for dn in DETS:
        if f"scores|{dn}" in prog:
            S[dn] = prog[f"scores|{dn}"]

    for c0 in range(0, N_POOL, CHUNK):
        if f"done|{c0}" in prog:
            continue
        tc0 = time.time()
        sl = slice(c0, min(c0 + CHUNK, N_POOL))
        us = units[sl]
        xs = np.stack([
            np.stack([sampler.data[k][w0:w0 + Np, c].astype(np.complex128),
                      sampler.data_cross[k][w0:w0 + Np, c].astype(np.complex128)])
            for (k, c, w0) in us])
        S["CNN-C"][sl] = score_windows(netC, xs)
        outs = det.network_out_batch(xs, chunk=64)
        for k in range(len(us)):
            S["Proposed"][c0 + k], _ = det.statistic_from_net(
                xs[k], outs[k], prep_sep=prep_sep)
        prog[f"done|{c0}"] = np.array([1])
        for dn in DETS:
            prog[f"scores|{dn}"] = S[dn]
        save_prog()
        log(f"chunk {c0}: {c0 + len(us)}/{N_POOL} ({time.time() - tc0:.0f}s)")

    for dn in DETS:
        assert np.all(np.isfinite(S[dn])), f"{dn}: non-finite scores"

    # ---------------- terciles, pools, thresholds ----------------------------
    v = nu[~np.isnan(nu)]
    t1, t2 = np.percentile(v, [100 / 3, 200 / 3])
    ok = ~np.isnan(nu)
    m_spike = ok & (nu <= t1)
    m_flat = ok & (nu > t2)
    rand_idx = np.arange(2000)                      # first 2000 in perm order
    in_rand = np.zeros(N_POOL, bool); in_rand[rand_idx] = True

    calib = {"C_flat": m_flat.copy(), "C_rand": in_rand.copy()}
    calib["C_rand"][~ok] = False   # tercile-consistent: nu-matched units only
    test = {"T_spike": m_spike & ~in_rand}

    out = {"task": "T4b second-seed replication of d49's headline cells "
                   "(fresh pool, seed 142)",
           "config": {"pool_seed": POOL_SEED, "n_pool": N_POOL,
                      "chunk": CHUNK,
                      "pool_recipe": "UnitSampler(test, seed=142), "
                                     "rng(142).permutation order",
                      "nu_source": "phase-1 clutter_stats.json MoM (k_moments)",
                      "terciles": {"t1": float(t1), "t2": float(t2)},
                      "threshold_rule": "phase-4 M3 extrapolated (n*1e-3 <= 2 "
                                        "in every calib pool)",
                      "counts": {"spike": int(m_spike.sum()),
                                 "flat": int(m_flat.sum()),
                                 "rand_nu_matched": int(calib["C_rand"].sum())},
                      "detectors": DETS},
           "thresholds": {}, "realized_pfa": {}, "notes": []}

    for dn in DETS:
        out["thresholds"][dn] = {}
        for cname, cmask in calib.items():
            arr = S[dn][cmask]
            out["thresholds"][dn][cname] = {
                "n": int(cmask.sum()),
                "t_1e3": extrapolate_threshold(arr, 1e-3)}
    for dn in DETS:
        out["realized_pfa"][dn] = {}
        for tname, tmask in test.items():
            blk = {}
            for cname in calib:
                thr = out["thresholds"][dn][cname]["t_1e3"]
                hits = S[dn][tmask] > thr
                k, n = int(hits.sum()), int(tmask.sum())
                lo, hi = clopper_pearson(k, n, 0.05)
                blk[f"{cname}@0.001"] = {"k": k, "n": n, "rate": k / max(n, 1),
                                         "ci_lo": lo, "ci_hi": hi}
            out["realized_pfa"][dn][tname] = blk
        r3 = out["realized_pfa"][dn]["T_spike"]
        log(f"{dn}: C_flat->T_spike {r3['C_flat@0.001']['k']}/"
            f"{r3['C_flat@0.001']['n']} = {r3['C_flat@0.001']['rate']:.2e} | "
            f"C_rand control {r3['C_rand@0.001']['k']}/"
            f"{r3['C_rand@0.001']['n']}")

    # headline arithmetic on the fresh pool
    c = out["realized_pfa"]["CNN-C"]["T_spike"]
    p = out["realized_pfa"]["Proposed"]["T_spike"]
    out["derived"] = {
        "cnnC_xnom": round(c["C_flat@0.001"]["rate"] / 0.001, 0),
        "hyb_xnom": round(p["C_flat@0.001"]["rate"] / 0.001, 0),
        "cnnC_control_rate": c["C_rand@0.001"]["rate"],
        "hyb_control_rate": p["C_rand@0.001"]["rate"],
        "ordering_holds": bool(c["C_flat@0.001"]["rate"]
                               > 5 * p["C_flat@0.001"]["rate"]),
        "controls_at_or_below_nominal": bool(
            c["C_rand@0.001"]["rate"] <= 0.005
            and p["C_rand@0.001"]["rate"] <= 0.005)}
    out["notes"] = [
        "Replicates d49's headline cells with a fresh pool (seed 142): "
        "flat-calibrated -> spike tercile and random-pool controls, CNN-C "
        "and hybrid only (the two arms that carry the story).",
        "Same recipe, tercile definition, threshold rule, and disjointness "
        "construction as d49 (d49_h0_shift.py); pool draw and threshold "
        "pools are the only things that change.",
        "The replication tests the ORDERING (CNN-C inflation >> hybrid "
        "inflation, controls at/nominal), not any single cell's value.",
    ]
    OUTJ.write_text(json.dumps(out, indent=1))
    log(f"wrote {OUTJ.name} ({time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
