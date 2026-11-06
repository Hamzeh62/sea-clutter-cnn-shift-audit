"""T3 — H0-side shift battery: thresholds vs the clutter they meet.

The d45 battery shifts the H1 side (target models).  This battery shifts the
H0 side: every detector is re-thresholded on a RESTRICTED clutter pool and
evaluated on the clutter pools the restriction excludes.  Question (reviewers):
does an injection-benchmark calibration transfer across the sea-state axis of
the very same clutter domain, or does the certified Pfa silently depend on
which clutter the calibration pool happened to contain?

Design (one fresh pool, all four detectors on IDENTICAL units — paired):
  Pool P: 6,000 clutter-only units, UnitSampler(split['test'], seed=141),
          rng(141).permutation order (the phase-2/trial pool family recipe;
          seed 141 does not collide with any deployed stream).
  nu-hat: per-unit MoM nu from phase-1 clutter_stats.json; terciles t1/t2 on
          P's nu distribution (same definition as d46).
  Calibration pools (thresholds FROM):
    C_spike  : P units with nu <= t1          (~2,000)
    C_flat   : P units with nu >  t2          (~2,000)
    C_rand   : first 2,000 units in permutation order (standard-recipe pool)
  Test pools (rates ON):
    T_all    : all 6,000 units
    T_spike  : spike units  NOT in C_rand    (independent of C_rand)
    T_mid    : mid units NOT in C_rand
    T_flat   : flat units  NOT in C_rand
  Thresholds: phase-4 extrapolated rule (M3 exponential tail; n*1e-3 <= 2 << 10
  for every calibration pool, so the rule is unambiguous) at design
  Pfa in {1e-3, 1e-4}; per-detector deployed statistic recipes.
  Rates: exact Clopper-Pearson CIs; every cell reports k/n.

Expected readings (reported either way they fall): a flat-calibrated threshold
met by spiky clutter should over-fire (H0 shift inflates Pfa); a
spike-calibrated threshold met by flat clutter should under-fire (wasted
operating point).  The random pool is the control approximating the benchmark's
matched design.

Detectors (deployed recipes, identical units):
  CNN-C / CNN-IQ : d40 checkpoints + normalize + score_windows
  Proposed       : HybridDetector best_s0.pt, d43 in-vivo path
                   (network_out_batch(chunk=64) + statistic_from_net, main
                   csep_prep, no Cinv) — NOT the hoisted path
  ANMF-DL0.001   : deployed per-unit dl0.001 eigen path (d43 recipe)

Discipline: exclusive-create pid lock; resumable per-chunk progress npz
(500-unit chunks: CNN+hybrid+ANMF per chunk ~7 min); OPENBLAS_NUM_THREADS=8
before numpy; no OMP/torch thread overrides.

Output: results/json/d49_h0_shift.json
        results/figures/fig_h0shift.{png,pdf}
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.eval.units import UnitSampler, load_split
from src.proposed.infer import HybridDetector
from src.baselines.anmf import steering_bank
import anmf_load_sweep as base

CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]
SEED = CFG["seed"]
K_SEC = CFG["baselines"]["anmf"]["K_secondary"]
N_SIDE = CFG["baselines"]["anmf"]["secondary_cells_each_side"]
N_POOL = 6000
CHUNK = 500
POOL_SEED = 141

JS = ROOT / "results/json"
FIG = ROOT / "results/figures"
PROG = JS / "d49_progress.npz"
OUTJ = JS / "d49_h0_shift.json"
DETS = ("CNN-C", "CNN-IQ", "Proposed", "ANMF-DL0.001")


def log(m):
    print(f"[d49] {m}", flush=True)


def acquire_lock():
    lock = JS / "d49_run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("[d49] another instance holds the lock - exiting", flush=True)
        return None
    return lock


def load_nu_map():
    """(file_stem, cell) -> MoM nu — d46's EXACT convention (R8c terciles):
    keys are the _cubes.npz stems with '.npz' stripped (UnitSampler keys are
    Path(f).stem of the _cubes path, i.e. '..._cubes' — NOT '.cdf'); strict
    0.05 < nu < 100 filter excludes clipped cells identically to d46."""
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

    # ---------------- pool P (units + windows) ------------------------------
    split = load_split(CFG["split"]["seed"])
    sampler = UnitSampler(split["test"], Np, HOP,
                          pol=CFG["data"]["primary_pol"], seed=POOL_SEED)
    n_total = len(sampler)
    order = np.random.default_rng(POOL_SEED).permutation(n_total)[:N_POOL]
    units = [sampler.units[i] for i in order]
    nu = np.array([nu_map.get((k, c), np.nan) for (k, c, _w) in units])
    n_match = int(np.sum(~np.isnan(nu)))
    log(f"pool P: {N_POOL} units, nu matched {n_match} ({n_match / N_POOL:.1%})")
    # The strict 0.05<nu<100 validity filter keeps ~62% of test-split cells
    # (d46: 13,286/21,462 = 61.9%; verified per-file for this split), so the
    # correct zero-hypothesis here is ~62%, NOT 100%.  Guard only against
    # key-format breakage (which matched 0 before the d46-convention fix).
    assert n_match >= int(0.55 * N_POOL), (
        f"nu lookup matched only {n_match}/{N_POOL} (expected ~62%; a much "
        "lower rate means a unit-key format mismatch vs clutter_stats.json)")

    # ---------------- detectors ---------------------------------------------
    det = HybridDetector(ROOT / "results/checkpoints/best_s0.pt")
    zsep = np.load(JS / "csep_prep.npz")
    Chat = zsep["Chat"]
    P = steering_bank(int(zsep["Np"]))
    CinvP = np.linalg.solve(Chat, P)
    a = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))
    prep_sep = {"Chat": Chat, "CinvP": CinvP, "a": a}
    det._prep_sep["__default__"] = prep_sep

    import torch
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
        log(f"resuming: {len(prog)} chunks present")

    def save_prog():
        tmp = JS / "d49_progress_tmp.npz"
        np.savez(tmp, **prog)
        tmp.replace(PROG)

    S = {dn: np.empty(N_POOL) for dn in DETS}
    for di, dn in enumerate(DETS):
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
        S["CNN-IQ"][sl] = score_windows(netR, xs)
        outs = det.network_out_batch(xs, chunk=64)
        for k in range(len(us)):
            S["Proposed"][c0 + k], _ = det.statistic_from_net(
                xs[k], outs[k], prep_sep=prep_sep)
        for k, (key, cell, w0) in enumerate(us):
            cand = [c for c in range(cell - N_SIDE, cell + N_SIDE + 1)
                    if c != cell and 0 <= c < len(sampler.data[key][0])]
            per_cell_n = int(np.ceil(K_SEC / len(cand)))
            offs = np.arange(per_cell_n) - per_cell_n // 2
            wins = []
            for c in cand:
                for o in offs:
                    s = w0 + o * HOP
                    if 0 <= s and s + Np <= sampler.data[key].shape[0]:
                        wins.append(sampler.data[key][s:s + Np, c]
                                    .astype(np.complex128))
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
            S["ANMF-DL0.001"][c0 + k] = base.stat_py(y, W, av, e,
                                                     Q.conj().T @ y)
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
    m_mid = ok & (nu > t1) & (nu <= t2)
    m_flat = ok & (nu > t2)
    rand_idx = np.arange(2000)                      # first 2000 in perm order
    in_rand = np.zeros(N_POOL, bool); in_rand[rand_idx] = True

    calib = {"C_spike": m_spike.copy(), "C_flat": m_flat.copy(),
             "C_rand": in_rand.copy()}
    calib["C_rand"][~ok] = False   # tercile-consistent: nu-matched units only
    test = {"T_all": ok.copy(),
            "T_spike": m_spike & ~in_rand,
            "T_mid": m_mid & ~in_rand,
            "T_flat": m_flat & ~in_rand}

    out = {"task": "T3 H0-side shift battery (re-threshold on restricted "
                   "clutter pools, evaluate on the excluded clutter)",
           "config": {"pool_seed": POOL_SEED, "n_pool": N_POOL,
                      "chunk": CHUNK,
                      "pool_recipe": "UnitSampler(test, seed=141), "
                                     "rng(141).permutation order",
                      "nu_source": "phase-1 clutter_stats.json MoM (k_moments)",
                      "terciles": {"t1": float(t1), "t2": float(t2)},
                      "threshold_rule": "phase-4 M3 extrapolated (n*1e-3 <= 2 "
                                        "in every calib pool)",
                      "counts": {"spike": int(m_spike.sum()),
                                 "mid": int(m_mid.sum()),
                                 "flat": int(m_flat.sum()),
                                 "rand_nu_matched": int(calib["C_rand"].sum())},
                      "detectors": DETS},
           "thresholds": {}, "realized_pfa": {}, "notes": []}

    from src.eval.metrics import clopper_pearson
    for dn in DETS:
        out["thresholds"][dn] = {}
        for cname, cmask in calib.items():
            arr = S[dn][cmask]
            out["thresholds"][dn][cname] = {
                "n": int(cmask.sum()),
                "t_1e3": extrapolate_threshold(arr, 1e-3),
                "t_1e4": extrapolate_threshold(arr, 1e-4)}

    for dn in DETS:
        out["realized_pfa"][dn] = {}
        for tname, tmask in test.items():
            blk = {}
            for cname in calib:
                thr = out["thresholds"][dn][cname]
                for p, tk in (("0.001", "t_1e3"), ("0.0001", "t_1e4")):
                    hits = S[dn][tmask] > thr[tk]
                    k, n = int(hits.sum()), int(tmask.sum())
                    lo, hi = clopper_pearson(k, n, 0.05)
                    blk[f"{cname}@{p}"] = {
                        "k": k, "n": n, "rate": k / max(n, 1),
                        "ci_lo": lo, "ci_hi": hi}
            out["realized_pfa"][dn][tname] = blk
        r3 = out["realized_pfa"][dn]
        log(f"{dn}: C_flat->T_spike @1e-3 "
            f"{r3['T_spike']['C_flat@0.001']['k']}/"
            f"{r3['T_spike']['C_flat@0.001']['n']}"
            f" = {r3['T_spike']['C_flat@0.001']['rate']:.2e} | "
            f"C_rand->T_all @1e-3 {r3['T_all']['C_rand@0.001']['k']}/"
            f"{r3['T_all']['C_rand@0.001']['n']}"
            f" = {r3['T_all']['C_rand@0.001']['rate']:.2e}")

    out["notes"] = [
        "Paired design: all four detectors score the SAME 6,000 units; "
        "calibration and test pools are disjoint by construction (tercile "
        "pools exclude each other; C_rand/T_* exclude the first 2,000 "
        "permutation positions).",
        "Thresholds are extrapolated on every calibration pool (the deployed "
        "phase-4 rule); the flat/spike asymmetry is the H0-side shift under "
        "test, the random pool is the matched-design control.",
        "Nu-hat unmatched units are excluded from ALL pools (counted in "
        "config.counts); tercile definition matches d46 (spike = lowest nu).",
        "No ground-truth claims: realized Pfa on defined test pools with "
        "exact Clopper-Pearson CIs is the entire output.",
    ]
    OUTJ.write_text(json.dumps(out, indent=1))
    _figs(out)
    log(f"wrote {OUTJ.name} ({time.time() - t00:.0f}s)")


def _figs(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    dets = list(DETS)
    cols = {"C_spike": "#0072B2", "C_flat": "#D55E00", "C_rand": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, p in zip(axes, ("0.001", "0.0001")):
        for i, dn in enumerate(dets):
            for j, tname in enumerate(("T_spike", "T_mid", "T_flat", "T_all")):
                for k, cname in enumerate(("C_spike", "C_flat", "C_rand")):
                    r = out["realized_pfa"][dn][tname][f"{cname}@{p}"]
                    x = j + (i - 1.5) * 0.24 + (k - 1) * 0.06
                    y = max(r["rate"], 0.4 / max(r["n"], 1))
                    ax.errorbar(x, y,
                                yerr=[[max(y - r["ci_lo"], 0)],
                                      [max(r["ci_hi"] - y, 0)]],
                                fmt="o", ms=2.2, lw=0.8, capsize=1.5,
                                color=cols[cname],
                                label=(cname if (i == 0 and j == 0) else None))
        ax.axhline(float(p), color="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["T_spike", "T_mid", "T_flat", "T_all"], fontsize=7)
        ax.set_title(f"design $10^{{{int(round(-np.log10(float(p))))}}}$")
        ax.grid(alpha=0.25, which="both")
        ax.set_ylim(2e-5, 1)
    axes[0].set_ylabel("realized $P_{FA}$ (95% exact CI)")
    axes[1].legend(fontsize=6, frameon=False, loc="upper left",
                   title="threshold pool", title_fontsize=6)
    fig.suptitle("H0-side shift: thresholds from one clutter pool, tested on "
                 "the others (blue=C$_{spike}$, orange=C$_{flat}$, "
                 "green=C$_{rand}$)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIG / "fig_h0shift.png", dpi=200)
    fig.savefig(FIG / "fig_h0shift.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
