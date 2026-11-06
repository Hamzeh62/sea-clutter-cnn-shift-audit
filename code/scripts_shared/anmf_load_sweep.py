"""ANMF covariance re-baselining (preempts the V5a reviewer attack).

V5a showed the harness ANMF ~5 dB right of the known-covariance oracle on
white Gaussian clutter (SCM K=1024=2*Np, no loading). This script re-runs the
REAL-data ANMF with covariance variants at matched empirical Pfa:

  orig       : deployed path (unit-norm secondaries, SCM, pinv rcond=1e-9;
               the 1e-10 jitter is negligible under that cutoff)
  dl0.01     : + diagonal loading  sigma^2 = (tr(Chat)/Np)*0.01
  dl0.1      : + diagonal loading  sigma^2 = (tr(Chat)/Np)*0.1
  ledoit     : shrinkage toward scaled identity, closed-form intensity
               (Ledoit-Wolf 2004 style), target = mean eigenvalue.
               (A second target tr(Chat)/Np would be identical here: K > Np
               makes Chat full-rank, so mean eig == tr/Np exactly.)

Trial recipe is byte-identical to phase4_montecarlo sw0 (and d2c's n=500):
  idx500 = rng(seed+7).permutation(len(sampler))[:500], fds from the same rng,
  per-trial rng_t = default_rng(int((scr+30)*2)*100000 + sw*1000 + k),
  amp = sqrt(10^(scr/10) * mean|vv|^2), s_T = swerling_target(..., sw=0, rng_t),
  statistic = ANMF on y = vv + s_T with per-unit secondaries.

Implementation: loading/shrinkage toward scaled identity preserve the
eigenvectors of Chat, so ONE eigendecomposition per unit serves all variants.
In the scoring loop the steering bank P is unitary, so Cinv y = CinvP (P^H y)
-- no eigendecomposition is needed per (unit, SCR) at all.

Outputs: results/json/anmf_load_sweep.json, results/figures/fig_anmf_load.{png,pdf}
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.units import UnitSampler, load_split
from src.eval.metrics import clopper_pearson
from src.baselines.anmf import prep_anmf_unit, anmf_glrt_from_prep, steering_bank
from src.data.synth import swerling_target, estimate_psd, clear_doppler_band

CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]
K_SEC = CFG["baselines"]["anmf"]["K_secondary"]
N_CELLS_SIDE = CFG["baselines"]["anmf"]["secondary_cells_each_side"]
SCR_GRID = CFG["eval"]["scr_grid_db"]
PFA = 1e-3
N_MC = 500
SW = 0
PRF = 1000.0
SEED = CFG["seed"]

P = steering_bank(Np)            # unitary: P^H P = I
PH = P.conj().T
N_H0 = 4000                      # same adaptive-pool size as phase2_calibrate
VARIANTS = ["orig", "dl0.001", "dl0.01", "dl0.1", "ledoit"]
# note: a second shrinkage target (tr/Np instead of mean-eig) would be
# IDENTICAL to `ledoit` here because K=1024 > Np=512 makes Chat full-rank,
# hence mean(sig) == tr/Np exactly; dl0.001 added as an extra ladder rung.

OUT = {
    "config": {"pfa": PFA, "n_mc": N_MC, "swerling": SW, "Np": Np, "K": K_SEC,
               "n_h0": N_H0, "seed": SEED, "variants": VARIANTS,
               "note": "trial recipe identical to phase4 sw0 / d2c n=500; "
                       "sw1/sw2 not run (phase-4 tables show sw0 is the "
                       "binding case for the ANMF gap)"},
    "validation": {},
    "h0_thresholds": {},
    "grid": {},
    "vs_proposed": {},
    "sensitivity": {},
}


# --------------------------------------------------------------------------
# covariance variants (all from ONE eigendecomposition of Chat per unit)
# --------------------------------------------------------------------------
def build_chat(secondary: np.ndarray) -> np.ndarray:
    """Unit-norm rows -> Chat (Np,Np), exactly as prep_anmf_unit builds it."""
    Yn = secondary / np.sqrt(np.sum(np.abs(secondary) ** 2, axis=1,
                                    keepdims=True) + 1e-30)
    return (Yn.conj().T @ Yn) / Yn.shape[0]


def chat_eigh(Chat: np.ndarray):
    """Ascending eigendecomposition of the Hermitian part of Chat."""
    Ch = 0.5 * (Chat + Chat.conj().T)
    sig, Q = np.linalg.eigh(Ch)
    return Q, sig


def shrink_intensity(Chat: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    """Closed-form shrinkage intensity toward mu*I (Ledoit-Wolf 2004 style).

    d2 = (1/Np) * sum_i (sig_i - m)^2   (eigenvalue dispersion),
    b2 = (1/Np) * sum_{i!=j} |T_ij|^2   with T = Chat - m*I (LW beta2),
    w* = min(b2/d2, 1), clipped to [0, 0.9] for numerical safety.
    Returns (w, m) with m = mean eigenvalue.
    """
    m = float(np.mean(sig))
    d2 = float(np.sum((sig - m) ** 2) / len(sig))
    T = Chat - m * np.eye(Chat.shape[0])
    np.fill_diagonal(T, 0.0)
    b2 = float(np.sum(np.abs(T) ** 2) / len(sig))
    w = 0.0 if d2 <= 1e-30 else min(max(b2 / d2, 0.0), 0.9)
    return w, m


def variant_isig(sig: np.ndarray, w: float, m: float, scale: float,
                 mode: str) -> np.ndarray:
    """Per-variant inverse eigenvalues for Cinv = Q diag(isig) Q^H.

    NOTE on 'orig': the deployed path is np.linalg.SOLVE(Chat + jitter, .)
    with jitter = load_jitter*tr(Chat)/Np (prep_anmf_unit) -- an EXACT
    inverse of the jittered matrix, NOT pinv with rcond. On ill-conditioned
    units the near-null directions get 1/(lambda + jitter) ~ 1e12 weights
    that pinv semantics would zero; those units dominate the H0 tail and
    silently shifted the threshold 0.872 -> 0.811 in the first run.
    s = lambda + jitter reproduces solve(Chat + jitter*I, .) EXACTLY
    (Hermitian + delta*I shares eigenvectors).
    """
    if mode == "orig":
        delta = 1e-10 * scale                  # load_jitter * tr(Chat)/Np
        s = sig + delta
    elif mode.startswith("dl"):
        eps = float(mode[2:])
        s = np.maximum(sig, scale * eps)
    elif mode == "ledoit":
        s = (1 - w) * sig + w * m
    elif mode == "ledoit_sc":
        s = (1 - w) * sig + w * scale
    else:
        raise ValueError(mode)
    return 1.0 / np.maximum(s, 1e-30)


def unit_transforms(Chat: np.ndarray):
    """One eigh -> {mode: (W_z, a, e)} with W_z = Q^H P and the per-variant
    inverse-eigenvalue weight vector e (len k), plus (Q, sig, w, m, scale).

    Scoring algebra (z = Q^H y, W = Q^H P, both unitary-shared):
      v_f = P_f^H Cinv y = w_f^H z            (w_f = column f of W)
      d   = y^H Cinv y = sum_i e_i |z_i|^2
      a_f = P_f^H Cinv P_f = sum_i e_i |W_if|^2
    So one (Np,k) matvec + elementwise ops per variant — no per-variant Np^3.
    """
    Q, sig = chat_eigh(Chat)
    w, m = shrink_intensity(Chat, sig)
    scale = float(np.sum(sig) / Np)
    W = Q.conj().T @ P
    out = {}
    for mode in VARIANTS:
        e = variant_isig(sig, w, m, scale, mode)
        a = np.real(np.einsum("ij,ij->j", W.conj(), e[:, None] * W))
        out[mode] = (W, a, e)
    return out, (Q, sig, w, m, scale)


def stat_py(yy: np.ndarray, W: np.ndarray, a: np.ndarray, e: np.ndarray,
            zy: np.ndarray) -> float:
    """ANMF statistic in the shared eigenbasis: z = Q^H y (caller-passed),
    W = Q^H P.  v_f = w_f^H z, d = sum_i e_i |z_i|^2,
    a_f = sum_i e_i |W_if|^2 (precomputed). Exact for every
    diagonal-modification variant (loading/shrinkage).
    """
    v = W.conj().T @ (e * zy)          # v_f = sum_i conj(W[i,f]) e_i z_i
    d = float(np.real(np.sum(e * np.abs(zy) ** 2)))
    v2 = np.abs(v) ** 2
    return float(np.max(v2 / (a * d + 1e-30)))


def bounded_extrapolation(arr: np.ndarray, pfa: float) -> float:
    u = -np.log1p(-np.clip(arr, 0, 1 - 1e-12))
    u0 = float(np.quantile(u, 0.995))
    exc = u[u > u0]
    lam = len(exc) / (exc.sum() - len(exc) * u0 + 1e-30)
    c = len(exc) / len(arr)
    return float(1 - np.exp(-(u0 + np.log(c / pfa) / lam)))


def threshold_for(arr: np.ndarray, pfa: float) -> tuple[float, str]:
    if len(arr) * pfa >= 10:
        return float(np.quantile(arr, 1 - pfa)), "empirical"
    return bounded_extrapolation(arr, pfa), "extrapolated"


def cell_psd_bands(sampler):
    bands = {}
    for key, cube in sampler.data.items():
        f, p = estimate_psd(cube[:, min(3, cube.shape[1] - 1)], PRF, 512)
        bands[key] = clear_doppler_band(p, f, guard_bins=8)
    return bands


# --------------------------------------------------------------------------
def main():
    t00 = time.time()

    split = load_split(SEED)
    sampler = UnitSampler(split["test"], Np, HOP,
                          pol=CFG["data"]["primary_pol"], seed=SEED + 99)
    bands = cell_psd_bands(sampler)

    # ---- validation: eigen path == deployed path ---------------------------
    rng0 = np.random.default_rng(0)
    key, cell, w0 = sampler.units[int(rng0.integers(len(sampler)))]
    sec = sampler.secondary_windows(key, cell, w0, K_SEC, N_CELLS_SIDE)
    assert len(sec) >= Np, "secondary pool too small"

    prep = prep_anmf_unit(sec)
    y = sampler.y_cpi(key, cell, w0)
    t_ref, _ = anmf_glrt_from_prep(y, prep)

    Chat = build_chat(sec)
    ut, (Q, sig, w, m, scale) = unit_transforms(Chat)
    W, a, e = ut["orig"]
    isig = variant_isig(sig, w, m, scale, "orig")
    Chat_rec = (Q * isig) @ Q.conj().T
    CinvP = Chat_rec @ P
    a_direct = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))
    Cinvy_direct = Chat_rec @ y
    t_direct = float(np.max(np.abs(CinvP.conj().T @ y) ** 2
                            / (a_direct * np.real(np.vdot(y, Cinvy_direct))
                               + 1e-30)))
    t_py = stat_py(y, W, a, e, zy=Q.conj().T @ y)
    rel_direct = abs(t_direct - t_ref) / max(t_ref, 1e-30)
    rel_py = abs(t_py - t_ref) / max(t_ref, 1e-30)
    OUT["validation"]["stat_rel_diff_eigen_vs_deployed"] = rel_direct
    OUT["validation"]["stat_rel_diff_pytrick_vs_deployed"] = rel_py
    OUT["validation"]["shrink_w_example"] = w
    print(f"[anmfr] validation: dt_direct={rel_direct:.2e} "
          f"dt_py={rel_py:.2e} w={w:.3f}", flush=True)
    assert rel_py < 1e-8, "eigen/Py-trick path does not reproduce deployed ANMF"

    # ---- H0 calibration (seed-42 order, first N_H0 valid units) ------------
    print(f"[anmfr] H0 calibration: {N_H0} units ...", flush=True)
    cal = np.load(ROOT / "results/json/cal_stats_raw.npz")
    anmf_h0_ref = cal["anmf"]
    order = np.random.default_rng(SEED).permutation(len(sampler))
    h0 = {m: [] for m in VARIANTS}
    guard_checked = 0
    guard_bad = 0
    eig_bad = 0
    for i in order:
        if len(h0["orig"]) >= N_H0:
            break
        key, cell, w0 = sampler.units[i]
        sec = sampler.secondary_windows(key, cell, w0, K_SEC, N_CELLS_SIDE)
        if len(sec) < Np:
            continue
        y = sampler.y_cpi(key, cell, w0)
        # bit-exactness guards: deployed-vs-saved (proves trial order) AND
        # eigen-vs-saved (proves the fast path on the units that matter,
        # including the ill-conditioned tail)
        if guard_checked < len(anmf_h0_ref):
            tp, _ = anmf_glrt_from_prep(y, prep_anmf_unit(sec))
            if not np.isclose(tp, anmf_h0_ref[guard_checked], rtol=1e-8,
                              atol=0.0):
                guard_bad += 1
                if guard_bad == 1:
                    OUT["validation"]["h0_guard_first_mismatch"] = {
                        "at": guard_checked, "recomputed": float(tp),
                        "saved": float(anmf_h0_ref[guard_checked])}
            guard_checked += 1
        ut, (Q_u, _, _, _, _) = unit_transforms(build_chat(sec))
        zy = Q_u.conj().T @ y
        for mm in VARIANTS:
            W, a, e = ut[mm]
            v = stat_py(y, W, a, e, zy)
            h0[mm].append(v)
            if (mm == "orig" and guard_checked <= len(anmf_h0_ref)
                    and not np.isclose(v, anmf_h0_ref[guard_checked - 1],
                                       rtol=1e-4, atol=1e-9)):
                eig_bad += 1
                if eig_bad == 1:
                    OUT["validation"]["h0_eig_guard_first_mismatch"] = {
                        "at": guard_checked - 1, "eigen": float(v),
                        "saved": float(anmf_h0_ref[guard_checked - 1])}
    h0 = {mm: np.asarray(v) for mm, v in h0.items()}
    OUT["validation"]["h0_guard_checked"] = guard_checked
    OUT["validation"]["h0_guard_mismatches"] = guard_bad
    OUT["validation"]["h0_eig_guard_mismatches"] = eig_bad
    print(f"[anmfr] H0 pools done ({time.time()-t00:.0f}s); guards: "
          f"{guard_checked} checked, dep-vs-saved {guard_bad} mismatches, "
          f"eig-vs-saved {eig_bad} mismatches", flush=True)

    thr = {}
    for mm in VARIANTS:
        thr[mm] = threshold_for(h0[mm], PFA)
        OUT["h0_thresholds"][mm] = {"threshold": thr[mm][0],
                                    "method": thr[mm][1],
                                    "n": int(len(h0[mm]))}
        print(f"[anmfr] thr {mm}: {thr[mm][0]:.5f} ({thr[mm][1]})", flush=True)

    # ---- MC (sw0, n=500, phase4 recipe) ------------------------------------
    rng = np.random.default_rng(SEED + 7)
    idx500 = rng.permutation(len(sampler))[:N_MC]
    base = [sampler.units[i] for i in idx500]
    fds = []
    for (key, cell, w0) in base:
        b = bands[key]
        fds.append(float(rng.uniform(b[0], b[1])))

    print("[anmfr] prepping 500 units (one eigh per unit) ...", flush=True)
    uts, w_vals = [], []
    for (key, cell, w0) in base:
        sec = sampler.secondary_windows(key, cell, w0, K_SEC, N_CELLS_SIDE)
        if len(sec) >= Np:
            ut, (Q_u, _, w_u, _, _) = unit_transforms(build_chat(sec))
            uts.append((ut, Q_u))
            w_vals.append(w_u)
        else:
            uts.append(None)
    n_valid = sum(u is not None for u in uts)
    OUT["shrink_intensity_over_units"] = {
        "median": float(np.median(w_vals)),
        "q25": float(np.percentile(w_vals, 25)),
        "q75": float(np.percentile(w_vals, 75)),
        "n": len(w_vals)}
    print(f"[anmfr] prepped {n_valid}/{N_MC} units ({time.time()-t00:.0f}s); "
          f"LW w median {np.median(w_vals):.3f}", flush=True)

    hits = {mm: np.zeros((len(SCR_GRID), N_MC), bool) for mm in VARIANTS}
    wall = {mm: 0.0 for mm in VARIANTS}
    for k_scr, scr in enumerate(SCR_GRID):
        t_scr = time.time()
        for k, (key, cell, w0) in enumerate(base):
            if uts[k] is None:
                continue
            rng_t = np.random.default_rng(int((scr + 30) * 2) * 100000
                                          + SW * 1000 + k)
            vv = sampler.data[key][w0:w0 + Np, cell].astype(np.complex128)
            m2 = np.mean(np.abs(vv) ** 2)
            amp = np.sqrt(10 ** (scr / 10.0) * m2)
            s_T = swerling_target(Np, fds[k], PRF, amp, SW, rng_t)
            yy = vv + s_T
            Q_u = uts[k][1]
            zy = Q_u.conj().T @ yy
            for mm in VARIANTS:
                W, a, e = uts[k][0][mm]
                t0 = time.time()
                v = stat_py(yy, W, a, e, zy)
                wall[mm] += time.time() - t0
                hits[mm][k_scr, k] = v > thr[mm][0]
        msg = " ".join(f"{mm}={hits[mm][k_scr].mean():.3f}" for mm in VARIANTS)
        print(f"[anmfr] SCR {scr:+6.1f} dB ({time.time()-t_scr:.0f}s): {msg}",
              flush=True)

    # ---- results + orig-vs-phase4 cross-check -------------------------------
    mc = json.loads((ROOT / "results/json/mc_results.json").read_text())
    ph4 = {r["scr_db"]: r["pd"] for r in mc["pd"]["sw0"]["ANMF"]["0.001"]}
    for k_scr, scr in enumerate(SCR_GRID):
        row = {}
        for mm in VARIANTS:
            h = hits[mm][k_scr]
            n = len(h)
            kd = int(h.sum())
            lo, hi = clopper_pearson(kd, n, 0.05)
            row[mm] = {"pd": kd / max(n, 1), "ci_lo": lo, "ci_hi": hi,
                       "k": kd, "n": n,
                       "ms_per_trial": 1000 * wall[mm] / (n * len(SCR_GRID))}
        row["_phase4_orig_ref"] = ph4.get(scr)
        OUT["grid"][f"{scr:g}"] = row
        if ph4.get(scr) is not None and abs(row["orig"]["pd"] - ph4[scr]) > 1e-9:
            OUT["validation"].setdefault("orig_vs_phase4_mismatch", [])\
                .append({"scr": scr, "eigen": row["orig"]["pd"],
                         "phase4": ph4[scr]})
        print(f"[anmfr] table SCR {scr:+6.1f}: " +
              " ".join(f"{mm} {row[mm]['pd']:.3f}" for mm in VARIANTS) +
              f" | phase4-orig {ph4.get(scr)}", flush=True)

    # ---- comparison vs Proposed (same 500 paired trials, totals-based) -----
    prop_pol = next(r for r in mc["pd"]["sw0"]["Proposed"]["0.001"]
                    if r["scr_db"] == 0)
    prop_delay = json.loads(
        (ROOT / "results/json/d2c_dirvs_ood.json").read_text())
    scr0 = [i for i, s in enumerate(SCR_GRID) if s == 0][0]
    best = max(VARIANTS, key=lambda mm: int(hits[mm][scr0].sum()))
    OUT["vs_proposed"] = {
        "scr_db": 0, "n": N_MC,
        "paired_trials_note": (
            "ANMF variants and both Proposed variants are evaluated on the "
            "SAME 500 trials (phase4 sw0 recipe == d2c n=500 recipe). "
            "Per-trial Proposed hit vectors are not persisted by any prior "
            "run, so a paired McNemar test is NOT computable without a "
            "re-run; comparisons here are totals-based with marginal CIs. "
            "Stated explicitly (no silent skip)."),
        "proposed_polarimetric": {"pd": prop_pol["pd"],
                                  "ci_lo": prop_pol["ci_lo"],
                                  "ci_hi": prop_pol["ci_hi"]},
        "proposed_delay_warm": {"pd": prop_delay["pd_ood"],
                                "threshold": prop_delay["config"]["thr_ood_d2b"]},
        "anmf_best_variant": best,
        "anmf_best": OUT["grid"][f"{SCR_GRID[scr0]:g}"][best],
        "anmf_orig": OUT["grid"][f"{SCR_GRID[scr0]:g}"]["orig"],
    }

    # ---- sensitivity: SCR at Pd=0.5 crossing (linear interp) ----------------
    def crossing(mm):
        pds = [OUT["grid"][f"{s:g}"][mm]["pd"] for s in SCR_GRID]
        for i in range(1, len(pds)):
            if pds[i - 1] < 0.5 <= pds[i]:
                x0, x1 = SCR_GRID[i - 1], SCR_GRID[i]
                return x0 + (0.5 - pds[i - 1]) / (pds[i] - pds[i - 1]) * (x1 - x0)
        return None
    sens = {}
    for mm in VARIANTS:
        sens[mm] = {"scr_at_pd50": crossing(mm)}
    base_c = sens["orig"]["scr_at_pd50"]
    if base_c is not None:
        for mm in VARIANTS:
            if sens[mm]["scr_at_pd50"] is not None:
                sens[mm]["shift_vs_orig_db"] = \
                    sens[mm]["scr_at_pd50"] - base_c
    OUT["sensitivity"] = sens
    print(f"[anmfr] SCR@Pd50: " +
          " ".join(f"{mm}={sens[mm]['scr_at_pd50']}" for mm in VARIANTS),
          flush=True)

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    styles = {"orig": ("C7", "o", "--", "ANMF orig (SCM, no loading)"),
              "dl0.001": ("C4", ".", "-", "ANMF + DL 0.001"),
              "dl0.01": ("C0", "s", "-", "ANMF + DL 0.01"),
              "dl0.1": ("C1", "^", "-", "ANMF + DL 0.1"),
              "ledoit": ("C2", "D", "-", "ANMF + Ledoit-Wolf shrinkage")}
    for mm in VARIANTS:
        c, mk, ls, lab = styles[mm]
        pds = np.array([OUT["grid"][f"{s:g}"][mm]["pd"] for s in SCR_GRID])
        lo = np.array([OUT["grid"][f"{s:g}"][mm]["ci_lo"] for s in SCR_GRID])
        hi = np.array([OUT["grid"][f"{s:g}"][mm]["ci_hi"] for s in SCR_GRID])
        ax.plot(SCR_GRID, pds, marker=mk, ls=ls, color=c, label=lab, ms=4)
        ax.fill_between(SCR_GRID, lo, hi, color=c, alpha=0.12)
    ax.axhline(prop_pol["pd"], color="red", lw=1.2, ls=":",
               label=f"Proposed polarimetric (same trials): {prop_pol['pd']:.3f}")
    ax.axhline(prop_delay["pd_ood"], color="darkred", lw=1.2, ls="-.",
               label=f"Proposed delay-warm (same trials): {prop_delay['pd_ood']:.3f}")
    ax.set_xlabel("SCR (dB, per-pulse input)")
    ax.set_ylabel("Pd")
    ax.set_title(f"ANMF covariance re-baselining (real IPIX clutter, sw0, "
                 f"n=500, matched Pfa={PFA:g})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/fig_anmf_load.png", dpi=300)
    fig.savefig(ROOT / "results/figures/fig_anmf_load.pdf")
    plt.close(fig)

    (ROOT / "results/json/anmf_load_sweep.json").write_text(
        json.dumps(OUT, indent=2))
    print(f"[anmfr] ALL DONE ({(time.time()-t00)/60:.1f} min) — wrote "
          f"anmf_load_sweep.json + fig_anmf_load.*", flush=True)


if __name__ == "__main__":
    main()
