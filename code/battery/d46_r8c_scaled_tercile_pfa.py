"""T1 (task R8c) — Scaled per-sea-state calibration on the full 21,462-unit pool.

Replaces the draft's n=100 per-tercile realized-Pfa claim (a 20x-nominal point
estimate with no CI — indefensible) with the scaled analysis the handoff
specifies:

  - per-unit MoM nu-hat (K shape, the project's sea-state proxy) on the FULL
    21,462-unit phase-2 pool (both CNN score arrays already saved in
    d40_scores_phase2.npz; nu-hat values from the phase-1 whole-cell MoM fit
    k_moments, the same parameterization clutter_stats.json uses);
  - tercile split of the pool by nu-hat (exact 1/3–2/3 percentiles);
  - realized Pfa at the deployed 1e-3 and 1e-4 thresholds for CNN-C / CNN-IQ /
    hybrid (Proposed) / loaded-ANMF (DL 0.001), each with exact Clopper-Pearson
    CIs at the design level;
  - the pool-size asymmetry stated explicitly (CNN pools cover the full
    21,462-unit phase-2 raw pool; the hybrid/ANMF pools are their own
    calibration pools: 1,500 seed-77 units / 4,000 seed-42 units).

No detector is re-run: every score array is a saved artifact.
  CNN-C/CNN-IQ : results/json/d40_scores_phase2.npz  (full_c, full_r — pool
                 POSITION order = seed-42 sampler permutation order)
  hybrid       : recomputed here on the seed-77 1,500-unit pool with the
                 deployed path, byte-guarded against prop_h0_stats.npz
                 (the guard IS the recipe pin); scores reused from the guard.
  ANMF DL0.001 : results/json/anmf_load_ext_progress.npz h0_dl0.001
                 (4,000 units, seed-42 permutation order, validated by D29).

Guards (abort conditions):
  G1  hybrid recomputation reproduces prop_h0_stats.npz exactly (max|diff|
      recorded; abort if > 1e-10 relative) — pins sampler/rng/window stream.
  G2  CNN pool-array positional sanity: the first 1,500 positions of
      d40_scores_phase2 full arrays are NOT the pool77 units (different
      sampler seed); instead we assert the array stats match the d40 artifact
      (n = 21,462; realized-Pfa @1e-3 reproduces d45b's corrected values).

Nu-hat qualification: a pool unit's (file, cell) must carry a MoM nu from
phase-1 (clutter_stats.json); units from unmeasured cells are excluded from
tercile assignment (counted; CNN arms additionally report the full-pool rate).

Outputs: results/json/d46_r8c_scaled_tercile_pfa.json
         results/figures/fig_r8c_tercile_pfa.{png,pdf}
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
# NOTE: no OMP_NUM_THREADS / torch.set_num_threads — byte-exact replay guards
# (deviations D44/D38 lessons).
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.units import UnitSampler, load_split  # noqa: E402
from src.proposed.infer import HybridDetector  # noqa: E402
from src.baselines.anmf import steering_bank  # noqa: E402

CFG = None
import yaml  # noqa: E402
CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]
SEED = CFG["seed"]

JS = ROOT / "results/json"
OUTJ = JS / "d46_r8c_scaled_tercile_pfa.json"
FIG = ROOT / "results/figures"


def log(msg):
    print(f"[d46] {msg}", flush=True)


def acquire_lock():
    """Single-job guard (16 GB machine discipline): exclusive-create pid lock.
    Two concurrent MC jobs have OOM-killed this box before."""
    lock = JS / "d46_run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("[d46] another instance holds the lock - exiting", flush=True)
        return None
    return lock


def load_nu_hat_map():
    """(file_stem, cell) -> MoM nu from phase-1 clutter_stats.json (the same
    source d45's load_nu_hat_map uses; MoM primary, not the KS polish)."""
    st_ = json.loads((JS / "clutter_stats.json").read_text())
    nmap = {}
    for fkey, fval in st_.items():
        stem = fkey[:-4] if fkey.endswith(".npz") else fkey
        for ci, c in enumerate(fval["cells"]):
            kk = c.get("K") or {}
            v = kk.get("nu")
            if v is not None and 0.05 < v < 100:
                nmap[(stem, ci)] = float(v)
    return nmap


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
    out = {
        "task": "T1/R8c scaled per-tercile calibration (replaces the n=100 tercile Pfa)",
        "config": {}, "validation": {}, "pools": {}, "nu_pool": {},
        "terciles": {}, "realized_pfa": {}, "notes": [],
    }

    # ---------------- deployed thresholds (unchanged) --------------------
    d40j = json.loads((JS / "d40_cnn_baselines.json").read_text())
    thrC = {n: {p: float(d40j["thresholds"][n][p]["threshold"])
                for p in ("0.001", "0.0001")} for n in ("CNN-C", "CNN-IQ")}
    ph0 = np.load(JS / "prop_h0_stats.npz")["stats"]
    thrP = {p: float(np.quantile(ph0, 1 - float(p))) for p in ("0.001", "0.0001")}
    extj = json.loads((JS / "anmf_load_ext.json").read_text())
    thrA = {k.split("|")[1]: float(v["threshold"])
            for k, v in extj["h0_thresholds"].items() if k.startswith("dl0.001|")}
    out["config"]["thresholds"] = {"CNN-C": thrC["CNN-C"], "CNN-IQ": thrC["CNN-IQ"],
                                   "Proposed": thrP, "ANMF-DL0.001": thrA}
    out["config"]["threshold_provenance"] = {
        "CNNs": "d40 phase-4 rule on the seed-77 1,500-unit pool (extrapolated both levels)",
        "Proposed": "empirical quantiles of the same seed-77 1,500-unit pool (prop_h0_stats)",
        "ANMF-DL0.001": "deployed extrapolated thresholds (anmf_load_ext.json, 4,000-unit pool)"}

    # ---------------- pool arrays ----------------------------------------
    z42 = np.load(JS / "d40_scores_phase2.npz")
    full_c, full_r = z42["full_c"], z42["full_r"]
    assert full_c.shape == full_r.shape == (21462,), "phase-2 pool shape changed"
    h0_dl = np.load(JS / "anmf_load_ext_progress.npz")["h0_dl0.001"]
    assert h0_dl.shape == (4000,), "ANMF H0 pool shape changed"

    # ---------------- G2: CNN pool sanity vs d45b ------------------------
    d45b = json.loads((JS / "d45b_d40_pfa_fixes.json").read_text())
    rp = d45b["d40_defect_a"]["realized_pfa_phase2_full"]
    for name, arr in (("CNN-C", full_c), ("CNN-IQ", full_r)):
        for p in ("0.001", "0.0001"):
            r = float(np.mean(arr > thrC[name][p]))
            ref = rp[f"{name}@{p}"]
            out["validation"][f"cnn_{name}_{p}_rate_matches_d45b"] = \
                bool(abs(r - ref) < 1e-12)
            assert abs(r - ref) < 1e-12, f"{name}@{p}: {r} vs d45b {ref}"
    log("G2 CNN pool arrays reproduce the d45b realized-Pfa values exactly")

    # ---------------- G1: hybrid pool recomputation (recipe pin) ---------
    # D52: must replicate phase-4 compute_proposed_h0 EXACTLY — the per-window
    # det.statistic() path (batch-1 LSTM forward + per-window whiten inside
    # statistic). The D36 hoisted batched path (network_out_batch +
    # statistic_from_net, first attempt) drifts to rel 1.27e-01 vs
    # prop_h0_stats.npz: the batch-32 LSTM forward is not bit-identical to
    # batch-1 (BLAS reduction order), and ICA fixed-point iterations amplify
    # the epsilon into 1e-1-scale statistic differences. d39's guard already
    # validated THIS stream (seed-77 test permutation) at rel <= 1e-9 via
    # det.statistic(); we reuse the same path for all 1,500 units.
    det = HybridDetector(ROOT / "results/checkpoints/best_s0.pt")
    zsep = np.load(JS / "csep_prep.npz")
    Chat = zsep["Chat"]
    P = steering_bank(int(zsep["Np"]))
    CinvP = np.linalg.solve(Chat, P)
    a = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))
    prep_sep = {"Chat": Chat, "CinvP": CinvP, "a": a}
    det._prep_sep["__default__"] = prep_sep

    split = load_split(SEED)
    sampler77 = UnitSampler(split["test"], Np, HOP, seed=77)  # Proposed pool units
    rng77 = np.random.default_rng(77)
    order77 = rng77.permutation(len(sampler77))[:1500]
    hyb = np.empty(1500)
    for j, i in enumerate(order77):
        key, cell, w0 = sampler77.units[i]
        vv = sampler77.data[key][w0:w0 + Np, cell].astype(np.complex128)
        vh = sampler77.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)
        hyb[j], _ = det.statistic(np.stack([vv, vh]), prep_sep=prep_sep)
        if (j + 1) % 250 == 0:
            log(f"  hybrid pool {j + 1}/1500 ({time.time() - t00:.0f}s elapsed)")
    rel = float(np.max(np.abs(hyb - ph0) / np.abs(ph0)))
    out["validation"]["hybrid_pool_guard_max_rel_diff"] = rel
    out["validation"]["hybrid_compute_path"] = "per-window det.statistic (batch-1 LSTM), exact phase-4 compute_proposed_h0 replication"
    log(f"G1 hybrid pool guard: max rel diff {rel:.2e} over 1,500 units")
    if rel > 1e-10:
        log("ABORT — hybrid recipe does not reproduce prop_h0_stats")
        OUTJ.write_text(json.dumps(out, indent=2))
        return

    # pool units for nu assignment
    units77 = [sampler77.units[i] for i in order77]

    # ---------------- per-unit MoM nu-hat --------------------------------
    nu_map = load_nu_hat_map()
    log(f"nu map: {len(nu_map)} cells with MoM nu")

    def nu_for(units):
        return np.array([nu_map.get((key, cell), np.nan)
                         for (key, cell, w0) in units])

    nu_cnn = nu_for([u for u in _phase2_units()])       # position order
    nu_hyb = nu_for(units77)
    nu_anmf = nu_for(_anmf_units())

    out["nu_pool"] = {
        "source": "results/json/clutter_stats.json (phase-1 MoM k_moments, "
                  "whole-cell amplitudes; same source as d45's terciles)",
        "cnn_matched": int(np.sum(~np.isnan(nu_cnn))),
        "hybrid_matched": int(np.sum(~np.isnan(nu_hyb))),
        "anmf_matched": int(np.sum(~np.isnan(nu_anmf))),
        "cnn_median": float(np.nanmedian(nu_cnn)),
        "cnn_iqr": [float(np.nanpercentile(nu_cnn, 25)),
                    float(np.nanpercentile(nu_cnn, 75))],
    }

    # ---------------- terciles on the full CNN pool ----------------------
    v = nu_cnn[~np.isnan(nu_cnn)]
    t1, t2 = np.percentile(v, [100 / 3, 200 / 3])
    log(f"full-pool terciles: t1={t1:.3f}, t2={t2:.3f} (n={len(v)})")
    out["terciles"] = {"t1": float(t1), "t2": float(t2),
                       "n_valid": int(len(v)),
                       "definition": "spike = nu <= t1, mid = t1 < nu <= t2, "
                                     "flat = nu > t2 (lower nu = spikier)"}

    def tercile_masks(nu_arr):
        ok = ~np.isnan(nu_arr)
        return {"spike": ok & (nu_arr <= t1),
                "mid": ok & (nu_arr > t1) & (nu_arr <= t2),
                "flat": ok & (nu_arr > t2),
                "unmatched": ~ok}

    # ---------------- realized Pfa per arm per tercile --------------------
    from src.eval.metrics import clopper_pearson
    ALPHA = 0.05

    def rate_row(hits):
        k = int(hits.sum()); n = int(len(hits))
        lo, hi = clopper_pearson(k, n, ALPHA)
        return {"k": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi}

    m_cnn = tercile_masks(nu_cnn)
    m_hyb = tercile_masks(nu_hyb)
    m_anmf = tercile_masks(nu_anmf)

    arms = {
        "CNN-C": {"scores": full_c, "masks": m_cnn, "thr": thrC["CNN-C"],
                  "pool": "phase-2 full pool (21,462 units, seed-42 permutation order)"},
        "CNN-IQ": {"scores": full_r, "masks": m_cnn, "thr": thrC["CNN-IQ"],
                   "pool": "phase-2 full pool (21,462 units, seed-42 permutation order)"},
        "Proposed": {"scores": hyb, "masks": m_hyb, "thr": thrP,
                     "pool": "seed-77 calibration pool (1,500 units; in-sample)"},
        "ANMF-DL0.001": {"scores": h0_dl, "masks": m_anmf, "thr": thrA,
                         "pool": "seed-42-order calibration pool (4,000 units; in-sample)"},
    }
    for name, A in arms.items():
        out["realized_pfa"][name] = {"pool": A["pool"], "thresholds": A["thr"]}
        for p in ("0.001", "0.0001"):
            hits_all = A["scores"] > A["thr"][p]
            blk = {"full_pool": rate_row(hits_all)}
            for tname in ("spike", "mid", "flat"):
                blk[tname] = rate_row(hits_all[A["masks"][tname]])
            blk["unmatched"] = rate_row(hits_all[A["masks"]["unmatched"]])
            out["realized_pfa"][name][p] = blk
            log(f"{name} @{p}: full {blk['full_pool']['k']}/{blk['full_pool']['n']}"
                f" = {blk['full_pool']['rate']:.2e} | spike "
                f"{blk['spike']['k']}/{blk['spike']['n']}"
                f" = {blk['spike']['rate']:.2e} "
                f"[{blk['spike']['ci_lo']:.2e},{blk['spike']['ci_hi']:.2e}]")

    # ---------------- figures --------------------------------------------
    _fig(out)
    out["notes"] = [
        "Design 1e-3/1e-4 realized rates; CIs are exact Clopper-Pearson at "
        "each observed k/n (binomial, two-sided 95%).",
        "Pool-size asymmetry (stated as required): the CNN arms see the full "
        "21,462-unit independent pool; the hybrid and loaded-ANMF see their "
        "own calibration pools (1,500 / 4,000 units) — the model-based rates "
        "are in-sample by construction and their CIs are correspondingly "
        "wider at the same k.",
        "Spike/mid/flat terciles are defined on the CNN pool's nu distribution "
        "(t1/t2 percentiles reported); the same thresholds are applied to the "
        "hybrid/ANMF pools' units for their splits.",
        "Spike tercile = smallest MoM nu (heaviest tails); 'flat' = largest nu.",
        "No detector re-run: CNN scores from d40_scores_phase2.npz; ANMF H0 "
        "from anmf_load_ext_progress.npz (D29-validated); hybrid recomputed "
        "once and byte-guarded against prop_h0_stats.npz.",
    ]
    out["wall_s"] = time.time() - t00
    OUTJ.write_text(json.dumps(out, indent=1))
    log(f"wrote {OUTJ.name} ({out['wall_s']:.0f}s)")


def _phase2_units():
    """The phase-2 pool units in POSITION order (sampler seed 42, rng 42
    permutation; exact replication of d40 stage_phase2's unit stream)."""
    if getattr(_phase2_units, "_cache", None) is not None:
        return _phase2_units._cache
    split = load_split(SEED)
    sampler = UnitSampler(split["test"], Np, HOP,
                          pol=CFG["data"]["primary_pol"], seed=SEED)
    n_total = len(sampler)
    assert n_total == 21462, f"unexpected n_total {n_total}"
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_total)
    units = [sampler.units[i] for i in order]
    _phase2_units._cache = units
    return units


def _anmf_units():
    """The 4,000-unit ANMF H0 pool units (seed-42 permutation order, first
    4,000 passing len(sec) >= Np; exact replication of anmf_load_ext's pool)."""
    if getattr(_anmf_units, "_cache", None) is not None:
        return _anmf_units._cache
    split = load_split(SEED)
    sampler = UnitSampler(split["test"], Np, HOP,
                          pol=CFG["data"]["primary_pol"], seed=SEED + 99)
    K_SEC = CFG["baselines"]["anmf"]["K_secondary"]
    N_SIDE = CFG["baselines"]["anmf"]["secondary_cells_each_side"]
    order = np.random.default_rng(SEED).permutation(len(sampler))
    keep = []
    for i in order:
        if len(keep) >= 4000:
            break
        key, cell, w0 = sampler.units[i]
        sec = sampler.secondary_windows(key, cell, w0, K_SEC, N_SIDE)
        if len(sec) >= Np:
            keep.append((key, cell, w0))
    _anmf_units._cache = keep
    return keep


def _fig(out):
    plt_style = {"font.size": 8}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(plt_style)
    arms = ("CNN-C", "CNN-IQ", "Proposed", "ANMF-DL0.001")
    labs = {"CNN-C": "CNN-C", "CNN-IQ": "CNN-IQ",
            "Proposed": "Hybrid (Proposed)", "ANMF-DL0.001": "ANMF (DL $10^{-3}$)"}
    cols = {"CNN-C": "#8C564B", "CNN-IQ": "#7F7F7F",
            "Proposed": "#D55E00", "ANMF-DL0.001": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, p in zip(axes, ("0.001", "0.0001")):
        for i, arm in enumerate(arms):
            blk = out["realized_pfa"][arm][p]
            xs, ys, los, his = [], [], [], []
            for j, t in enumerate(("spike", "mid", "flat", "full_pool")):
                r = blk[t]
                xs.append(j + (i - 1.5) * 0.19)
                ys.append(max(r["rate"], 0.4 / max(r["n"], 1)))
                los.append(max(r["rate"] - r["ci_lo"], 0))
                his.append(max(r["ci_hi"] - r["rate"], 0))
            ax.errorbar(xs, ys, yerr=[los, his], fmt="o", ms=3, lw=1,
                        color=cols[arm], label=labs[arm], capsize=2)
        ax.axhline(float(p), color="k", ls="--", lw=0.8)
        ax.text(3.45, float(p) * 1.35, f"design $10^{{{int(round(-np.log10(float(p))))}}}$",
                fontsize=7, va="bottom", ha="right")
        ax.set_yscale("log")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["spike", "mid", "flat", "full pool"])
        ax.set_title(f"$P_{{FA}}$ design $10^{{{int(round(-np.log10(float(p))))}}}$")
        ax.grid(alpha=0.25, which="both")
        ax.set_ylim(3e-6, 1.2e-1)
    axes[0].set_ylabel("realized $P_{FA}$ (95% exact CI)")
    axes[1].legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.suptitle("Scaled per-sea-state realized $P_{FA}$ (per-unit MoM $\\hat\\nu$ terciles)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIG / "fig_r8c_tercile_pfa.png", dpi=200)
    fig.savefig(FIG / "fig_r8c_tercile_pfa.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
