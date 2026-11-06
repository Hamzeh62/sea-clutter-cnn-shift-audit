"""D40 (stage 2) — M6 learned baselines: calibration, paired MC, tails, verdict.

Consumes the two CNN checkpoints from d40_train_cnn_baselines.py and produces
results/json/d40_cnn_baselines.json + two figures. Stages are cached to npz
so a restart never redoes completed scoring.

Stages
  1  seed-77 H0 pool (the SAME 1,500 test units as the Proposed detector's
     calibration pool) -> CNN score pools -> thresholds at 1e-3/1e-4 via the
     phase-4 rule (empirical if n*pfa >= 10 else the M3 exponential tail above
     the 0.995 quantile — CNN scores are unbounded, so the raw fit applies).
  2  phase-2 pool: score ALL clutter-only test units (permutation of seed-42
     sampler) for both CNNs; also identify the exact 4000-unit ANMF H0 subset
     (replicating phase-2's secondary-window qualification) -> independent
     realized-Pfa checks + ROC H0 baseline.
  3  paired sw0 Monte-Carlo, phase-4 recipe byte-exact (sampler seed 141,
     rng 49, per-trial rng_t; d38-validated): 17-point SCR grid x n=500.
     Guard: the Proposed statistic at 0 dB must reproduce v1_repro_sw0.json
     exactly before any CNN number is trusted.
  4  analysis: Pd rows + Clopper-Pearson CIs; paired McNemar vs the saved
     ANMF-DL0.001 arm (all 17 SCRs) and vs the Proposed arm (0/+10 dB, v1);
     ROC AUC / common-range AUC / pAUC per the d31 recipe; SCR@Pd50;
     realized Pfa at the calibrated thresholds on three pools (in-sample
     seed-77 pool, phase-2 4000-pool, phase-2 full pool) with the deployed
     ANMF-DL0.001 thresholds as reference; H0-tail figure; runtime.

Figures: results/figures/fig_cnn_power.{pdf,png}, fig_cnn_h0tail.{pdf,png}
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch
from tqdm import tqdm
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.units import UnitSampler, load_split
from src.eval.metrics import clopper_pearson, auc_from_scores
from src.eval.plots import STYLE
from src.data.synth import swerling_target, estimate_psd, clear_doppler_band
from src.proposed.infer import HybridDetector
from src.baselines.anmf import steering_bank

CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]
PRF = 1000.0
SEED = CFG["seed"]
K_SEC = CFG["baselines"]["anmf"]["K_secondary"]
SCR_GRID = CFG["eval"]["scr_grid_db"]
N_MC = CFG["eval"]["n_mc"]
CI = CFG["eval"]["ci"]
ROC_SCR = CFG["eval"]["roc_scr_db"]
N_ADAPTIVE = 4000

CKPT_C = ROOT / "results/checkpoints/best_cnn_complex.pt"
CKPT_R = ROOT / "results/checkpoints/best_cnn_real_iq.pt"
JS = ROOT / "results/json"
OUTJ = JS / "d40_cnn_baselines.json"
FIG = ROOT / "results/figures"
C77 = JS / "d40_scores_pool77.npz"
C42 = JS / "d40_scores_phase2.npz"
CTR = JS / "d40_scores_trials.npz"

sys.path.insert(0, str(ROOT / "scripts"))
from d40_train_cnn_baselines import CNNComplex, CNNRealIQ, normalize  # noqa: E402

PFA_GRID = (1e-3, 1e-4)
CNN_STYLE = {
    "CNN-C": dict(color="#8C564B", marker="v", linestyle="-", label="CNN (complex)"),
    "CNN-IQ": dict(color="#7F7F7F", marker="d", linestyle="-", label="CNN (I/Q real)"),
}


def make_target(scr_db, f_d, mode, rng, clutter_ref):
    m2 = np.mean(np.abs(clutter_ref) ** 2)
    amp = np.sqrt(10 ** (scr_db / 10.0) * m2)
    return swerling_target(Np, f_d, PRF, amp, mode, rng)


def score_windows(net, xs, batch=256, desc=""):
    """Batched forward over (n, 2, Np) complex windows; returns (n,) scores."""
    out = np.empty(len(xs), np.float64)
    with torch.no_grad():
        for i in tqdm(range(0, len(xs), batch), desc=desc, leave=False,
                      total=(len(xs) + batch - 1) // batch):
            xb = torch.from_numpy(normalize(xs[i:i + batch]).astype(np.complex64))
            out[i:i + batch] = net(xb).numpy()
    return out


def thresholds_for(arr, pfa):
    """phase-4 rule: empirical quantile if n*pfa >= 10 else M3 exponential tail."""
    if len(arr) * pfa >= 10:
        return float(np.quantile(arr, 1 - pfa)), "empirical"
    t0 = float(np.quantile(arr, 0.995))
    exc = arr[arr > t0]
    lam = len(exc) / (exc.sum() - len(exc) * t0 + 1e-30)
    c = len(exc) / len(arr)
    return float(t0 + np.log(c / pfa) / lam), "extrapolated"


def mean_tpr_log(f, t, lo, hi):
    o = np.argsort(f)
    f, t = np.asarray(f)[o], np.asarray(t)[o]
    lf = np.log10(f)
    lg, hg = np.log10(lo), np.log10(hi)
    xs = np.concatenate([[lg], lf[(lf > lg) & (lf < hg)], [hg]])
    ys = np.interp(xs, lf, t)
    return float(np.trapezoid(ys, xs) / (hg - lg))


def scr_at_pd50(rows):
    xs = np.array([r["scr_db"] for r in rows], float)
    ys = np.array([r["pd"] for r in rows], float)
    if ys.max() < 0.5:
        return None
    for i in range(1, len(ys)):
        if ys[i - 1] < 0.5 <= ys[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return float(xs[int(np.argmax(ys >= 0.5))])


def load_nets():
    netC, netR = CNNComplex(), CNNRealIQ()
    netC.load_state_dict(torch.load(CKPT_C, map_location="cpu"))
    netR.load_state_dict(torch.load(CKPT_R, map_location="cpu"))
    netC.eval(); netR.eval()
    return netC, netR


# ---------------------------------------------------------------- stage 1
def stage_pool77(netC, netR):
    if C77.exists():
        z = np.load(C77)
        return z["c_c"], z["c_r"]
    split = load_split(SEED)
    sampler = UnitSampler(split["test"], Np, HOP, seed=77)  # Proposed pool units
    rng = np.random.default_rng(77)
    idx = rng.permutation(len(sampler))[:1500]
    xs = np.stack([
        np.stack([sampler.data[key][w0:w0 + Np, cell].astype(np.complex128),
                  sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)])
        for (key, cell, w0) in [sampler.units[i] for i in idx]])
    c_c = score_windows(netC, xs, desc="pool77 CNN-C")
    c_r = score_windows(netR, xs, desc="pool77 CNN-IQ")
    np.savez(C77, c_c=c_c, c_r=c_r)
    return c_c, c_r


# ---------------------------------------------------------------- stage 2
def anmf_unit_subset(sampler, order):
    """First N_ADAPTIVE units (in phase-2 permutation order) whose secondary
    windows pass phase-2's `len(sec) >= Np` check. Exact replication."""
    from src.baselines.anmf import prep_anmf_unit  # noqa: F401  (parity with phase-2)
    keep = []
    for i in tqdm(order, desc="anmf-subset scan"):
        if len(keep) >= N_ADAPTIVE:
            break
        key, cell, w0 = sampler.units[i]
        sec = sampler.secondary_windows(key, cell, w0, K_SEC,
                                        CFG["baselines"]["anmf"]["secondary_cells_each_side"])
        if len(sec) >= Np:
            keep.append(i)
    return np.array(keep)


def stage_phase2(netC, netR):
    if C42.exists():
        z = np.load(C42)
        return z["full_c"], z["full_r"], z["anmf4000_idx"]
    split = load_split(SEED)
    sampler = UnitSampler(split["test"], Np, HOP, pol=CFG["data"]["primary_pol"], seed=SEED)
    n_total = len(sampler)
    assert n_total == 21462, f"unexpected n_total {n_total}"
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_total)
    xs = np.stack([
        np.stack([sampler.data[key][w0:w0 + Np, cell].astype(np.complex128),
                  sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)])
        for (key, cell, w0) in [sampler.units[i] for i in order]])
    full_c = score_windows(netC, xs, desc="phase2 CNN-C")
    full_r = score_windows(netR, xs, desc="phase2 CNN-IQ")
    keep = anmf_unit_subset(sampler, order)
    np.savez(C42, full_c=full_c, full_r=full_r, anmf4000_idx=keep)
    return full_c, full_r, keep


# ---------------------------------------------------------------- stage 3
def stage_trials(netC, netR):
    if CTR.exists():
        z = np.load(CTR)
        return z["c"], z["r"], float(z["guard_maxdiff"]), int(z["guard_mismatch"])
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

    # ---- guard: Proposed statistic at 0 dB must reproduce v1 exactly -------
    det = HybridDetector(ROOT / "results/checkpoints/best_s0.pt")
    z = np.load(JS / "csep_prep.npz")
    Chat = z["Chat"]
    P = steering_bank(int(z["Np"]))
    CinvP = np.linalg.solve(Chat, P)
    a = np.real(np.einsum("ij,ij->j", P.conj(), CinvP))
    prep_sep = {"Chat": Chat, "CinvP": CinvP, "a": a}
    det._prep_sep["__default__"] = prep_sep
    v1 = json.loads((JS / "v1_repro_sw0.json").read_text())
    xs = []
    for k, (key, cell, w0) in enumerate(base0):
        rng_t = np.random.default_rng(int((0 + 30) * 2) * 100000 + 0 * 1000 + k)
        vv = sampler.data[key][w0:w0 + Np, cell].astype(np.complex128)
        s_T = make_target(0, fds0[k], 0, rng_t, vv)
        vh = sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)
        xs.append(np.stack([vv + s_T, vh]))
    net_outs = det.network_out_batch(np.stack(xs))
    s_replay = np.empty(N_MC)
    for k in tqdm(range(N_MC), desc="guard Proposed 0dB"):
        tp, _ = det.statistic_from_net(xs[k], net_outs[k], prep_sep=prep_sep)
        s_replay[k] = float(tp)
    s_saved = np.asarray(v1["stats"]["0"]["Proposed"])
    mism = int(np.sum(~np.isclose(s_replay, s_saved, rtol=1e-10, atol=1e-12)))
    maxd = float(np.max(np.abs(s_replay - s_saved)))
    print(f"[d40] guard Proposed @0 dB: {mism} mismatches (max |diff| {maxd:.2e})",
          flush=True)
    if mism:
        raise SystemExit("ABORT: trial stream does not reproduce the v1 reference")

    # ---- CNN scores over the 17-point grid ---------------------------------
    sc = np.zeros((len(SCR_GRID), N_MC))
    sr = np.zeros((len(SCR_GRID), N_MC))
    for k_scr, scr in enumerate(SCR_GRID):
        xs = []
        for k, (key, cell, w0) in enumerate(base0):
            rng_t = np.random.default_rng(int((scr + 30) * 2) * 100000 + 0 * 1000 + k)
            vv = sampler.data[key][w0:w0 + Np, cell].astype(np.complex128)
            s_T = make_target(scr, fds0[k], 0, rng_t, vv)
            vh = sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)
            xs.append(np.stack([vv + s_T, vh]))
        xs = np.stack(xs)
        sc[k_scr] = score_windows(netC, xs, desc=f"CNN-C {scr:+g}dB")
        sr[k_scr] = score_windows(netR, xs, desc=f"CNN-IQ {scr:+g}dB")
        print(f"[d40] trials {scr:+g} dB done", flush=True)
    np.savez(CTR, c=sc, r=sr, guard_maxdiff=maxd, guard_mismatch=mism)
    return sc, sr, maxd, mism


# ---------------------------------------------------------------- analysis
def pd_rows(hits):
    rows = []
    for k_scr, scr in enumerate(SCR_GRID):
        h = hits[k_scr]
        kd = int(h.sum())
        lo, hi = clopper_pearson(kd, len(h), 1 - CI)
        rows.append({"scr_db": float(scr), "pd": kd / len(h), "ci_lo": lo,
                     "ci_hi": hi, "k": kd, "n": int(len(h))})
    return rows


def mcnemar(a_hits, b_hits):
    """Exact McNemar: a, b boolean arrays of paired hits."""
    b01 = int(np.sum(a_hits & ~b_hits))
    b10 = int(np.sum(~a_hits & b_hits))
    n = b01 + b10
    p = float(binomtest(min(b01, b10), n, 0.5).pvalue) if n else 1.0
    return {"n01": b01, "n10": b10, "n_pairs": int(len(a_hits)), "p": p}


def main():
    t00 = time.time()
    netC, netR = load_nets()
    out = {"config": {
        "checkpoints": [CKPT_C.name, CKPT_R.name],
        "recipe": "phase-4 seed-42 sw0 (sampler 141, rng 49, per-trial rng_t); "
                  "d38-validated stream re-guarded here against v1_repro_sw0",
        "calibration_pool": "seed-77 test units, 1500 windows (same units as "
                            "prop_h0_stats.npz)",
        "threshold_rule": "phase-4 thresholds_for (empirical if n*pfa>=10 else "
                          "M3 exponential tail above the 0.995 quantile)"},
        "validation": {}, "thresholds": {}, "realized_pfa": {}, "pd": {},
        "roc": {}, "scr_at_pd50": {}, "mcnemar": {}, "tail": {}, "runtime": {}}

    # stage 1: calibration pool + thresholds
    c_c, c_r = stage_pool77(netC, netR)
    for name, arr in (("CNN-C", c_c), ("CNN-IQ", c_r)):
        out["thresholds"][name] = {}
        for p in PFA_GRID:
            t, m = thresholds_for(arr, p)
            out["thresholds"][name][f"{p:g}"] = {"threshold": t, "method": m}
            print(f"[d40] {name} threshold Pfa={p:g}: {t:.4f} ({m})", flush=True)

    # stage 2: phase-2 pools
    full_c, full_r, keep = stage_phase2(netC, netR)
    h0c, h0r = full_c[keep], full_r[keep]
    print(f"[d40] phase-2 pools: full {len(full_c)}, anmf-4000 subset {len(keep)}",
          flush=True)

    # stage 3: paired trials
    sc, sr, maxd, mism = stage_trials(netC, netR)
    out["validation"] = {
        "guard_proposed_0db_mismatches": mism, "guard_max_absdiff": maxd,
        "reference": "results/json/v1_repro_sw0.json (Proposed, sw0 @0 dB)"}

    # thresholds + Pd rows
    hits = {}
    for name, arr, pool in (("CNN-C", c_c, sc), ("CNN-IQ", c_r, sr)):
        hits[name] = {}
        for p in PFA_GRID:
            thr = out["thresholds"][name][f"{p:g}"]["threshold"]
            hits[name][f"{p:g}"] = pool > thr
            out["pd"].setdefault(name, {})[f"{p:g}"] = pd_rows(hits[name][f"{p:g}"])
        out["scr_at_pd50"][name] = {"1e-3": scr_at_pd50(out["pd"][name]["0.001"]),
                                    "1e-4": scr_at_pd50(out["pd"][name]["0.0001"])}

    # realized Pfa on three pools + ANMF-DL0.001 reference
    extj = json.loads((JS / "anmf_load_ext.json").read_text())
    z = np.load(JS / "anmf_load_ext_progress.npz")
    h0_dl = z["h0_dl0.001"]
    for p in PFA_GRID:
        key = f"{p:g}"
        for name, cal, f4, ff in (("CNN-C", c_c, h0c, full_c), ("CNN-IQ", c_r, h0r, full_r)):
            thr = out["thresholds"][name][key]["threshold"]
            out["realized_pfa"].setdefault(name, {})[key] = {
                "pool77_insample": float(np.mean(cal > thr)),
                "phase2_anmf4000": float(np.mean(f4 > thr)),
                "phase2_full": float(np.mean(ff > thr))}
        thrA = extj["h0_thresholds"][f"dl0.001|{key}"]["threshold"]
        out["realized_pfa"][f"ANMF-DL0.001@{key}"] = {
            "threshold_deployed": thrA,
            "phase2_anmf4000_insample": float(np.mean(h0_dl > thrA)),
            "phase2_full_partial": None}

    # McNemar vs saved arms
    stats_sw0 = z["stats_sw0"]     # (3 modes, 17 SCR, 500): 0=orig, 1=dl0.001, 2=dl0.01
    v1 = json.loads((JS / "v1_repro_sw0.json").read_text())
    for name in ("CNN-C", "CNN-IQ"):
        out["mcnemar"][name] = {"vs_ANMF_DL0.001": {}, "vs_Proposed": {}}
        for k_scr, scr in enumerate(SCR_GRID):
            an = stats_sw0[1, k_scr] > extj["h0_thresholds"]["dl0.001|0.001"]["threshold"]
            out["mcnemar"][name]["vs_ANMF_DL0.001"][f"{scr:g}"] = mcnemar(
                hits[name]["0.001"][k_scr], an)
        # Proposed thresholds (phase-4 recomputed, from prop_h0_stats)
        ph0 = np.load(JS / "prop_h0_stats.npz")["stats"]
        tP = {p: thresholds_for(ph0, p)[0] for p in PFA_GRID}
        for scr in (0, 10):
            k_scr = SCR_GRID.index(scr)
            pr = np.asarray(v1["stats"][f"{scr:g}"]["Proposed"]) > tP[1e-3]
            out["mcnemar"][name]["vs_Proposed"][f"{scr:g}"] = mcnemar(
                hits[name]["0.001"][k_scr], pr)
        out["mcnemar"][name]["vs_Proposed_thresholds"] = {f"{p:g}": t for p, t in tP.items()}

    # ROC / AUC / pAUC per the d31 recipe (H0 = the 4000-unit pool)
    pfa_r = np.logspace(-4, -1.4, 25)
    for name, h0 in (("CNN-C", h0c), ("CNN-IQ", h0r)):
        out["roc"][name] = {}
        for scr in ROC_SCR:
            k_scr = SCR_GRID.index(scr)
            h1 = sc[k_scr] if name == "CNN-C" else sr[k_scr]
            pd_r = [float(np.mean(h1 > np.quantile(h0, 1 - q))) for q in pfa_r]
            out["roc"][name][f"{scr:g}"] = {
                "auc": float(auc_from_scores(h0, h1)),
                "auc_common": mean_tpr_log(pfa_r, pd_r, 1e-4, 1e-2),
                "pauc": mean_tpr_log(pfa_r, pd_r, 1e-4, 1e-3),
                "pfa": pfa_r.tolist(), "pd": pd_r}

    # H0 tail data + figure (survival curves + deployed thresholds)
    out["tail"] = {
        "pool_sizes": {"pool77": 1500, "phase2_anmf4000": int(len(keep))},
        "exceedance_at_1e4": {n: out["realized_pfa"][n]["0.0001"]["phase2_anmf4000"]
                              for n in ("CNN-C", "CNN-IQ")}}
    fig_tail(h0c, h0r, h0_dl, out)

    # power figure (sw0 @1e-3): Proposed + ANMF-DL0.001 from saved rows + CNNs
    fig_power(out)

    # runtime: single-window latency + batch-32 amortized
    xs = np.random.default_rng(0).standard_normal((320, 2, Np)) \
        + 1j * np.random.default_rng(1).standard_normal((320, 2, Np))
    xs = xs.astype(np.complex64)
    for name, net in (("CNN-C", netC), ("CNN-IQ", netR)):
        x1 = torch.from_numpy(normalize(xs[:1]))
        t0 = time.time()
        with torch.no_grad():
            for _ in range(200):
                net(x1)
        single = 1000 * (time.time() - t0) / 200
        x32 = torch.from_numpy(normalize(xs[:32]))
        with torch.no_grad():
            net(x32)  # warmup
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                net(x32)
        batched = 1000 * (time.time() - t0) / 10 / 32
        out["runtime"][name] = {"single_ms": float(single),
                                "batch32_amortized_ms": float(batched)}

    OUTJ.write_text(json.dumps(out, indent=2))
    print(f"[d40] wrote {OUTJ.name} ({time.time()-t00:.0f}s total)", flush=True)


def fig_tail(h0c, h0r, h0_dl, out):
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    data = [("CNN (complex)", h0c, CNN_STYLE["CNN-C"]["color"], "-"),
            ("CNN (I/Q real)", h0r, CNN_STYLE["CNN-IQ"]["color"], "-"),
            ("ANMF (DL $10^{-3}$)", h0_dl, "#0072B2", "-")]
    for lab, s, col, ls in data:
        xs = np.sort(s)
        surv = 1.0 - (np.arange(1, len(xs) + 1) / len(xs))
        ax.loglog(xs, np.maximum(surv, 1 / (2 * len(xs))), color=col, ls=ls, lw=1.1,
                  label=lab)
    extj = json.loads((JS / "anmf_load_ext.json").read_text())
    for lab, col, key in (("CNN (complex)", "#8C564B", "CNN-C"),
                          ("CNN (I/Q real)", "#7F7F7F", "CNN-IQ")):
        t = out["thresholds"][key]["0.0001"]["threshold"]
        ax.axvline(t, color=col, ls=":", lw=0.9)
    tA = extj["h0_thresholds"]["dl0.001|0.0001"]["threshold"]
    ax.axvline(tA, color="#0072B2", ls=":", lw=0.9)
    ax.set_xlabel("detector score (a.u.)")
    ax.set_ylabel("$P(score > s)$")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("H0 score tails (phase-2 4000-unit pool)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cnn_h0tail.pdf")
    fig.savefig(FIG / "fig_cnn_h0tail.png", dpi=200)
    plt.close(fig)


def fig_power(out):
    mc = json.loads((JS / "mc_results.json").read_text())
    d31 = json.loads((JS / "d31_baseline_consolidation.json").read_text())
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    prop = mc["pd"]["sw0"]["Proposed"]["0.001"]
    xs = [r["scr_db"] for r in prop]
    ax.plot(xs, [r["pd"] for r in prop], **STYLE["Proposed"])
    ax.fill_between(xs, [r["ci_lo"] for r in prop], [r["ci_hi"] for r in prop],
                    color=STYLE["Proposed"]["color"], alpha=0.15, lw=0)
    an = d31["power_rows"]["sw0"]["0.001"]["ANMF-DL0.001"]
    xs = [r["scr_db"] for r in an]
    ax.plot(xs, [r["pd"] for r in an], **STYLE["ANMF"])
    ax.fill_between(xs, [r["ci_lo"] for r in an], [r["ci_hi"] for r in an],
                    color=STYLE["ANMF"]["color"], alpha=0.15, lw=0)
    for name in ("CNN-C", "CNN-IQ"):
        rows = out["pd"][name]["0.001"]
        xs = [r["scr_db"] for r in rows]
        ax.plot(xs, [r["pd"] for r in rows], **CNN_STYLE[name])
        ax.fill_between(xs, [r["ci_lo"] for r in rows], [r["ci_hi"] for r in rows],
                        color=CNN_STYLE[name]["color"], alpha=0.15, lw=0)
    ax.set_xlabel("SCR (dB)")
    ax.set_ylabel("$P_D$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Power, Swerling 0, $P_{FA}=10^{-3}$, $n=500$ paired", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cnn_power.pdf")
    fig.savefig(FIG / "fig_cnn_power.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
