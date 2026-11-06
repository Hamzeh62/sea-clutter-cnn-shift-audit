"""T1b — Occlusion probe upgrade: score distributions + delta-score histograms.

Replaces the draft's hit-count-only occlusion report ("457-500 of 500") with the
full per-trial distributions the handoff asks for, at SCR {0, 5, 10} dB (the
handoff names {0, +10}; +5 is in the saved artifact and included for
completeness):

  - per-detector base vs tone-occluded score distributions (overlay + exceedance
    counts at the deployed 1e-3 thresholds);
  - per-trial delta scores  Delta = base - occluded  (positive = evidence
    removed by occluding the |f_D| +/-3-bin tone region) with histograms,
    medians, IQRs, sign-test p-values, and delta scaled by each detector's
    base-score IQR (cross-detector comparability);
  - interpretation guard: the artifact notes phrase the finding as
    "substantially driven by non-coherent features" (the measured statement);
    the word "energy detector" is NOT used.

Data: results/json/d45_shift_battery.npz mechanism arrays (no detector re-run):
  mech|occ|hyb|{scr}      (2, 500)  row 0 = base, row 1 = occluded
  mech|occ|cnnC|{scr}     (500,)    base      (paired in-distribution arm windows)
  mech|occ|cnnC_occ|{scr} (500,)    occluded  (same windows, tone region zeroed)
  mech|occ|cnnR|{scr} / cnnR_occ|{scr}  as above for CNN-IQ
Windows: the SAME paired seed-42 in-distribution H1 windows as the headline
grid (sampler 141, rng 49; d38/d40-validated stream).  Thresholds: deployed
d40 CNN thresholds; hybrid = the d45-battery convention (quantile of
prop_h0_stats at 1 - pfa), recorded in the artifact.

Outputs: results/json/d47_occlusion_dists.json
         results/figures/fig_r8c_occlusion.{png,pdf}      (score distributions)
         results/figures/fig_r8c_occlusion_delta.{png,pdf} (delta histograms)
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "results/json"
FIG = ROOT / "results/figures"

SCR_LIST = (0.0, 5.0, 10.0)


def log(m):
    print(f"[d47] {m}", flush=True)


def main():
    t0 = time.time()
    z = np.load(JS / "d45_shift_battery.npz", allow_pickle=True)

    d40j = json.loads((JS / "d40_cnn_baselines.json").read_text())
    thrC = {n: float(d40j["thresholds"][n]["0.001"]["threshold"])
            for n in ("CNN-C", "CNN-IQ")}
    ph0 = np.load(JS / "prop_h0_stats.npz")["stats"]
    thrP = float(np.quantile(ph0, 1 - 0.001))  # d45-battery hybrid convention

    arms = {"CNN-C": ("cnnC", "cnnC_occ", thrC["CNN-C"]),
            "CNN-IQ": ("cnnR", "cnnR_occ", thrC["CNN-IQ"]),
            "Proposed": None}  # hybrid arrays handled separately (2, 500)

    out = {"task": "T1b occlusion score distributions + delta histograms "
                   "(replaces hit-count-only occlusion report)",
           "config": {"thresholds_1e3": {"CNN-C": thrC["CNN-C"],
                                         "CNN-IQ": thrC["CNN-IQ"],
                                         "Proposed": thrP},
                      "threshold_convention":
                          "CNNs: deployed d40 extrapolated thresholds; "
                          "Proposed: quantile of prop_h0_stats at 1-1e-3 "
                          "(the d45-battery convention)",
                      "windows": "paired seed-42 in-distribution arm windows "
                                 "(500; sampler 141, rng 49)",
                      "occlusion": "|f_D| +/-3-bin Doppler region of the VV "
                                   "channel zeroed (d45 mech probe 1)"},
           "per_scr": {}, "notes": []}

    for scr in SCR_LIST:
        blk = {}
        hyb = np.asarray(z[f"mech|occ|hyb|{scr:g}"], float)
        assert hyb.shape == (2, 500), f"hyb@{scr}: {hyb.shape}"
        base, occ = hyb[0], hyb[1]
        arrays = {"CNN-C": (np.asarray(z[f"mech|occ|cnnC|{scr:g}"], float),
                            np.asarray(z[f"mech|occ|cnnC_occ|{scr:g}"], float)),
                  "CNN-IQ": (np.asarray(z[f"mech|occ|cnnR|{scr:g}"], float),
                             np.asarray(z[f"mech|occ|cnnR_occ|{scr:g}"], float)),
                  "Proposed": (base, occ)}
        thr_by = {"CNN-C": thrC["CNN-C"], "CNN-IQ": thrC["CNN-IQ"],
                  "Proposed": thrP}
        for name, (b, o) in arrays.items():
            assert b.shape == o.shape == (500,), f"{name}@{scr}"
            d = b - o
            thr = thr_by[name]
            sign = sps.binomtest(int((d > 0).sum()), 500, 0.5,
                                 alternative="two-sided").pvalue
            blk[name] = {
                "base_mean": float(b.mean()), "base_median": float(np.median(b)),
                "base_iqr": [float(np.percentile(b, 25)),
                             float(np.percentile(b, 75))],
                "occ_mean": float(o.mean()), "occ_median": float(np.median(o)),
                "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
                "delta_iqr": [float(np.percentile(d, 25)),
                              float(np.percentile(d, 75))],
                "delta_over_base_iqr_median":
                    float(np.median(d) / max(np.percentile(b, 75)
                                             - np.percentile(b, 25), 1e-30)),
                "rel_drop_mean": float((b.mean() - o.mean())
                                       / max(abs(b.mean()), 1e-30)),
                "frac_delta_pos": float((d > 0).mean()),
                "sign_test_p": float(sign),
                "hits_base_1e3": int((b > thr).sum()),
                "hits_occ_1e3": int((o > thr).sum()),
                "threshold": thr,
                "base_scores": b.tolist(), "occ_scores": o.tolist(),
                "delta_scores": d.tolist(),
            }
            log(f"scr {scr:+g} {name}: base med {np.median(b):.4g} occ med "
                f"{np.median(o):.4g} | med D {np.median(d):+.3g} "
                f"({blk[name]['delta_over_base_iqr_median']:+.3f} x IQR) | "
                f"hits {blk[name]['hits_base_1e3']} -> "
                f"{blk[name]['hits_occ_1e3']}")
        out["per_scr"][f"{scr:g}"] = blk

    out["notes"] = [
        "Delta = base - occluded; positive means the score DECREASED when the "
        "+/-3-bin tone region around the true f_D was zeroed, i.e. evidence "
        "that was localized in the tone region.",
        "Cross-detector scale: CNN and NMF statistics live on different scales; "
        "delta_over_base_iqr_median is the scale-free comparison.",
        "Phrasing guard (handoff): the interpretation is that CNN evidence is "
        "'substantially driven by non-coherent features' wherever occlusion "
        "leaves the scores (and detections) largely intact; no "
        "'energy detector' wording.",
        "No detector re-run: all arrays are the saved d45_shift_battery.npz "
        "mechanism probes on the headline-grid windows.",
    ]
    (JS / "d47_occlusion_dists.json").write_text(json.dumps(out, indent=1))
    _figs(out)
    log(f"wrote d47_occlusion_dists.json + figures ({time.time() - t0:.0f}s)")


def _figs(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    dets = ("CNN-C", "CNN-IQ", "Proposed")
    labs = {"CNN-C": "CNN-C", "CNN-IQ": "CNN-IQ", "Proposed": "Hybrid (Proposed)"}
    cols = {"base": "#0072B2", "occ": "#D55E00"}
    scrs = ("0", "10")

    # --- score distributions -------------------------------------------------
    fig, axes = plt.subplots(len(scrs), 3, figsize=(7.0, 4.2),
                             sharex="row", squeeze=False)
    for r, scr in enumerate(scrs):
        for c, det in enumerate(dets):
            ax = axes[r][c]
            b = np.array(out["per_scr"][scr][det]["base_scores"])
            o = np.array(out["per_scr"][scr][det]["occ_scores"])
            thr = out["per_scr"][scr][det]["threshold"]
            lo = min(b.min(), o.min()); hi = max(b.max(), o.max())
            grid = np.linspace(lo, hi, 60)
            for arr, key in ((b, "base"), (o, "occ")):
                ax.hist(arr, bins=grid, histtype="step", lw=1.2,
                        color=cols[key], density=True,
                        label=("base" if (r == 0 and c == 0) else
                               ("tone occluded" if (r == 0 and c == 1) else None)))
            ax.axvline(thr, color="k", ls="--", lw=0.8)
            if r == 0:
                ax.set_title(labs[det], fontsize=8.5)
            blk = out["per_scr"][scr][det]
            ax.text(0.03, 0.97,
                    f"$\\Delta$med {blk['delta_median']:+.3g}\n"
                    f"hits {blk['hits_base_1e3']}$\\to${blk['hits_occ_1e3']}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel(f"SCR {float(scr):+g} dB\ndensity")
    axes[0][0].legend(fontsize=6.5, frameon=False, loc="center right")
    fig.suptitle("Tone-region occlusion: score distributions (500 paired trials, "
                 "deployed 1e-3 thresholds dashed)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIG / "fig_r8c_occlusion.png", dpi=200)
    fig.savefig(FIG / "fig_r8c_occlusion.pdf")
    plt.close(fig)

    # --- delta histograms -----------------------------------------------------
    fig, axes = plt.subplots(len(scrs), 3, figsize=(7.0, 4.2),
                             sharex="row", squeeze=False)
    for r, scr in enumerate(scrs):
        for c, det in enumerate(dets):
            ax = axes[r][c]
            d = np.array(out["per_scr"][scr][det]["delta_scores"])
            grid = np.linspace(d.min(), d.max(), 50)
            ax.hist(d, bins=grid, color="#0072B2", alpha=0.75, density=True)
            ax.axvline(0, color="k", lw=0.8)
            ax.axvline(np.median(d), color="#D55E00", lw=1.0)
            if r == 0:
                ax.set_title(labs[det], fontsize=8.5)
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel(f"SCR {float(scr):+g} dB\ndensity")
            ax.set_xlabel("$\\Delta$ score (base $-$ occluded)")
    fig.suptitle("Tone-region occlusion: per-trial $\\Delta$-score histograms "
                 "(orange line: median)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig_r8c_occlusion_delta.png", dpi=200)
    fig.savefig(FIG / "fig_r8c_occlusion_delta.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
