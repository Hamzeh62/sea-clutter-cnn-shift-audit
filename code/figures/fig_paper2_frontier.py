"""Paper2 Figure 1 (frontier map) — built ONLY from saved artifacts.

Outputs (results/figures/):
  fig_paper2_frontier.{png,pdf}  three panels:
    (a) the two-sided frontier: x = region-average Pd retention under the H1
        shift arms {walk, gauss20} (d45 robustness block, per-detector own
        in-distribution baseline); y = flat-calibrated false-alarm inflation
        on the spike tercile relative to the 1e-3 design (d49 seed-141 open
        circles, d51 seed-142 filled diamonds; CNN-IQ has seed-141 only --
        the second pool replicated the headline pair). Log y.
    (b) mechanism probes (d45 mechanism block): noise-surrogate firing at
        0 dB (bars, of 500) and +3 dB power-boost response on a log axis
        (mean CNN score rise vs NMF median |delta| ~ 1.4e-4, 0/500 hits).
    (c) in-vivo adjacent-cell apparent rate vs per-cell MoM nu-hat, 64
        cells pooled across the 8 stare files (d48), CNN-C vs hybrid with
        their Spearman rho annotated.

Every plotted number traces to an artifact already read by
paper2/build_claims.py (d45_shift_battery / d49_h0_shift /
d51_h0_seed2 / d48_real_target_invivo); no re-computation, no
re-thresholding, no new measurements.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "results" / "json"
FIG = ROOT / "results" / "figures"

# d46 palette (consistent across the paper's figures)
COLS = {"CNN-C": "#8C564B", "CNN-IQ": "#7F7F7F",
        "Proposed": "#D55E00", "ANMF-DL0.001": "#0072B2"}
LABS = {"CNN-C": "CNN-C", "CNN-IQ": "CNN-IQ",
        "Proposed": "Hybrid", "ANMF-DL0.001": "Loaded ANMF"}
DETS = ("CNN-C", "CNN-IQ", "Proposed", "ANMF-DL0.001")

d45 = json.loads((JS / "d45_shift_battery.json").read_text())
d49 = json.loads((JS / "d49_h0_shift.json").read_text())
d51 = json.loads((JS / "d51_h0_seed2.json").read_text())
d48 = json.loads((JS / "d48_real_target_invivo.json").read_text())

plt.rcParams.update({"font.size": 8})
fig = plt.figure(figsize=(7.16, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 0.95, 1.0],
                      wspace=0.40, left=0.075, right=0.985,
                      top=0.86, bottom=0.185)

# ---------------- (a) the frontier -----------------------------------------
ax = fig.add_subplot(gs[0, 0])
# x: mean relative Pd retention across the walk and gauss20 arms
ret = {d: 1.0 - (d45["robustness"][d]["walk"]["rel_degradation"]
                 + d45["robustness"][d]["gauss20"]["rel_degradation"]) / 2.0
       for d in DETS}
# y: spike-tercile realized Pfa / 1e-3 design, per seed
Y1 = {d: d49["realized_pfa"][d]["T_spike"]["C_flat@0.001"]["rate"] / 1e-3
      for d in DETS}
Y2 = {d: d51["realized_pfa"][d]["T_spike"]["C_flat@0.001"]["rate"] / 1e-3
      for d in DETS if d in d51["realized_pfa"]}
for d in DETS:
    x = ret[d]
    lab = LABS[d] + (" (pool 141)" if d in Y2 else "")
    # open ring large enough to stay visible when the two seeds coincide
    ax.plot(x, Y1[d], "o", ms=9.5, mfc="none", color=COLS[d], mew=1.7,
            label=lab, zorder=3)
    if d in Y2:
        ax.plot([x, x], [Y1[d], Y2[d]], "-", color=COLS[d], lw=0.8,
                alpha=0.55, zorder=2)
        ax.plot(x, Y2[d], "D", ms=4.5, color=COLS[d], zorder=4,
                label=LABS[d] + " (pool 142)")
# quadrant guides
ax.axvline(0.5, color="k", lw=0.5, ls=":", alpha=0.6)
ax.axhline(10.0, color="k", lw=0.5, ls=":", alpha=0.6)
ax.set_yscale("log")
ax.set_ylim(5, 420)
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("region-avg $P_D$ retention, H1 shift arms\n"
              r"(walk $+\ \sigma_f{=}20$ Hz, d45)")
ax.set_ylabel(r"spike-tercile $P_{FA}$ ($\times$ nominal)")
ax.text(0.985, 0.955, "H1-robust\nH0-fragile", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.3, color="0.35")
ax.text(0.03, 0.045, "H1-fragile\nH0-graceful", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=6.3, color="0.35")
# upper-left quadrant is empty -- keep the legend clear of all markers
ax.legend(fontsize=6.0, loc="upper left", frameon=False, handletextpad=0.4,
          borderaxespad=0.3, labelspacing=0.32)
ax.grid(alpha=0.25, which="both")
ax.set_title("(a) The two-sided frontier", fontsize=8.5)

# ---------------- (b) mechanism probes -------------------------------------
# proper subpanels: top = noise-surrogate bars, bottom = +3 dB dot plot
gsb = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.78)
axb_top = fig.add_subplot(gsb[0])
axb_bot = fig.add_subplot(gsb[1])
ns = d45["mechanism"]["noise_surrogate"]["0db"]
dets3 = ("CNN-C", "CNN-IQ", "Proposed")
ys = list(range(len(dets3)))
hits = [ns["cnnC_hits"], ns["cnnR_hits"], ns["hybrid_hits"]]
axb_top.barh(ys, hits, height=0.55, color=[COLS[d] for d in dets3],
             alpha=0.85)
axb_top.set_yticks(ys)
axb_top.set_yticklabels([LABS[d] for d in dets3], fontsize=6.5)
axb_top.invert_yaxis()
axb_top.set_xlim(0, 570)
axb_top.set_xticks([0, 250, 500])
axb_top.tick_params(labelsize=6.5)
axb_top.set_title("(b) Probes (d45)", fontsize=8.5)
axb_top.set_xlabel("noise-surrogate firing / 500", fontsize=6.8, labelpad=1.5)
for y, v in zip(ys, hits):
    axb_top.text(v + 10, y, str(v), va="center", fontsize=6.3, color="0.25")
pb = d45["mechanism"]["power_boost_3db"]
rows = [("CNN-C", pb["cnnC_score_mean"][1] - pb["cnnC_score_mean"][0]),
        ("CNN-IQ", pb["cnnR_score_mean"][1] - pb["cnnR_score_mean"][0]),
        ("NMF stat", pb["nmf_invariance_median_absdiff"])]
axb_bot.axvline(1e-3, color="k", lw=0.5, ls=":", alpha=0.5)
for y, (_name, v) in zip(ys, rows):
    axb_bot.plot([v], [y], "o", ms=5, color=COLS[dets3[y]] if y < 2
                 else "#0072B2")
    axb_bot.annotate(f"{v:.2g}", (v, y), textcoords="offset points",
                     xytext=(6, 0), va="center", fontsize=6.3, color="0.25")
axb_bot.set_xscale("log")
axb_bot.set_xlim(4e-5, 6)
axb_bot.set_yticks(ys)
axb_bot.set_yticklabels([r[0] for r in rows], fontsize=6.5)
axb_bot.invert_yaxis()
axb_bot.tick_params(labelsize=6.5)
axb_bot.set_xlabel(r"$+3$ dB response (score units, log scale)",
                   fontsize=6.8, labelpad=1.5)
axb_bot.text(0.97, 0.06, "NMF: $0/500$ hits", transform=axb_bot.transAxes,
             ha="right", va="bottom", fontsize=5.8, color="0.25")
axb_bot.grid(alpha=0.25, which="both")

# ---------------- (c) in-vivo vs spikiness ---------------------------------
axc = fig.add_subplot(gs[0, 2])
nus, rates = [], {"CNN-C": [], "Proposed": []}
for _f, fd in d48["per_file"].items():
    for _c, cd in fd["per_cell"].items():
        if cd["role"] == "adjacent":
            nus.append(cd["nu_hat"])
            rates["CNN-C"].append(cd["rates"]["CNN-C"])
            rates["Proposed"].append(cd["rates"]["Proposed"])
for d, mk in (("CNN-C", "o"), ("Proposed", "s")):
    axc.scatter(nus, [r * 100 for r in rates[d]], s=11, marker=mk,
                color=COLS[d], alpha=0.75, label=LABS[d], linewidths=0)
axc.set_xscale("log")
axc.set_xlabel(r"per-cell MoM $\hat\nu$ (log)")
axc.set_ylabel("adjacent-cell firing (%)")
rho_c = d48["nu_correlation"]["CNN-C"]["rho"]
rho_p = d48["nu_correlation"]["Proposed"]["rho"]
axc.text(0.97, 0.95, f"$\\rho$ = {rho_c:+.2f} (CNN-C)\n"
                     f"$\\rho$ = {rho_p:+.2f} (hybrid)",
         transform=axc.transAxes, ha="right", va="top", fontsize=6.5)
axc.legend(fontsize=6.0, loc="center left", frameon=False, handletextpad=0.4)
axc.grid(alpha=0.25, which="both")
axc.set_title("(c) In-vivo: firing vs spikiness", fontsize=8.5)

FIG.mkdir(exist_ok=True, parents=True)
fig.savefig(FIG / "fig_paper2_frontier.png", dpi=200)
fig.savefig(FIG / "fig_paper2_frontier.pdf")
print("wrote fig_paper2_frontier.{png,pdf}")
