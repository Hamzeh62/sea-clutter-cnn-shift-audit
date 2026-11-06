# Paper 2 — numerical evidence

All files are **original bytes** copied with `shutil.copy2`. Float64 arrays were not downcast, JSON was
not re-serialised, thresholds were not rounded. `provenance/score_arrays.csv` lists all **902** array
members across the 8 `.npz` files with dtype, shape and a SHA-256 of the exact value bytes.

## Contents

### `h1_shift_battery/` — the eleven-arm distribution-shift battery (D45)
- `d45_shift_battery.json` — `config` (17-point SCR grid, `n_mc = 500`, seed 42, the arm definitions,
  the four frozen thresholds and their provenance), `validation` (seven in-distribution replay guards),
  `pd` (11 arms × 4 detectors), `robustness` (region-average relative degradation), `realized_pfa`,
  `mechanism` (noise surrogate + 3 dB power boost), `mcnemar`, `d40_fixes`, `runtime_s`.
- `d45_shift_battery.npz` — 778 per-trial score arrays, keyed `arm|detector|scr_db`
  (68 keys per arm = 17 SCR × 4 detectors; plus `mech` 24 and `tercpfa` 6).
  Arms: `indist`, `sw1`, `sw2`, `fullband`, `walk`, `gauss5`, `gauss10`, `gauss20`, `logn`, `spike`,
  `flat`. Detectors: `Proposed`, `CNN-C`, `CNN-IQ`, `ANMF`.
  **Masked arms:** `spike` and `flat` score only the units that belong to their sea-state tercile.
  Their arrays are length 500, with a finite value **at the tercile positions** and `NaN` everywhere
  else, so the scored-trial count is 100 (`spike`) and 84 (`flat`) — which is exactly the `n` recorded
  in the `pd` block. The mask is not a prefix and not contiguous: `a[0]` can be `NaN` while `a[1]` is
  finite. **Aggregate with `np.nanmean`, or slice by `np.isfinite`, never by position.**
  `provenance/score_arrays.csv` gives `n_valid`/`n_nan` for every member, and `verify_handoff.py`
  re-checks both against the documented subset sizes.
- `d45b_d40_pfa_fixes.json` — the two D40 realised-Pfa defects (index misalignment in the 4,000-position
  block; the CNN-IQ 1e-4 extrapolation) and the corrected values.

### `h0_calibration_transfer/`
- `d49_h0_shift.json` + `d49_progress.npz` — H0-side shift: thresholds applied to shifted clutter pools,
  realized Pfa with exact intervals, per detector and pool.
- `d51_h0_seed2.json` + `d51_progress.npz` — the same measurement on an independent second pool.

### `coherence_energy/`
- `d46_r8c_scaled_tercile_pfa.json` — scaled per-tercile calibration on the full 21,462-unit pool:
  moment-matched ν̂, tercile split, exact binomial intervals, both design levels.
- `d47_occlusion_dists.json` — tone-occlusion probe: base vs. occluded score distributions and
  Δ-score histograms at SCR {0, +10} dB, per detector, with each arm's own threshold.

### `real_target/`
- `d48_real_target_invivo.json` — per-file, per-cell detector outputs on the literature-annotated 1993
  cells and their adjacent cells, per-cell rates, ν̂ correlation, Doppler moments.
- `d48_progress.npz` — the per-CPI arrays behind them.
- `d43_real_target.json` — the earlier real-target audit.

### `learned_baselines/` — CNN-C / CNN-IQ (D40)
- `d40_cnn_baselines.json` — pd (with Clopper–Pearson intervals), roc, thresholds, realized_pfa,
  McNemar, H0 tail block, runtime.
- `d40_cnn_train.json` — training config, 47-epoch history, final metrics.
- `d40_scores_pool77.npz` — CNN scores on the seed-77 1,500-unit threshold pool (`c_c`, `c_r`).
- `d40_scores_phase2.npz` — CNN scores on the full 21,462-unit pool plus the 4,000-position index.
- `d40_scores_trials.npz` — per-trial scores `(17, 500)` plus the replay-guard scalars.

### `shared_with_paper1/`
Historical-protocol comparator cells quoted verbatim by both papers, so the two manuscripts cannot
disagree about a shared number: `mc_results.json` (hybrid Monte Carlo), `anmf_load_ext.json` (loaded
ANMF), `d31_baseline_consolidation.json` (AUC / common-range pAUC), `d35_runtime.json`,
`d36_runtime_decomp.json`, `tab_v6_auc_common.csv`.

### `supplementary/`
- `d50_chirp_bank.json` — chirp-bank walk recoverability (its 44.6 MiB cache is not shipped).
- `nb3_stage_b_verdict.json` + `.seal.txt` — the sealed NB3 stage-B adjudication the manuscript quotes.

### `paper_index/`
- `thresholds.json` — frozen thresholds and the pool sizes behind them.
- `prop_h0_stats.npz` — the 1,500 hybrid H0 statistics that set the comparator's operating point.
- `v1_repro_sw0.json` — the replay-guard reference for the D40 validation block.
- `target_cells.json` — per-file power z-scores defining target cells and the ±1 guard.
- `phase0_metadata.json` — the IPIX file inventory and literature annotations.

## Two reading warnings

1. **`thresholds.json` carries two ANMF values for the same nominal Pfa** — 0.8721 in the `thresholds`
   block and 0.8027 in the `pfa_calibration` block (different estimation paths). Paper 2's ANMF
   comparator uses neither: it uses the deployed extrapolated values in the D45 config
   (0.6567 at 1e-3, 0.8212 at 1e-4). `provenance/thresholds_frozen.csv` prints all of them side by side
   with their source, so nothing is silently resolved.
2. **`d40_scores_phase2.npz`'s 4,000-position block** is the one that was mislabelled; read
   `d45b_d40_pfa_fixes.json` before quoting any row derived from `anmf4000_idx`.
3. **The tercile arms report on 100 and 84 trials, not 500.** Their confidence intervals are therefore
   about 2.4× and 2.9× wider than a 500-trial cell at the same rate. Any table that prints `spike` or
   `flat` beside a 500-trial arm without saying so is comparing unequal precision.
