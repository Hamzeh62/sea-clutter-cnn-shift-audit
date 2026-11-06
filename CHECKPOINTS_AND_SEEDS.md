# Checkpoints, seeds and training configuration

## 1. The two checkpoints

| File | Model | Parameters | Bytes | SHA-256 |
|---|---|---|---|---|
| `checkpoints/best_cnn_complex.pt` | CNN-C (complex-valued) | 183,265 | 747,417 | `292be8e46eaa766c01ce235c0b014b743735f5e4798c5332a2e05397fda9195a` |
| `checkpoints/best_cnn_real_iq.pt` | CNN-IQ (real I/Q pairs) | 52,897 | 223,705 | `6e2f1512fff2618d6f2a86feceafde1eaf0c95886992bb25ac3ef5fec0e34979` |

These are the **original checkpoint bytes**, copied with `shutil.copy2` and never re-saved through
torch. Both are small enough to sit inside this ZIP, so no separate checkpoint archive is needed.

Architectures, verbatim from `data/learned_baselines/d40_cnn_train.json:config.architecture`:

- **CNN-C** — `ComplexConv1d 2→32 k7 / 32→64 k5 s2 / 64→64 k5 s2 / 64→64 k3 s2`, CBN + ModReLU,
  magnitude mean+std head `128→64→1`.
- **CNN-IQ** — `Conv1d 4→32 k7 / 32→64 k5 s2 / 64→64 k5 s2 / 64→64 k3 s2`, BN + GELU, mean+std head
  `128→64→1`.

## 2. Training run (D40)

From `d40_cnn_train.json:config`:

| Item | Value |
|---|---|
| Protocol | S1 — real clutter + injected Swerling-0, polarimetric geometry (`s_T` in VV only, VH clutter-only), target cells excluded, `P_target = 0.5`, SCR ~ U(−5, 15) dB, `f_D` from the clear-Doppler band (guard 8) |
| Sampler seed | 143 |
| RNG seed | 144 |
| torch seed | 145 |
| Normalisation | per-window RMS, both channels, both nets, train **and** eval |
| Loss / optimiser | `BCEWithLogits`; AdamW lr 1e-3, wd 1e-4, cosine to 1e-5 |
| Budget | 60 epochs cap, 50 steps/epoch, batch 16, ≤ 48,000 window draws, patience 10 |
| Actual | 47 epochs, early-stopped, 796 s wall |
| Best val BCE | CNN-C 0.009872, CNN-IQ 0.007136 |

Note the SCR prior: **U(−5, +15) dB**, i.e. the training law deliberately excludes the lowest cells of
the 17-point evaluation grid. That is the mechanism the shift battery exploits, and it is stated in the
paper.

## 3. Seeds of the evaluation stream

| Purpose | Seed |
|---|---|
| In-distribution / shift-battery trials (D45) | 42 — "phase-4 seed-42 stream: sampler 141, rng 49, per-trial `rng_t` slot convention" |
| CNN threshold pool (D40) | 77 (1,500 units, same units as `prop_h0_stats.npz`) |
| H0-shift battery (D49) | as D45's own draw; see `d49_h0_shift.json:config` |
| Second H0 pool (D51) | seed 2 pool, see `d51_h0_seed2.json:config` |
| Hybrid's phase-4 H0 pool | 1,500 units, quantiles in `data/paper_index/thresholds.json` |

The per-trial target draw is **deterministic in the trial id**, not in a stream position, so a trial
number identifies the same clutter window and the same target across every arm and every detector.
That is what makes the paired contrasts valid.

## 4. Acquisition partition

`provenance/split_seed42.json` holds the exact 16-acquisition partition produced by
`src.data.ipix_loader.file_split` at seed 42 (campaign-stratified, odd file to val):

- **train (9)** — 19931107_135603, 19931108_213827, 19931109_202217, 19931110_001635,
  19931118_023604, 19931118_035737, 19931118_162155, 19931118_174259, 19980204_163113_ANTSTEP
- **val (4)** — 19931107_145028, 19931108_220902, 19931109_191449, 19980205_171437_ANTSTEP
- **test (3)** — 19931107_141630, 19931111_163625, 19931118_162658

This was reproduced from cache **stem names only** (no cube data read), and it reproduces the split the
batteries actually used.

## 5. Threshold rule

`d40_cnn_baselines.json:config` states the rule: *phase-4 `thresholds_for` — empirical if
n·Pfa ≥ 10, else an M3 exponential tail fit above the 0.995 quantile*. Consequences at 10⁻⁴ are
documented in `data/h1_shift_battery/d45b_d40_pfa_fixes.json` and summarised in
`CLAIMS_AFFECTED_BY_CORRECTED_P1.md` §3. The frozen deployed values for every arm are tabulated in
`provenance/thresholds_frozen.csv` (32 rows, with method, pool size and source artifact).
