# Paper 2 — engineering code

Copied unmodified from `ipix_hybrid_detector/`. Byte-for-byte; hashes in `../SHA256_MANIFEST.txt`.
No script was edited to point at this tree, so what you read here is what produced the results.

## Layout

| Path | Purpose |
|---|---|
| `cnns/d40_train_cnn_baselines.py` | trains CNN-C and CNN-IQ (S1: real clutter + injected Swerling-0) |
| `cnns/d40_cnn_eval.py` | evaluates both CNNs; produces `fig_cnn_h0tail` |
| `battery/d45_shift_battery.py` | the eleven-arm H1 distribution-shift battery |
| `battery/d45b_d40_pfa_fixes.py` | the two D40 realised-Pfa defects and their fixes |
| `battery/d46_r8c_scaled_tercile_pfa.py` | scaled 21,462-unit per-tercile calibration; `fig_r8c_tercile_pfa` |
| `battery/d47_occlusion_dists.py` | tone-occlusion score distributions; `fig_r8c_occlusion` |
| `battery/d48_real_target_invivo.py` | in-vivo real-target study; `fig_realtarget_*` |
| `battery/d49_h0_shift.py` | H0-side shift battery; `fig_h0shift` |
| `battery/d50_chirp_bank.py` | chirp-bank walk-recoverability probe |
| `battery/d51_h0_seed2.py` | second-pool H0 replication |
| `figures/fig_paper2_frontier.py` | the two-sided frontier figure, reads four result JSONs |
| `scripts_shared/anmf_load_sweep.py` | helper imported by `d48`/`d49` |
| `src/` | full implementation package (21 modules) |
| `tests/` | the project's own unit tests |
| `configs/default.yaml` | master config: seed, split, calibration, ICA, training |

## Module dependency of the Paper 2 scripts

```
src.eval.units       UnitSampler, load_split, target_cells_map
src.eval.metrics     clopper_pearson, auc_from_scores
src.eval.plots       STYLE
src.data.synth       swerling_target, estimate_psd, clear_doppler_band
src.data.ipix_loader read_ipix_file, complex_cubes, file_split
src.proposed.infer   HybridDetector
src.proposed.m1_network   ComplexConv1d, ComplexBatchNorm1d, ModReLU, c_to_r
src.proposed.m4_detector  nmf_statistic, prep_sep_from_windows
src.baselines.anmf   steering_bank
```

## Environment the results were produced in

Python 3.12 with `numpy`, `scipy`, `torch` (CPU), `matplotlib`, `tqdm`, `pyyaml`. The checkpoint
format is `torch.save` of a state dict and loads on CPU only (`map_location="cpu"`).

## Two operational notes worth preserving

1. **Thread environment is part of the protocol.** The original runs set `OPENBLAS_NUM_THREADS` before
   importing numpy and deliberately did **not** set `OMP_NUM_THREADS` or `torch.set_num_threads`.
   Changing the BLAS thread count changes reduction order and breaks the `1e-10` replay guards. If you
   re-run anything, keep that convention.
2. **Axis discipline.** The saved `stats_sw*` arrays are `(mode, SCR, trial)`. Each axis has been
   mis-sliced once in this project's history; assert shapes before trusting a guard.

## Running them

`ROOT = Path(__file__).resolve().parents[1]` and `RJ = ROOT / "ipix_hybrid_detector/results/json"`.
To execute inside this package, either reconstruct that layout or patch those two constants. The
scripts also expect `cache/` and the raw CDF directory; see `../UNAVAILABLE_ARTIFACTS.md`.
