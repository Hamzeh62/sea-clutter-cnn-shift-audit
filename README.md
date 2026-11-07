# Sea-Clutter CNNs: Distribution Shift and Coherence–Energy Analysis

**Research artifact for Paper 2 (unpublished working manuscript).**  
**Author:** Hamzeh Ghahramani  
Department of Electrical Engineering, Bandar Khamir Center / Bandar Lengeh Branch, Islamic Azad University, Hormozgan, Iran  
Email: hamze.ghahramani@gmail.com · hamze.ghahramani@iau.ac.ir

**GitHub:** https://github.com/Hamzeh62/sea-clutter-cnn-shift-audit  
**Published software release v1.0.0:** https://doi.org/10.5281/zenodo.24033776  
**Software concept DOI (all versions):** https://doi.org/10.5281/zenodo.24033775  
**CNN checkpoints and archived numerical results:** https://doi.org/10.5281/zenodo.24035348 (separate Dataset; CC BY 4.0). The software DOI does not contain these files.

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24033776.svg)](https://doi.org/10.5281/zenodo.24033776)

## Scope

This repository accompanies the working manuscript *Do Learned Detectors Transfer? A Distribution-Shift Battery and Coherence–Energy Decomposition for Supervised Sea-Clutter Detection Baselines*.
It includes original research scripts, saved per-trial numerical evidence, and original scientific figure PDFs. The two trained CNN checkpoint files are archived separately at https://doi.org/10.5281/zenodo.24035348, not inside the GitHub repository; the unpublished manuscript text is not redistributed here. The principal empirical finding is the **H1/H0 trade-off**: CNNs have strong detection on injected-target tests and show sensitivity to incoherent energy, but false-alarm rates can deteriorate severely after clutter-calibration shift. These are findings from the **historical Paper 2 protocol**, not the later corrected Paper 1 cross-fold study.

## Reproducibility: what works

| Task | Status |
|---|---|
| Check historical artifact hashes, checkpoint identities, 340 claim links, and 902 NPZ array members | **Verified** with `python verify_handoff.py` |
| Reproduce figures 2–5 and 7 from included saved JSON/score data | **Available** via `python reproduce/rebuild_saved_figures.py` (no IPIX data) |
| Original figure 1 H0 tails | Original vector PDF included; CNN-only tail recreation is available, **ANMF raw H0 score array not included** |
| Figure 6 range–Doppler map | Original vector PDF included; **raw IPIX CDF required** to regenerate pixels |
| Inspect per-trial scores and original H0/H1 comparisons | **Available** from `data/` |
| Rerun detector inference, rebuild window cache, train CNNs | **Requires** separately obtained original IPIX recordings and ~887 MiB window cache or reconstruction |

The 3,177 historical handoff checks verify the preserved artifacts and their internal consistency. They do NOT independently validate the physical correctness of the original experimental protocol or demonstrate fresh raw-I/O reproduction.

## Quick start

```bash
python -m pip install -r requirements-figures.txt
python verify_handoff.py
python reproduce/rebuild_saved_figures.py
```

The last command regenerates five publication-figure types from saved numerical artifacts and a **CNN-only** H0-tail diagnostic; regenerated figures go to `reproduced_figures/`. The seven original publication figure PDFs are in `manuscript/figures/`. Do not treat the CNN-only diagnostic as an exact replacement for the original three-detector Figure 1.

The preserved original experimental scripts are under `code/`, **unchanged**, and contain absolute/relative workstation layout assumptions. They do not run end-to-end from this archive without the external cache/raw data and validated path configuration. See `code/README_CODE.md`, `UNAVAILABLE_ARTIFACTS.md`, and `docs/REPRODUCIBILITY_SCOPE.md`.

## Primary contents

- `CHECKPOINTS_LICENSED_ZENODO.md`: hashes and retrieval instructions for `best_cnn_complex.pt` and `best_cnn_real_iq.pt` (original weights archived separately)
- `code/`: original CNN training/evaluation, D45–D51 tests, radar baselines, model code
- `data/`: H1 shift tests, H0 calibration transfer, mechanism probes and real-target numerical evidence
- `provenance/`: 340 source-linked claims, 902 score-array records and frozen thresholds
- `manuscript/figures/`: all 7 original vector figure PDFs; manuscript text is excluded from the public research-code deposit
- `reproduce/`: independent no-rerun figure-generation and manifest checks
- `docs/`: dataset-access restrictions, protocol differences, author-release checklist

## Important scientific cautions

1. Historical target/guard filtering and calibration were subsequently found to have limitations; the historical numerical curves are preserved unchanged, not recertified under the corrected 16-acquisition Paper 1 protocol.
2. Tests at design PFA 10^-4 use extrapolated thresholds; pooled low-PFA rates are **not** acquisition-wise CFAR guarantees.
3. 11 H1 perturbations use a 9 train / 4 validation / 3 held-out file split, not the later Paper 1 8/8 cross-fold design.
4. The networks are **not capacity-matched** (183,265 vs 52,897 parameters).
5. Raw third-party IPIX CDF recordings and the window cache are **not included**; see `docs/DATA_ACCESS_AND_RIGHTS.md`.
6. The related Paper 1 is an unpublished manuscript, and no arXiv or journal DOI is asserted.

## Citation and rights

See `CITATION.cff` for the published research software DOI; the related manuscript remains unpublished. **MIT** applies to original author-owned code (`LICENSE`); **CC BY 4.0** applies to author-owned research outputs, figure artifacts, and separately archived CNN checkpoints (`DATA_LICENSE.md`). Licenses were approved by the author on 2026-11-06. No third-party raw IPIX data are redistributed.

## Verification and issue reporting

`python verify_handoff.py` checks the unmodified original artifact inventory. `python reproduce/verify_release.py` separately checks newly staged files. Compare reproduced figures with original PDFs, allowing plotting-backend differences; no detector data are regenerated in this process. For discrepancies, report script, platform, package versions and source artifact hash.

## Publication and DOI status

The Paper 2 repository is public at https://github.com/Hamzeh62/sea-clutter-cnn-shift-audit. Its GitHub release v1.0.0 is archived under the software version DOI https://doi.org/10.5281/zenodo.24033776 (concept DOI https://doi.org/10.5281/zenodo.24033775). The separate CNN checkpoint and numerical-results dataset is available at https://doi.org/10.5281/zenodo.24035348. Neither archived record includes the third-party raw IPIX recordings. Software source code and datasets/checkpoints have different licenses; the source-repository license indicator `MIT` does not relicense the numerical data or CNN weights. The related Paper 1 software/dataset DOIs refer to a **different** study and are not reused here.
