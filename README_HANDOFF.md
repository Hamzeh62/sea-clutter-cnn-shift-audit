> **Historical private-handoff description:** this document predates the 2026-11-06 code/data license approval and the separation of checkpoint binaries for Zenodo. The operative licenses and current release scope are `LICENSE`, `DATA_LICENSE.md`, `PUBLICATION_RIGHTS_AND_SCOPE.md`, and `README.md`.

# Paper 2 — editing and reproducibility handoff

**What this is.** The complete numerical evidence, source code, trained checkpoints and provenance for
the Paper 2 companion benchmark-audit study, packaged so that an editor can (a) re-check every number
quoted in the manuscript against its artifact, (b) re-plot all seven figures, (c) see exactly which
claims the corrected Paper 1 protocol affects, and (d) load the two CNN models. It is **not** a
re-issuance of the manuscript: `paper2.rar` already holds the LaTeX source, bibliography, compiled
figures, claim registry and pre-submission audit, and none of that is duplicated here.

**Verified state.** `verify_handoff.py` runs 3,177 checks and passes with 0 failures on a clean
extraction of this ZIP: 81 files hash-identical to the manifest (68 copied artifacts + 13 authored
documents), and of those 68 artifacts an independent script confirmed **byte-for-byte identity with
the workspace originals**. 22 JSON files parse, **902** array members match a recorded
`(key, dtype, shape, n_valid, n_nan, sha256-of-values)` inventory, all **340** manuscript claims
resolve to a present artifact, all 7 figure sources resolve, and both checkpoints load with 32
tensors each. The ZIP is 4.10 MiB.

**Honest scope limit.** This package does **not** make Paper 2 end-to-end reproducible. The raw IPIX
CDF files and the 887 MiB window cache are not inside it, so no detector can be re-run and the CNNs
cannot be retrained. What *is* fully reproducible here: every reported statistic from the saved scores,
every figure except the range–Doppler map pixels, and every claim-to-artifact link. See
`UNAVAILABLE_ARTIFACTS.md` before writing any reproducibility statement.

---

## How to verify (commands actually run)

```bash
cd PAPER2_EDITING_AND_REPRODUCIBILITY_HANDOFF
python verify_handoff.py        # 3,177 checks, no network, ~20 s
```

Expected final line: `PASS — 3177 checks, 0 failures`. The script needs only `numpy`; `torch` is used
for the checkpoint check and is skipped cleanly if absent.

Individual artifacts can be inspected directly:

```python
import numpy as np, json
z = np.load("data/h1_shift_battery/d45_shift_battery.npz")
a = z["indist|CNN-C|0"]        # 500 paired per-trial scores at 0 dB
z.close()
d = json.load(open("data/h1_shift_battery/d45_shift_battery.json"))
d["pd"]["indist"]["CNN-C"]["0"]   # {'pd': .., 'ci_lo': .., 'ci_hi': .., 'k': .., 'n': 500}
```

## Contents

| Path | What it holds |
|---|---|
| `SHA256_MANIFEST.txt` | one line per file: `<sha256>  <path>` |
| `PACKAGE_CONTENTS.csv` | path, category, bytes, sha256, scientific role |
| `PROTOCOL_COMPARISON_P1_vs_P2.md` | Paper 2 protocol vs the corrected Paper 1 protocol: shared, independent, superseded |
| `CLAIMS_AFFECTED_BY_CORRECTED_P1.md` | the claims whose *interpretation* changes, with the required wording action |
| `CHECKPOINTS_AND_SEEDS.md` | both checkpoints, training run, every seed, the acquisition split, the threshold rule |
| `UNAVAILABLE_ARTIFACTS.md` | what is absent, why, and what may not be claimed because of it |
| `code/` | CNN training/eval, the D45–D51 batteries, figure script, `src/` package, tests, master config |
| `checkpoints/` | `best_cnn_complex.pt` (CNN-C), `best_cnn_real_iq.pt` (CNN-IQ) — original bytes |
| `data/h1_shift_battery/` | the eleven-arm H1 battery (JSON + 778 score arrays) and the D40 Pfa fixes |
| `data/h0_calibration_transfer/` | D49 H0-shift battery, D51 second-pool replication |
| `data/coherence_energy/` | D46 scaled per-tercile calibration, D47 occlusion distributions |
| `data/real_target/` | D48 in-vivo study, per-CPI arrays, D43 audit |
| `data/learned_baselines/` | D40 JSONs and the three score pools |
| `data/shared_with_paper1/` | comparator cells quoted identically by both papers |
| `data/supplementary/` | chirp bank (D50), sealed NB3 stage-B verdict |
| `data/paper_index/` | thresholds, the 1,500-unit hybrid H0 pool, replay guard, target cells, file metadata |
| `data/figure_sources.csv` | all seven figures → script → exact source fields |
| `provenance/score_arrays.csv` | all 902 array members with `n_valid` / `n_nan` / value hash |
| `provenance/claim_sources.csv` | all 340 claims → group → artifact → path in this package |
| `provenance/thresholds_frozen.csv` | every frozen threshold, its method, pool size and source |
| `provenance/split_seed42.json` | the exact 16-acquisition train/val/test partition |

## Three things to know before touching the numbers

1. **The `spike` and `flat` arms are masked subsets.** Their arrays are length 500 with finite values
   only at tercile positions and `NaN` elsewhere — 100 and 84 scored trials respectively, which is the
   `n` in the `pd` block. The masked entries are not a contiguous prefix. Aggregate with
   `np.nanmean`/`np.isfinite`, never by position.
2. **The 10⁻⁴ columns are the extrapolated-threshold regime.** `d45b_d40_pfa_fixes.json` records that
   CNN-IQ's extrapolated 10⁻⁴ threshold realises 4.66×10⁻⁴ on the full pool (4.7× nominal).
3. **The comparator cells are the companion study's deployed thresholds**, not a re-calibration under
   the corrected Paper 1 protocol. `CLAIMS_AFFECTED_BY_CORRECTED_P1.md` §6 details the consequence.

## Provenance chain

`phase0_metadata.json` (file inventory) → `cache/*_cubes.npz` (window cubes, not shipped) →
`d40`/`d45`/`d46`–`d51` artifacts (shipped) → `paper2/build_claims.py` (in `paper2.rar`) →
`claims.json` (340 claims) → `draft.tex`. Every link except the window cache is present or mapped in
this package; `provenance/claim_sources.csv` is the machine-readable form of the last two links.

## What has *not* been done, deliberately

No model was retrained, no experiment was re-run, no result was recomputed, no manuscript file was
modified, and no historical artifact was overwritten. Every file here is a byte-for-byte copy of the
file that produced the published numbers.
