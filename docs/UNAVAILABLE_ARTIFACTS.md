# Unavailable, non-reproducible and deliberately excluded artifacts

Everything excluded here is excluded by **size**, by **licensing/redistribution**, or because it is
**derived** from something absent. None of it was needed to produce a number quoted in the manuscript.

## 1. Genuinely unavailable from this package

| Artifact | Where it lives | Size | Why it matters | Consequence |
|---|---|---|---|---|
| Raw IPIX CDF files | `K:\All_14p2_Files` (external drive) | GB scale | source of every cube | no score can be re-derived from raw data here |
| Window cache `cache/*_cubes.npz` | `ipix_hybrid_detector/cache` | 887 MiB, 16 files | the unit pool every detector scores | **the detectors cannot be re-run from this package** |
| Chirp-bank cache `d50_chirp_bank_cache.npz` | `results/json` | 44.6 MiB | intermediate for D50 | D50's saved JSON is included; the cache is regenerable only with the window cache |

**What this means, stated without hedging:** this package makes the **numerical evidence inspectable
and re-checkable against the saved artifacts**, and it makes every figure and table re-plottable. It
does **not** make the paper independently reproducible end-to-end, because the dilution-free path
(raw IPIX → cache → scores) is not inside it. Any statement to the contrary would be false.

## 2. Retraining is not reproducible, by design

Even with the checkpoints, seeds and code present, retraining is not reproducible here: the training
run drew 48,000 windows from the cache and the raw files are not shipped. The **checkpoints are the
artifacts of record** (byte-identical originals, hashes in `SHA256_MANIFEST.txt`). Do not represent a
retraining attempt from this package as a reproduction.

## 3. Deliberately excluded (already present elsewhere)

- **Manuscript, LaTeX source, bibliography, compiled figures, claim registry, pre-submission audit** —
  all in `paper2.rar`. Not duplicated here because their contents did not change.
- **`paper2/claims.json`** — not duplicated; its content is re-expressed here as
  `provenance/claim_sources.csv` (the claim → artifact → package-path map), which is new information.
- **Paper 1 archives and the Phase 5–9 workspaces** — untouched, not shipped.
- **`d2c_*`, `d3*`, `d32/d33/d34` progress arrays, `cal_stats_*`, `csep_prep.npz`, `d39_m4_bridge*`** —
  earlier-phase calibration outputs of the companion study. Included only where a Paper 2 claim cites
  them; none of the 340 claims cites them.
- **`nb1_*`, `nb2_*`, `nb3_dev`, `nb3_stage_b.json`** — 0.3–0.9 MiB each. The **verdict** is included
  (`nb3_stage_b_verdict.json` + its seal, the exact file the manuscript quotes); its parent arrays are
  not, since no claim reads them.

## 4. Artifacts that exist only as aggregates

- **Per-iteration ICA convergence trajectories** — do not exist for Paper 2 (Paper 2 trains no ICA
  model; it uses the companion study's hybrid). Not applicable rather than missing.
- **Per-CPI real-target score cubes** — `d48_progress.npz` carries them (included). The **map pixels**
  of `fig_realtarget_rdmaps` are FFT magnitudes of the raw CDF and are *not* stored; the figure can be
  re-plotted only with the raw file. Its per-cell rates (`fig_realtarget_cellrates`) *are* fully
  reproducible from the included JSON.
- **D50 gate values** — the JSON holds the gate outcomes, not the underlying trial arrays beyond the
  reported `pd`/`mcnemar` blocks. Marked `partial` in `data/figure_sources.csv`.

## 5. Runnability caveat for the scripts

The scripts resolve their repository root as `Path(__file__).parents[1]` and read from
`ipix_hybrid_detector/results/json`. Inside this package they therefore run **only** if you place the
package where that layout holds, or patch the `ROOT`/`RJ` constants (two lines per script). This is a
deliberate consequence of copying the originals unmodified rather than editing them to point at a new
tree — the code you inspect here is the code that ran.
