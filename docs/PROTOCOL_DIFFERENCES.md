# Paper 2 — protocol comparison against the corrected Paper 1 protocol

**Scope.** This document compares the experimental protocol that produced every Paper 2 number with
the protocol of the *corrected* Paper 1 campaign (Phases 5–7). It exists so that an editor can see,
without re-running anything, which Paper 2 evidence is shared with Paper 1, which was generated
independently, and which rests on rules Paper 1 later replaced.

**Reading rule.** "Corrected Paper 1" means the two-fold, 16-acquisition campaign frozen in
`Paper1_Revision_Working/phase5_minimal/configs/campaign.json` and analysed in Phases 6–7. It does
*not* mean the historical Paper 1 results (`mc_results.json` and the 17-point figures), which use
the same protocol family as Paper 2.

---

## 1. Side-by-side

| Dimension | Paper 2 (all results) | Corrected Paper 1 (Phases 5–7) |
|---|---|---|
| Design Pfa | 1e-3 primary, 1e-4 reported | 1e-3 nominal; matched-rate analysis at 5e-3 |
| H1 trials per cell | 500 (paired) | 1,990 (fold 1) / 2,000 (fold 2), paired |
| SCR grid | 17 points, −30 … +10 dB, 2.5 dB step | 9 points, −20 … +10 dB, 2.5 dB step (primary test at 0 dB) |
| Target model | Swerling-0 base + ten explicit perturbations (arms) | Swerling-0, one shared waveform per trial id |
| Target RNG | seed 42, "phase-4 seed-42 stream (sampler 141, rng 49, per-trial rng_t slot)" | `sha256(trial_id) → default_rng`, seeds 20261029 (pool, bootstrap, CSEP) |
| Partition | one flat 60/20/20 file split (seed 42) → 9 train / 4 val / **3 test** acquisitions | two acquisition-level folds; each fold calibrates on 8 acquisitions and tests on the other 8 |
| H0 threshold pool | 1,500 units, one pool, shared by all arms | 8,000 CAL windows per arm per fold, identical units for all eight arms |
| H0 evaluation pool | 21,462-unit full pool (D46), 4,000-unit ANMF-position pool | 4,000 TEST_H0 windows per arm per fold |
| Calibration rule | pooled empirical quantile of the 1,500-unit pool; extrapolation for 1e-4 | pooled quantile **plus** acquisition-stratified, Pareto-tail and hierarchical rules, compared |
| Thresholds for baselines | ANMF-DL: extrapolated deployed values from `anmf_load_ext.json`; CNNs: D40 phase-4 rule, 1e-4 extrapolated | refit per fold on CAL only, one common selection protocol for every arm |
| Test-set independence | TEST is 3 held-out files; the threshold pool overlaps the training partition (pool drawn from the phase-4 H0 superset) | TEST_H0 never touched during calibration; thresholds derived from CAL only |
| Uncertainty | exact 95 % binomial (Clopper–Pearson) per cell; exact McNemar on paired trials | acquisition-clustered bootstrap, Holm across a 24-contrast family, exact McNemar |
| Pfa reporting | nominal vs realized, per arm, per pool | nominal vs pooled-realized vs per-acquisition realized vs leave-one-out transfer |
| Detectors | Proposed (hybrid), CNN-C, CNN-IQ, ANMF-DL | 8 arms: hybrid, ICA-only, cold-start, no-physics, AR(8)+NMF, two-tap notch, ANMF-DL, CFAR |

---

## 2. What is shared

1. **The trial stream of the in-distribution arm.** `d45`'s `indist` arm uses the *same* seed-42
   recipe as the historical Monte Carlo (`v1_repro_sw0`, `anmf_load_ext`, `d40`). Its agreement with
   the historical Paper 1 curves is therefore a **re-execution of the same stream**, not an
   independent replication. No editor should read that agreement as independent confirmation.
2. **Three comparator cells.** `mc_results.json`, `anmf_load_ext.json`, `d31_baseline_consolidation.json`
   and `tab_v6_auc_common.csv` are quoted verbatim by both manuscripts (deliberately: the two papers
   must not disagree about a shared number). These are historical-protocol numbers.
3. **Processing constants.** `Np = 512`, `hop = 256`, `K_secondary = 1024`, PRF 1000 Hz, and the
   16-acquisition cache are common to both papers.
4. **The hybrid's deployed thresholds** (`Proposed` 1e-3 = 0.4677, 1e-4 = 0.5224) come from the
   1,500-unit phase-4 pool — the *same* pool Paper 2 uses. Paper 1's corrected campaign refit these
   per fold on 8,000 CAL windows, so Paper 2's hybrid operating point is not the corrected one.

## 3. What was generated independently

- **All of D45–D51**: the eleven H1 arms, the H0-side shift battery (D49), its seed-2 replication
  (D51), the scaled per-tercile calibration (D46), the occlusion probes (D47), the in-vivo
  real-target study (D48), and the chirp bank (D50). No Paper 1 phase computes any of these.
- **The CNN baselines (D40)**: training, evaluation, thresholds and ROC. Paper 1 contains no CNN arm.
- **The 21,462-unit full H0 pool** used by D46 for realized-Pfa scaling. Paper 1's corrected campaign
  uses 4,000-window TEST_H0 pools per arm and per fold; the numbers are not interchangeable.

## 4. Where Paper 2 rests on rules Paper 1 replaced

This is the material part, stated plainly.

1. **Pooled-quantile thresholds.** Paper 2 sets every operating point from one pooled 1,500-unit
   quantile. Paper 1's Phase 7 V2 evaluated exactly that rule against three alternatives and found
   it transfers poorly: worst pooled realized-Pfa ratio up to 9.75× nominal, worst single acquisition
   up to 46.7×, and the intuitive remedy (per-acquisition stratification) *worse* than the pooled
   rule (20.5× / 74.8×). Paper 2's own D49/D51 batteries measure the same phenomenon and reach a
   consistent conclusion — the two papers agree — but the *operating point* underneath Paper 2's H1
   comparisons is the rule Paper 1 showed does not transfer.
2. **Order-statistic resolution.** In a 1,500-unit pool the 1e-3 quantile is set by roughly the
   second-largest order statistic; realized Pfa near 1e-3 cannot be estimated from that pool at all.
   D46 fixes this for the *measurement* (scaling to 21,462 units, MoM ν̂ per tercile) but the
   *thresholds* themselves remain 1,500-unit quantiles (and, at 1e-4, extrapolations). Paper 1's
   corrected campaign uses 8,000 CAL windows per arm for the same reason.
3. **1e-4 operating points.** Paper 1 explicitly forbids promoting 1e-4 to a confirmed operating
   point. Paper 2 reports 1e-4 columns; `d45b_d40_pfa_fixes.json` already documents that the CNN-IQ
   1e-4 threshold, extrapolated from the 1,500-unit pool, realizes 4.66e-4 on the full pool
   (4.7× nominal), and that re-thresholding at the full-pool quantile gives 1.40e-4. Any 1e-4 claim
   in Paper 2 must carry that label.
4. **Independence of the test draw.** Paper 2's 500 trials are drawn from 3 held-out acquisitions
   (seed-42 split). Corrections that assume independent trials understate variance when the
   acquisition is the real sampling unit; Paper 1's corrected analysis clusters at acquisition level
   for this reason. Paper 2's binomial intervals are *within-cell* intervals and are not
   acquisition-clustered.
5. **The 17-point grid.** Paper 2 keeps the historical 17-point grid. Paper 1's corrected campaign
   uses 9 points and a single primary test cell (0 dB) because the historical grid's *selection*
   history (choosing the operating point after seeing the curve) was one of the defects the revision
   corrected. Paper 2's curves are descriptive; they should not be read as a preregistered
   confirmatory grid.

## 5. What this does *not* imply

- It does not invalidate the eleven-arm comparisons. Every arm shares one trial stream, one target
  RNG and one threshold, and the contrasts are paired, so the *relative* ordering of arms at a common
  operating point is internally consistent. D45's `realized_pfa` block reports the shared operating
  point honestly.
- It does not make Paper 2's H0-shift conclusion circular: D49/D51 measure the *transfer*, which is
  the paper's subject. A threshold that does not transfer is the finding, not a flaw in measuring it.
- The Paper 2 sets that are genuinely at risk are the absolute `Pd` values quoted against a nominal
  Pfa and the claim that any arm is "calibrated at 1e-3" in a deployment sense.
