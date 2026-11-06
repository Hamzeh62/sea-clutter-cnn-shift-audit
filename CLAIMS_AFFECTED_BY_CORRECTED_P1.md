# Paper 2 — manuscript claims whose interpretation is affected by the corrected Paper 1 protocol

**Basis.** All 340 claim ids in `paper2/claims.json` were enumerated and each id was resolved to a
source artifact. The 16 distinct sources are all present in this package; the full mapping (claim id →
group → artifact → path in package) is `provenance/claim_sources.csv`. This document lists the
**subset whose *interpretation* changes** once the corrected Paper 1 protocol (Phases 5–7) is treated
as the reference. No number in Paper 2 is altered, and nothing here asks for a result to be
suppressed; the required action is wording, labelling, or an added caveat.

Counts by group: `shift` 93 · `h0` 53 · `cnn` 35 · `r6` 30 · `r8b` 28 · `r8c` 26 · `s2` 19 ·
`occ1` 17 · `mech` 16 · `prop` 4 · `anmfDL` 4 · `tercpfa` 4 · `nb3` 3 · `aucL` 2 · `rt` 2 ·
`aucfull`/`aucU`/`tab`/`occ` 1 each.

Status legend: **A** = consistent with corrected Paper 1, keep (optionally cite the agreement);
**B** = needs qualification (the number stands, the sentence over-reaches); **C** = needs an explicit
label to stay defensible; **D** = provenance must be re-checked before submission.

---

## 1. The abstract's calibration clause — **B** (highest priority)

> "two small CNNs dominate in distribution … **and calibrate**: realized pfa of 9.3×10⁻⁴ and
> 2.2×10⁻³ at the 10⁻³ design on 21,462 independent clutter windows."

Claims: `cnn.realized.1e3.full.c`, `cnn.realized.1e3.full.r`, `cnn.pd.1e3.0.c`,
`cnn.pd.1e3.0.r`, `prop.sw0.1e3.0db`.

*Why affected:* this is a **pooled** calibration statement over 21,462 windows. Corrected Paper 1
(Phase 7 V2) measured exactly this class of pooled rule across 16 acquisitions and found: pooled
realized rate up to 9.75× nominal, **worst single acquisition up to 46.7× nominal**, 6 acquisitions
above 2× for the best of four rules, and no rule (pooled, stratified, Pareto-tail, hierarchical)
achieving exact calibration. A pooled rate of 9.3×10⁻⁴ does not imply per-acquisition control at
10⁻³; the corrected Paper 1 evidence says the opposite is likely.

*Required action:* keep the numbers, replace "calibrate" with an explicitly pooled claim
(e.g. "calibrate on a pooled 21,462-window pool; per-acquisition control is not established here"),
and cross-reference the paper's own tercile result (152× nominal) so the abstract does not read as a
general calibration endorsement.

## 2. "Answered against the expectation" — **B**

> "The calibration clause of the conflation thesis is thus answered **against** the expectation …
> it is still controllable at these design levels."

Claims: same block as §1 plus `cnn.realized.1e4.full.*`.

*Why affected:* the sentence generalises from one pooled pool at two design levels. Paper 1's
corrected finding is that controllability is a property of the *pool*, not of the detector: the same
detector/rule pair spans 1× to 47× nominal depending on which acquisition supplies the tail. The
paper's own H0-shift section already contradicts a general reading, so the abstract and §IV are in
tension with §V.

*Required action:* scope to "controllable on a pooled pool of this size", and state that
acquisition-level transfer is treated in the H0-shift section, not certified here.

## 3. Every 10⁻⁴ column — **C**

Rows: `cnn.pd.1e4.0.c`, `cnn.pd.1e4.0.r`, `prop.sw0.1e4.0db`, `anmfDL.sw0.1e4.*`,
`cnn.realized.1e4.full.c`, `cnn.realized.1e4.full.r`, `r8c.spike.1e4.*`, and the right-hand Pd column
of `tab:cnn` (0.812 / 0.634 / 0.082 / 0.410).

*Why affected:* these thresholds are **extrapolations from a 1,500-unit pool**. `d45b_d40_pfa_fixes.json`
records the consequence for CNN-IQ: deployed extrapolated 3.2937 → realized 4.66×10⁻⁴ on the full pool
(4.7× nominal); re-thresholding at the full-pool quantile (3.9176) gives 1.40×10⁻⁴. So at 10⁻⁴ the two
"10⁻⁴" arms are not operating at the same realized rate, and one of them is 4.7× off design.
Corrected Paper 1 explicitly refuses to promote 10⁻⁴ to a confirmed operating point.

*Required action:* label the 10⁻⁴ column as the *extrapolated-threshold regime*, print the realized
rate beside each Pd (already available), and drop any sentence that ranks detectors at 10⁻⁴ as if the
rates were matched. The figure caption at `fig:cnntail` already says "against the 10⁻⁴ design level" —
make it "4.7× above the 10⁻⁴ design level" for CNN-IQ.

## 4. H1 battery contrasts — **A with a clustering caveat**

`shift.*` (93 claims), `mech.*` (16), `occ1.*` (17), `tercpfa.*` (4), `tab.cnn.all`.

*Status:* the eleven arms share one trial stream, one target RNG and one threshold, and the contrasts
are paired, so the **ordering** is internally consistent and is not disturbed by Paper 1's
corrections. Paper 1 has no equivalent battery.

*Caveat to add:* the 500 trials are drawn from 3 held-out acquisitions (seed-42 split: 9 train / 4 val
/ 3 test). The reported 95 % intervals are within-cell binomial intervals and are **not
acquisition-clustered**; corrected Paper 1 clusters at acquisition level precisely because
acquisition is the dominant variance component. One sentence in the statistics section should say so.

## 5. H0-shift and second-pool results — **A** (these are the paper's strongest link to Paper 1)

`h0.*` (53), `s2.*` (19), `r8c.*` (26).

*Status:* consistent with, and independently corroborating, corrected Paper 1 Phase 7 V2: pooled
calibration does not transfer across sea state; the inflation is spikiness-dependent; the hybrid
degrades less than the CNN. Nothing to soften.

*Opportunity:* Paper 1 currently reports the transfer failure for four generic rules; Paper 2 reports
it for learned detectors with a spikiness proxy. A single cross-reference makes both papers stronger
and removes the appearance of two unrelated calibration sections.

## 6. Hybrid as the "reference coherent-only instrument" — **B / D**

`prop.*` (4), `anmfDL.*` (4), and the framing sentence "0/500 firing on non-coherent surrogates;
scale-invariant statistic".

*Why affected:* two separate issues.
1. The `prop.*` comparator cells (`prop.sw0.1e3.0db` = 0.480, `prop.sw0.1e4.0db` = 0.082) come from
   `mc_results.json` — the **historical** protocol, with the hybrid's operating point set from the
   1,500-unit phase-4 pool. Corrected Paper 1 refits the hybrid per fold on 8,000 CAL windows. The two
   numbers are not the same operating point, and Paper 2 must not imply it has re-derived them.
2. Paper 1 Phases 6–7 measured the hybrid's dependence on the cross-polarised channel directly. On the
   identical paired trials the hybrid's Pd at a fixed operating point is **0.5005 with φ = 0 and 0.5106
   with random φ** (score correlation 0.663) — i.e. the hybrid is *largely* but **not exactly**
   independent of VH content. "Coherent-only instrument" and "target-model-agnostic" are therefore
   slight over-statements; "reference-coherent and near-invariant to the cross-polarised target model"
   is what the evidence supports.

*Required action:* qualify the framing; state that the comparator is the companion study's deployed
threshold, not a re-calibration.

## 7. Real-target in-vivo rates — **C**

`r6.*` (30), including `r6.corr.c` (ρ = −0.47) and the per-cell rate figure.

*Why affected:* the rates use **deployed 10⁻³ thresholds**, which Paper 1 established are nominal only
(realized 0.68×–74.8× depending on acquisition/rule). The paper already calls the rates "apparent",
which is the right word; the caveat should name the threshold-transfer measurement as its basis.

*Required action:* one clause in the caption and one in the text: the floor is the *deployed* threshold,
whose realized rate is not 10⁻³ on every acquisition.

## 8. NB3 recovery gate — **D**

`nb3.q1.delta`, `nb3.q1.lo`, `nb3.q1.hi` (source `nb3_stage_b_verdict.json`, included with its
original seal).

*Why affected:* the Δ is a Pd difference at a fixed operating point; if that point is a 1,500-unit-pool
quantile, the contrast inherits the same transfer caveat as §1. The artifact states its own thresholds,
but the verdict file must be read before the claim is quoted.

*Required action:* confirmation only — read `data/supplementary/nb3_stage_b_verdict.json` and state
which threshold rule and pool the recovery gate used.

## 9. D40 published Pfa rows — **D (do not reintroduce the old rows)**

`d45b_d40_pfa_fixes.json` records that the originally published `phase2_anmf4000` realized-Pfa rows
indexed the full-pool arrays by unit index instead of pool position: published 0.0015 / 0.00275,
corrected 0.0020 / 0.00225. The current draft quotes the `phase2_full` block
(9.3×10⁻⁴ / 2.19×10⁻³), which is correct and unaffected. Any editing that reaches for a
"4,000-position" row must use the corrected values and say so.

## 10. Claims that are **not** affected

- `aucL.*`, `aucfull.*`, `aucU.*`, `tab.*` AUC/pAUC cells: computed on the ROC itself; independent of
  threshold placement. (Common-range pAUC is the conservative version and is already reported.)
- `rt.*` (runtime): implementation only.
- `r8b.*` (chirp bank, 28 claims): an independent probe of walk recoverability. Its *gates* are
  threshold-dependent, so the same "extrapolated-threshold" label as §3 applies to any gate quoted at
  10⁻⁴.

---

## Summary of required edits

| # | Location | Action |
|---|---|---|
| 1 | Abstract | replace "calibrate" with a pooled-only statement; add per-acquisition caveat |
| 2 | §IV closing paragraph | scope "answered against the expectation" to the pooled pool |
| 3 | every 10⁻⁴ cell + `fig:cnntail` caption | label extrapolated-threshold regime; print realized rate |
| 4 | statistics section | state that intervals are not acquisition-clustered |
| 5 | H0-shift section | add cross-reference to corrected Paper 1 calibration transfer |
| 6 | intro + comparator table | qualify "coherent-only" / "target-model-agnostic"; name the threshold source |
| 7 | real-target captions | name the threshold-transfer basis of the floor |
| 8 | NB3 sentence | verify and state the threshold rule used by the verdict |
| 9 | any 4,000-position Pfa row | use the d45b-corrected values only |
