# Paper 2 — editorial handoff verification and revision status

## Completed in this session
- Read `PAPER2_EDITING_AND_REPRODUCIBILITY_HANDOFF.zip` (83 archive entries) and ran the supplied verification script: **3177 checks, zero failures**.
- Verified the two trained CNN checkpoint objects load (32 tensor objects each).
- Confirmed 902 NPZ array members, 340 evidence claims, and 7 mapped figures in the handoff.
- Extracted the original source from `paper2.rar` using libarchive, without modifying it.
- Created an editorial revision preserving the seven original vector figures and all main numerical tables.
- Replaced missing citation placeholders with verified references: Trabelsi et al. (2018), Ovadia et al. (2019), and Geirhos et al. (2020).
- Corrected the description of differently sized CNN architectures (183,265 vs 52,897 parameters), the draft companion citation, the author contact metadata, and pooled-vs-transfer false-alarm claims.
- Documented that Paper 2 was generated with the historical 9/4/3 split, not the later Paper 1 8/8 cross-fold design, and noted the historical target/guard exclusion limitation. Original result numbers remain unchanged.
- LaTeX compiled successfully to an eight-page IEEE two-column WORKING DRAFT with all 7 original figures; no overfull box or unresolved references.
- Ran the original `verify_claims.py` on this revised source and retained claim registry: markers and numeric coverage PASS; editorial lint clean.

## Present but not end-to-end reproducible here
- CNN source/checkpoints, numerical H1/H0 shift scores, calibration transfer, probes, real-target per-CPI arrays, claims and most figure inputs.
- Missing from compact handoff: raw third-party IPIX files, 887 MiB processed window cache; therefore no exact raw-to-score re-execution/training demonstrated.
- Range-Doppler map pixels require the external raw data, although the already generated original figure is preserved in the working manuscript.

## Critical qualifiers for the editorial interpretation
1. Pooled CNN PFA about 9.3e-4 and 2.2e-3 at nominal 1e-3 is **not** acquisition-wise control.
2. Target/guard exclusions in the historical sampler were later found unreliable for some cached file identifiers; original figures are retained as historical-protocol figures, not recertified under repaired splits.
3. The nominal 1e-4 columns are tail-extrapolated and do not imply equal independently realized false-alarm rates.
4. The main H1 shift comparisons use three held-out acquisitions; binomial intervals do not provide cluster-level generalization guarantees.
5. The two CNNs differ in parameter count and are not capacity matched.
6. Paper 1 remains an unpublished related manuscript; no invented arXiv ID, acceptance status or journal citation.
7. AI-use disclosures, license choices, and public repository/Zenodo links must reflect the actual author-approved status before the final journal upload.

## Status
`PAPER2_EDITORIAL_WORKING_DRAFT_COMPLETE` — not yet journal-template final or journal submission ready. No simulations or training were run.
