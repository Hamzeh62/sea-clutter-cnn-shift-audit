# Reproducing the numerical evidence from released artifacts

1. Python 3.12 (tested here with an available Python environment, not a clean Windows/IPEX raw-data runtime); install `numpy scipy matplotlib PyYAML`.
2. Run `python verify_handoff.py`. This checks 3,177 stored-artifact properties and, if PyTorch is installed, loads both checkpoint state dictionaries.
3. Run `python reproduce/rebuild_saved_figures.py` to recreate five full figure types and a diagnostic of the two CNN H0 tails. This never reads raw IPIX CDF and never modifies original numerical files.
4. Compare `reproduced_figures/*.pdf` with `manuscript/figures/*.pdf`. The original Figure 1 additionally includes ANMF-DL H0 scores not provided in this transfer package. Figure 6 is preserved exactly as a PDF but its raw range-Doppler pixels cannot be regenerated without IPIX data.
5. Verify claims with `provenance/claim_sources.csv` (340 claim mappings) and saved JSON/NPZ arrays; compare study conditions with `docs/PROTOCOL_DIFFERENCES.md`.
6. For complete raw-data-to-score regeneration, obtain authorized IPIX CDF, reconstruct the window cache, and adapt historical workstation paths in `code/` after independently validating data indexing and the historical protocol. The released artifacts do **not** certify this path.

The original benchmark uses a historical 9/4/3 split and is not equivalent to Paper 1's corrected 8/8 cross-fold campaign. Never silently relabel its operating points.
