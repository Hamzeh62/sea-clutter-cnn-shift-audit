# Reproducibility scope and limitations

The included saved arrays, logs and metadata permit result inspection and independent recalculation of many quoted numbers **without** rerunning the radar detectors. This is **artifact-level reanalysis**, not raw-data-to-score reproducibility.

**Exact original experiment:** CNNs trained with historical 9/4/3 acquisition split and a VV-only synthetic target injection protocol. Code and weights are preserved as-is. Later Paper 1 analysis identified unreliable target/guard lookup for some cached identifiers and differences in calibration protocols. Numerical results are retained, but cannot be presented as independent validation under the later corrected protocol.

**Plot regeneration:** five original publication plot types can be regenerated from saved results; Figure 1 can be regenerated for its two CNN curves but the independent ANMF-DL H0 score vector is not present; Figure 6 requires raw CDF for true range–Doppler pixels. Exact original vector PDFs for both are preserved.

**Dataset:** externally obtained raw IPIX is required for full rerun. The source window cache (reported 887 MiB) is not included. Original code retains historical paths and dependencies. Never use simulated/synthetic substitute data to claim replication of an IPIX measurement.

**Checkpoints:** model weights are included, but re-training has not been independently repeated in the release packaging environment.

**Statistical scope:** 10^-4 design calibration relies on tail extrapolation; nominal design PFA must not be equated with independently validated PFA after sea-state/acquisition shift. Source statistical uncertainty is preserved as reported, not strengthened.
