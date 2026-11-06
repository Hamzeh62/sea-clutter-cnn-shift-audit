"""D45b — D40 realized-Pfa presentation fixes (task R8, item 3).

Two defects found during D45 recon, corrected here from the saved artifacts:

(a) d40_cnn_eval.stage_phase2 indexed full_c/full_r by UNIT INDEX instead of
    pool position: `keep` stores sampler.unit indices from the phase-2
    permutation, but full_c/full_r are ordered by pool position (order[i]).
    The published "phase2_anmf4000" realized-Pfa values therefore used a
    different (statistically similar) subset than labeled.  Corrected values
    are computed at the same deployed thresholds via pool-position indexing
    (pos = searchsorted(order, keep)).

(b) the CNN-IQ 1e-4 threshold is extrapolated (phase-4 rule on the 1,500-unit
    calibration pool) while its realized rate on the full 21,462-unit pool is
    far from nominal; the task asks for a re-threshold at the realized
    quantile of the full pool (empirical, n*pfa = 2.1 >= 10 across the pool)
    OR per-arm realized-Pfa annotation.  Both are recorded here.

No MC is re-run: this reads results/json/d40_scores_phase2.npz and the d40
json, and writes results/json/d45b_d40_pfa_fixes.json for the paper/claims
pipeline to cite.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "results/json"

d40 = json.loads((JS / "d40_cnn_baselines.json").read_text())
z = np.load(JS / "d40_scores_phase2.npz")
full_c, full_r, keep = z["full_c"], z["full_r"], z["anmf4000_idx"]

# order replication: d40 stage_phase2 used rng = default_rng(42); order[i]
# is the pool position of unit order[i] (probe: order[0] == keep[0] == 3136)
order = np.random.default_rng(42).permutation(21462)
assert order[0] == keep[0], "phase-2 order replication failed"
# keep is in permutation order (unsorted); map unit indices -> pool positions
# via the inverse permutation (searchsorted would need a sorted array)
inv = np.empty(order.size, dtype=np.int64)
inv[order] = np.arange(order.size)
pos = inv[keep]
assert pos.max() < len(full_c), "pool position out of bounds"

out = {"config": {
    "pool_full_n": int(len(full_c)),
    "pool_anmf4000_n": int(len(keep)),
    "threshold_sources": "d40_cnn_baselines.json (deployed)"},
    "d40_defect_a": {
        "description": ("stage_phase2 indexed full_c/full_r by unit index "
                        "instead of pool position; published phase2_anmf4000 "
                        "rows used a mislabeled subset"),
        "realized_pfa_phase2_anmf4000_corrected": {},
        "realized_pfa_phase2_anmf4000_as_published": {},
        "realized_pfa_phase2_full": {}},
    "d40_defect_b": {
        "description": ("CNN-IQ 1e-4 threshold extrapolated from the 1,500-unit "
                        "pool; re-thresholded at the realized quantile of the "
                        "full 21,462-unit pool (empirical)"),
        "rethreshold": {}}}

for name, arr in (("CNN-C", full_c), ("CNN-IQ", full_r)):
    aspub = full_c if name == "CNN-C" else full_r
    for pfa_key, thr in ((k, v["threshold"]) for k, v in
                         d40["thresholds"][name].items()):
        pfa = float(pfa_key)
        out["d40_defect_a"]["realized_pfa_phase2_anmf4000_corrected"][
            f"{name}@{pfa_key}"] = float(np.mean(aspub[pos] > thr))
        out["d40_defect_a"]["realized_pfa_phase2_anmf4000_as_published"][
            f"{name}@{pfa_key}"] = float(np.mean(aspub[keep] > thr))
        out["d40_defect_a"]["realized_pfa_phase2_full"][
            f"{name}@{pfa_key}"] = float(np.mean(aspub > thr))

# (b) CNN-IQ 1e-4 re-threshold on the full pool
t_emp = float(np.quantile(full_r, 1 - 1e-4))
out["d40_defect_b"]["rethreshold"]["CNN-IQ@0.0001"] = {
    "threshold_deployed_extrapolated": d40["thresholds"]["CNN-IQ"]["0.0001"]["threshold"],
    "threshold_realized_quantile_full_pool": t_emp,
    "rate_at_deployed_full_pool": float(np.mean(full_r > d40["thresholds"]["CNN-IQ"]["0.0001"]["threshold"])),
    "rate_at_rethreshold_full_pool": float(np.mean(full_r > t_emp)),
    "rate_at_rethreshold_anmf4000_poolpos": float(np.mean(full_r[pos] > t_emp))}
t_emp_c = float(np.quantile(full_c, 1 - 1e-4))
out["d40_defect_b"]["rethreshold"]["CNN-C@0.0001"] = {
    "threshold_deployed_extrapolated": d40["thresholds"]["CNN-C"]["0.0001"]["threshold"],
    "threshold_realized_quantile_full_pool": t_emp_c,
    "rate_at_deployed_full_pool": float(np.mean(full_c > d40["thresholds"]["CNN-C"]["0.0001"]["threshold"])),
    "rate_at_rethreshold_full_pool": float(np.mean(full_c > t_emp_c))}

# (c) spot-cell Pd at the re-thresholded 1e-4 values, on the SAME paired
# trials as the d40 MC (d40_scores_trials.npz, rows in ascending-SCR order)
import yaml
grid = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["eval"]["scr_grid_db"]
zt = np.load(JS / "d40_scores_trials.npz")
c_tr, r_tr = zt["c"], zt["r"]
out["d40_defect_b"]["rethreshold_pd_spot_cells"] = {}
for scr in (0.0, 5.0, 10.0):
    i = grid.index(scr)
    out["d40_defect_b"]["rethreshold_pd_spot_cells"][f"CNN-C@{scr:g}"] = {
        "pd_deployed_extrapolated": float(np.mean(c_tr[i] > d40["thresholds"]["CNN-C"]["0.0001"]["threshold"])),
        "pd_rethreshold": float(np.mean(c_tr[i] > t_emp_c))}
    out["d40_defect_b"]["rethreshold_pd_spot_cells"][f"CNN-IQ@{scr:g}"] = {
        "pd_deployed_extrapolated": float(np.mean(r_tr[i] > d40["thresholds"]["CNN-IQ"]["0.0001"]["threshold"])),
        "pd_rethreshold": float(np.mean(r_tr[i] > t_emp))}

(JS / "d45b_d40_pfa_fixes.json").write_text(json.dumps(out, indent=2))
print("[d45b] wrote d45b_d40_pfa_fixes.json")
for k, v in out["d40_defect_a"]["realized_pfa_phase2_anmf4000_corrected"].items():
    pub = out["d40_defect_a"]["realized_pfa_phase2_anmf4000_as_published"][k]
    print(f"  {k}: corrected {v:.5f} (as-published {pub:.5f})")
print(f"  CNN-IQ 1e-4 re-threshold: deployed rate "
      f"{out['d40_defect_b']['rethreshold']['CNN-IQ@0.0001']['rate_at_deployed_full_pool']:.5f}"
      f" -> empirical-quantile rate "
      f"{out['d40_defect_b']['rethreshold']['CNN-IQ@0.0001']['rate_at_rethreshold_full_pool']:.5f}")
for _k, _v in out["d40_defect_b"]["rethreshold_pd_spot_cells"].items():
    print(f"  {_k}: Pd {_v['pd_deployed_extrapolated']:.3f} -> {_v['pd_rethreshold']:.3f}")
