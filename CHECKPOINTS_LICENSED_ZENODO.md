# Trained CNN checkpoints (archived separately)

Both trained PyTorch checkpoints have been preserved without altering their bytes. They are **not committed to GitHub**, to keep the software release lightweight. The original `.pt` files are included inside the separate `PAPER2_CHECKPOINTS_v1.0.0.zip` archive for direct Zenodo deposit.

| Filename | Original SHA-256 | License |
| --- | --- | --- |
| `best_cnn_complex.pt` | `292be8e46eaa766c01ce235c0b014b743735f5e4798c5332a2e05397fda9195a` | CC BY 4.0 |
| `best_cnn_real_iq.pt` | `6e2f1512fff2618d6f2a86feceafde1eaf0c95886992bb25ac3ef5fec0e34979` | CC BY 4.0 |


Use the actual published Zenodo DOI once the artifacts record exists; do not invent one. Only load PyTorch checkpoints from sources you trust; prefer `weights_only=True` where supported. These are model weights, not original IPIX radar recordings.
