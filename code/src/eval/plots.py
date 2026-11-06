"""Shared figure styling (spec §8): colorblind-safe palette, fixed markers.

Proposed=red circles, ANMF=blue squares, GLRT=green triangles,
CA-CFAR=orange diamonds, OS-CFAR=purple stars; ablations dashed.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "Proposed": dict(color="#D55E00", marker="o", linestyle="-", label="Proposed"),
    "ANMF": dict(color="#0072B2", marker="s", linestyle="-", label="ANMF"),
    "GLRT": dict(color="#009E73", marker="^", linestyle="-", label="GLRT"),
    "CA-CFAR": dict(color="#E69F00", marker="D", linestyle="-", label="CA-CFAR"),
    "OS-CFAR": dict(color="#CC79A7", marker="*", linestyle="-", label="OS-CFAR"),
    # ablations (dashed)
    "Proposed-cold-ica": dict(color="#D55E00", marker="o", linestyle="--", label="cold ICA"),
    "Proposed-no-phys": dict(color="#D55E00", marker="v", linestyle="--", label="no physics loss"),
    "Proposed-net-only": dict(color="#D55E00", marker="x", linestyle="--", label="network only"),
    "ICA-only": dict(color="#56B4E9", marker="o", linestyle="--", label="ICA only (cold)"),
}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
})
