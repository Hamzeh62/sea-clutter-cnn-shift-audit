"""End-to-end hybrid detector inference: network -> warm-started cICA ->
separated target channel -> NMF statistic (M4)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.proposed.m1_network import M1Backbone
from src.proposed.m2_mixhead import M2Heads
from src.proposed.m3_cica import newton_schulz_polar, whiten
from src.proposed.m4_detector import separate, nmf_statistic


class HybridDetector:
    def __init__(self, ckpt: str | Path | None = None, cfg_path: str | Path | None = None,
                 device: str = "cpu"):
        cfg = yaml.safe_load((cfg_path or ROOT / "configs/default.yaml").read_text())
        self.cfg = cfg
        self.Np = cfg["processing"]["Np"]
        self.backbone = M1Backbone(M=2, chans=tuple(cfg["model"]["conv_channels"]),
                                   ks=tuple(cfg["model"]["kernel_sizes"]),
                                   hidden=cfg["model"]["lstm_hidden"])
        self.model = M2Heads(M=2, d=2, Np=self.Np, backbone=self.backbone,
                             det_head=True).to(device)
        if ckpt is not None and Path(ckpt).exists():
            self.model.load_state_dict(torch.load(ckpt, map_location=device))
        self.model.eval()
        self.device = device
        self.ica_cfg = cfg["model"]["ica"]
        self.ica_fix = {k: cfg["model"].get("ica_fix", {}).get(k)
                        for k in ("damping", "best_iterate", "perm_invariant",
                                  "stagnation_window", "stagnation_thresh", "delta")}
        self._prep_sep = {}   # cache: file_key -> prep
        self.iters_log = {"warm": [], "cold": []}
        self.wall_log = []
        # D36 fine-grained per-stage timers (cumulative seconds)
        self._T = {"whiten": 0.0, "warm_svd": 0.0, "ica": 0.0,
                   "nmf": 0.0, "hoisted": 0}

    @torch.no_grad()
    def network_out(self, x: np.ndarray) -> dict:
        xt = torch.from_numpy(np.ascontiguousarray(x.astype(np.complex64))[None]).to(self.device)
        return {k: (v[0].cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in self.model(xt).items()}

    @torch.no_grad()
    def network_out_batch(self, xs: np.ndarray, chunk: int = 32) -> list[dict]:
        """xs: (B, 2, Np) complex -> list of per-sample output dicts (batched)."""
        outs = []
        for i in range(0, len(xs), chunk):
            xb = torch.from_numpy(np.ascontiguousarray(xs[i:i + chunk].astype(np.complex64))).to(self.device)
            o = self.model(xb)
            outs.extend([{k: (v[j].cpu().numpy() if torch.is_tensor(v) else v)
                          for k, v in o.items()} for j in range(xb.shape[0])])
        return outs

    def _fix_kwargs(self) -> dict:
        f = self.ica_fix
        stag = None
        if f.get("stagnation_window"):
            stag = (int(f["stagnation_window"]), float(f.get("stagnation_thresh", 1e-4)))
        return {"damping": float(f.get("damping") or 0.0),
                "best_iterate": bool(f.get("best_iterate", False)),
                "perm_invariant": bool(f.get("perm_invariant", False)),
                "stagnation": stag}

    @property
    def _delta(self) -> float:
        """D36 hoist gate: > 0 enables single-whiten/single-polar in M3."""
        return float(self.ica_fix.get("delta") or 0.0)

    def _timed_whiten(self, x: np.ndarray):
        """D36: whiten once per unit (timed); result reused by the warm-start
        SVD and handed to separate() so the M3 path does not recompute it."""
        t0 = time.perf_counter()
        z, V, d, _lam = whiten(x, mdl=True)
        self._T["whiten"] += time.perf_counter() - t0
        return z, V, d

    def statistic_from_net(self, x: np.ndarray, net_out: dict, prep_sep=None,
                           warm_start: bool = True) -> tuple[float, dict]:
        """Separation + NMF using precomputed network outputs (batched path).

        D36: whiten once per unit (reused for the warm-start polar), skip the
        redundant Newton-Schulz polar pass when the warm start is already the
        SVD polar factor, and skip the per-unit Chat refactorization when the
        prep carries the hoisted pool-level inverse (key `Cinv`). Preps and
        configs without the D36 keys keep the exact legacy compute path.
        """
        A_hat, r = net_out["A"], net_out["r"]
        z, V, d = self._timed_whiten(x)
        W_warm = None
        if warm_start:
            t0 = time.perf_counter()
            VA = V @ A_hat[:, :d]
            U_, S_, Vh_ = np.linalg.svd(VA)
            W_warm = U_ @ Vh_
            self._T["warm_svd"] += time.perf_counter() - t0
        hoisted = ("Cinv" in prep_sep) if prep_sep is not None else False
        kw = self._fix_kwargs()
        t0 = time.perf_counter()
        if hoisted and self._delta > 0.0:
            sep = separate(x, W_warm=W_warm, r=r, xi=self.ica_cfg["xi"],
                           mu=self.ica_cfg["mu"], tol=self.ica_cfg["tol"],
                           max_sweeps=self.ica_cfg["max_sweeps"],
                           pre_whiten=(z, V, d), polar_done=True, **kw)
            self._T["hoisted"] += 1
        else:
            sep = separate(x, W_warm=W_warm, r=r, xi=self.ica_cfg["xi"],
                           mu=self.ica_cfg["mu"], tol=self.ica_cfg["tol"],
                           max_sweeps=self.ica_cfg["max_sweeps"], **kw)
        self._T["ica"] += time.perf_counter() - t0
        self.iters_log["warm" if warm_start else "cold"].append(sep["iters"])
        self.wall_log.append(sep.get("wall_s", float("nan")))
        s_T = sep["S"][0]
        prep = prep_sep if prep_sep is not None else self._prep_sep["__default__"]
        t0 = time.perf_counter()
        t = nmf_statistic(s_T, prep)
        self._T["nmf"] += time.perf_counter() - t0
        return t, {"iters": sep["iters"], "warm": sep["warm"], "s_T": s_T}

    def statistic(self, x: np.ndarray, prep_sep=None, use_network=True,
                  warm_start=True) -> tuple[float, dict]:
        """x: (2, Np) complex channels (VV, VH). Returns (t, info).

        use_network=False + warm_start=False -> 'ICA alone, cold' ablation.
        use_network=True  + warm_start=False -> 'cold-start ICA' ablation.
        """
        if use_network:
            out = self.network_out(x)
            A_hat = out["A"]            # (M, d)
            r = out["r"]                # (Np,)
        else:
            # deterministic fallback mixing init from principal eigenpair
            z0, V0, d0, _ = whiten(x, mdl=False)
            A_hat = (np.linalg.pinv(V0)).conj().T[:, :d0]
            P = np.eye(x.shape[1], dtype=np.complex128)
            r = None
        z, V, d = self._timed_whiten(x)
        W_warm = None
        if warm_start and use_network:
            t0 = time.perf_counter()
            VA = V @ A_hat[:, :d]       # (d, d)
            U_, S_, Vh_ = np.linalg.svd(VA)
            W_warm = U_ @ Vh_           # P Q^H  (spec M3 warm start)
            self._T["warm_svd"] += time.perf_counter() - t0
        kw = self._fix_kwargs()
        t0 = time.perf_counter()
        if "Cinv" in (prep_sep or {}) and self._delta > 0.0:
            sep = separate(x, W_warm=W_warm, r=r, xi=self.ica_cfg["xi"],
                           mu=self.ica_cfg["mu"], tol=self.ica_cfg["tol"],
                           max_sweeps=self.ica_cfg["max_sweeps"],
                           pre_whiten=(z, V, d), polar_done=True, **kw)
            self._T["hoisted"] += 1
        else:
            sep = separate(x, W_warm=W_warm, r=r, xi=self.ica_cfg["xi"],
                           mu=self.ica_cfg["mu"], tol=self.ica_cfg["tol"],
                           max_sweeps=self.ica_cfg["max_sweeps"], **kw)
        self._T["ica"] += time.perf_counter() - t0
        key = "warm" if (warm_start and use_network) else "cold"
        self.iters_log[key].append(sep["iters"])
        self.wall_log.append(sep.get("wall_s", float("nan")))
        s_T = sep["S"][0]               # anchored target channel
        prep = prep_sep if prep_sep is not None else self._prep_sep.get("__default__")
        if prep is None:
            # fallback: identity-covariance NMF (rare; only if caller forgot prep)
            prep = {"Chat": np.eye(self.Np), "CinvP": np.fft.fft(np.eye(self.Np), axis=0),
                    "a": np.full(self.Np, self.Np / 3)}
        t0 = time.perf_counter()
        t = nmf_statistic(s_T, prep)
        self._T["nmf"] += time.perf_counter() - t0
        # NB3 (D-NB3-2): full separated source matrix exposed for readout
        # variants; a dict key only -- no numeric value changes.
        return t, {"iters": sep["iters"], "warm": sep["warm"],
                   "anchor_corr": sep["anchor_corr"], "s_T": s_T,
                   "A_hat": A_hat, "S_full": sep["S"]}
