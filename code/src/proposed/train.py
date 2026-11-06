"""Training (spec §4.4): S0 pretrain on synthetic SIRP clutter (full supervision),
S1 fine-tune on real clutter + injected targets (partial supervision).
AdamW, cosine lr 1e-3 -> 1e-5, early stopping; curriculum ramp of lambda_1/2.
CPU pragmatism: average-pool (stride 4) before the CV-LSTM (deviation D12).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.synth import synth_compound_gaussian, swerling_target, clear_doppler_band
from src.data.ipix_loader import cpi_windows
from src.proposed.m1_network import M1Backbone
from src.proposed.m2_mixhead import M2Heads
from src.proposed.losses import total_loss
from src.eval.units import UnitSampler, load_split, target_cells_map


def load_K_params() -> tuple[float, float]:
    st = json.loads((ROOT / "results/json/clutter_stats.json").read_text())
    nus = [c["K"]["nu"] for f in st.values() for c in f["cells"] if 0.05 < c["K"]["nu"] < 100]
    bs = [c["K"]["b"] for f in st.values() for c in f["cells"]]
    return float(np.median(nus)), float(np.median(bs))


def synth_batch(B, Np, nu, b, rng, prf=1000.0):
    """Synthetic S0 batch: x = A [s_C ; s_T], M=2 channels."""
    M = 2
    A = (rng.standard_normal((B, M, M)) + 1j * rng.standard_normal((B, M, M))) * 0.7
    x = np.zeros((B, M, Np), np.complex64)
    s_T = np.zeros((B, Np), np.complex64)
    a_T = np.zeros((B, M), np.complex64)
    label = np.zeros(B, np.float32)
    for i in range(B):
        sc = synth_compound_gaussian(Np, nu, b, None, 0.9, rng)
        if rng.random() < 0.5:
            scr_db = rng.uniform(-5, 20)
            f_d = rng.uniform(-0.4, 0.4) * prf
            m2 = np.mean(np.abs(sc) ** 2)
            amp = np.sqrt(10 ** (scr_db / 10) * m2)
            st = swerling_target(Np, f_d, prf, amp, int(rng.integers(0, 3)), rng)
            a = rng.standard_normal(M) + 1j * rng.standard_normal(M)
            a_T[i] = a / np.linalg.norm(a)
            s_T[i] = st.astype(np.complex64)
            label[i] = 1.0
        else:
            a_T[i] = np.array([1, 0], np.complex64)
        s_vec = np.stack([sc, s_T[i] if label[i] else np.zeros(Np, np.complex64)])
        x[i] = A[i] @ s_vec
    return x, a_T, s_T, label


def s_pre_from(x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
    """s_pre = pinv(A_hat) x  (torch complex)."""
    Xc = torch.view_as_complex(x.float()) if not torch.is_complex(x) else x
    # x arrives as (B, M, Np) complex
    pinv = torch.linalg.pinv(A_hat)
    return pinv @ Xc


CFG_B = 16


def run_epoch(model, opt, batches, Np, nu, b, rng, device, lambdas, prf=1000.0):
    model.train()
    tot, nb = 0.0, 0
    for _ in range(batches):
        xb, a_T, s_T, y = synth_batch(CFG_B, Np, nu, b, rng, prf)
        xt = torch.from_numpy(xb).to(device)
        a_Tt = torch.from_numpy(a_T).to(device)
        s_Tt = torch.from_numpy(s_T).to(device)
        yt = torch.from_numpy(y).to(device)
        out = model(xt)
        S_pre = s_pre_from(xt, out["A"])
        loss, parts = total_loss(out, xt, S_pre, s_Tt, a_Tt, yt,
                                 K_params=(nu, b), lambdas=lambdas)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += float(loss)
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def val_epoch(model, batches, Np, nu, b, rng, device, lambdas, prf=1000.0):
    model.eval()
    tot, nb = 0.0, 0
    for _ in range(max(4, batches // 4)):
        xb, a_T, s_T, y = synth_batch(CFG_B, Np, nu, b, rng, prf)
        xt = torch.from_numpy(xb).to(device)
        out = model(xt)
        S_pre = s_pre_from(xt, out["A"])
        loss, _ = total_loss(out, xt, S_pre, torch.from_numpy(s_T).to(device),
                             torch.from_numpy(a_T).to(device),
                             torch.from_numpy(y).to(device),
                             K_params=(nu, b), lambdas=lambdas)
        tot += float(loss)
        nb += 1
    return tot / max(nb, 1)


def main():
    CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    TCFG = CFG["training"]
    torch.manual_seed(CFG["seed"])
    torch.set_num_threads(TCFG["torch_threads"])
    rng = np.random.default_rng(CFG["seed"])
    device = "cpu"
    Np = CFG["processing"]["Np"]
    nu, b = load_K_params()
    print(f"[train] K params (nu, b) = ({nu:.3f}, {b:.3f})")

    backbone = M1Backbone(M=2, chans=tuple(CFG["model"]["conv_channels"]),
                          ks=tuple(CFG["model"]["kernel_sizes"]),
                          hidden=CFG["model"]["lstm_hidden"])
    model = M2Heads(M=2, d=2, Np=Np, backbone=backbone, det_head=True).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=TCFG["lr"])
    steps = TCFG["epochs"] * TCFG["steps_per_epoch"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=TCFG["lr_min"])
    best_val, patience = np.inf, 0
    t0 = time.time()
    step = 0
    for epoch in range(TCFG["epochs"]):
        ramp = min(1.0, epoch / max(1, TCFG["ramp_epochs"]))
        lambdas = (TCFG["lambda_A"], TCFG["lambda_r"], TCFG["lambda_1"],
                   TCFG["lambda_2"] * (0.1 + 0.9 * ramp) if TCFG["lambda_2"] else 0.0,
                   TCFG["lambda_3"], TCFG["lambda_4"])
        tr = run_epoch(model, opt, TCFG["steps_per_epoch"], Np, nu, b, rng, device, lambdas)
        for _ in range(TCFG["steps_per_epoch"]):
            sched.step()
            step += 1
        va = val_epoch(model, TCFG["steps_per_epoch"], Np, nu, b, rng, device, lambdas)
        marker = ""
        if va < best_val - 1e-4:
            best_val, patience = va, 0
            torch.save(model.state_dict(), ROOT / "results/checkpoints/best_s0.pt")
            marker = " *"
        else:
            patience += 1
        print(f"[train] ep {epoch+1}/{TCFG['epochs']} loss {tr:.3f} val {va:.3f}"
              f" lr {sched.get_last_lr()[0]:.2e} {int(time.time()-t0)}s{marker}")
        if patience >= TCFG["early_stop_patience"]:
            print("[train] early stopping")
            break
    print(f"[train] S0 done in {int(time.time()-t0)}s; best val {best_val:.3f}")


if __name__ == "__main__":
    main()
