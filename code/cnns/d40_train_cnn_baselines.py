"""D40 (stage 1) — M6 learned baselines: train two small window classifiers.

Baselines (Intro challenge (c)): a plain discriminative CNN given the SAME
real clutter, the SAME injection protocol and the SAME windowing as the
hybrid — one complex-valued CNN (ComplexConv1d/ModReLU stack, ~1.8e5 params)
and one real-valued I/Q CNN (matched real-Conv1d capacity, ~1.8e5 params).

Training data (S1 protocol, polarimetric geometry — documented):
  sampler = UnitSampler(TRAIN files, Np=512, hop=256, pol=VV, seed=42+101=143)
  units from clutter-only cells (target_cells_map exclusion), 50% of windows
  receive an injected Swerling-0 target (phase-4 geometry: s_T added to the
  VV channel only; VH stays clutter-only), f_D from the file's clear-Doppler
  band (guard 8), SCR ~ U(-5, 15) dB, amplitude = sqrt(SCR * m2(vv)).
  Per-window RMS normalization (both channels, both nets) — the hybrid needs
  no such choice because whitening normalizes scale; documented for M6.
Both nets see IDENTICAL batches (paired training, shared rng 42+102=144);
per-epoch validation on fresh batches (seed 900+epoch, S1 convention);
BCE loss, AdamW(1e-3, wd 1e-4) cosine to 1e-5; capped at 60 epochs x 50
steps x 16 = 48k window draws (more optimization than the hybrid's S1
budget of 450 steps; still capped), early stop patience 10 on val BCE.

Outputs:
  results/checkpoints/best_cnn_complex.pt, best_cnn_real_iq.pt
  results/json/d40_cnn_train.json  (architecture, params, budget, history)
  results/train_d40_log.txt        (per-epoch progress lines)
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.units import UnitSampler, load_split, target_cells_map
from src.eval.metrics import auc_from_scores
from src.data.synth import swerling_target, estimate_psd, clear_doppler_band
from src.proposed.m1_network import ComplexConv1d, ComplexBatchNorm1d, ModReLU, c_to_r

CFG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
Np = CFG["processing"]["Np"]
HOP = CFG["processing"]["hop"]
PRF = 1000.0
SAMPLER_SEED = CFG["seed"] + 101   # 143 (S1 convention)
RNG_SEED = CFG["seed"] + 102       # 144 (S1 convention)
TORCH_SEED = CFG["seed"] + 103
BATCH = 16
EPOCHS_CAP = 60
STEPS_PER_EPOCH = 50
PATIENCE = 10
LR, WD, LR_MIN = 1e-3, 1e-4, 1e-5
SCR_RANGE = (-5.0, 15.0)
P_TARGET = 0.5

CKPT_C = ROOT / "results/checkpoints/best_cnn_complex.pt"
CKPT_R = ROOT / "results/checkpoints/best_cnn_real_iq.pt"
OUTJ = ROOT / "results/json/d40_cnn_train.json"
LOG = ROOT / "results/train_d40_log.txt"


def normalize(xs: np.ndarray) -> np.ndarray:
    """Per-window RMS normalization (identical for both nets, train + eval)."""
    rms = np.sqrt(np.mean(np.abs(xs) ** 2, axis=(1, 2), keepdims=True))
    return xs / np.maximum(rms, 1e-12)


def make_batch(sampler, bands, cells_excl, rng, B=BATCH):
    """S1 protocol, polarimetric geometry: channel0 = VV (+ target), channel1 = VH."""
    xs = np.zeros((B, 2, Np), np.complex64)
    y = np.zeros(B, np.float32)
    for i in range(B):
        for _ in range(50):
            key, cell, w0 = sampler.units[int(rng.integers(0, len(sampler)))]
            if cells_excl.get(key) and cell in cells_excl[key]:
                continue
            break
        vv = sampler.data[key][w0:w0 + Np, cell].astype(np.complex128)
        vh = sampler.data_cross[key][w0:w0 + Np, cell].astype(np.complex128)
        if rng.random() < P_TARGET:
            b = bands[key]
            f_d = float(rng.uniform(b[0], b[1]))
            scr_db = float(rng.uniform(*SCR_RANGE))
            m2 = float(np.mean(np.abs(vv) ** 2))
            amp = np.sqrt(10 ** (scr_db / 10.0) * m2)
            st = swerling_target(Np, f_d, PRF, amp, 0, rng)
            xs[i] = np.stack([vv + st, vh]).astype(np.complex64)
            y[i] = 1.0
        else:
            xs[i] = np.stack([vv, vh]).astype(np.complex64)
    return normalize(xs), y


class CNNComplex(nn.Module):
    """Complex-valued CNN: ComplexConv1d/BN/ModReLU stack + magnitude-stat head."""
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv1d(2, 32, 7, padding=3)
        self.bn1 = ComplexBatchNorm1d(32)
        self.act1 = ModReLU(32)
        self.conv2 = ComplexConv1d(32, 64, 5, stride=2, padding=2)
        self.bn2 = ComplexBatchNorm1d(64)
        self.act2 = ModReLU(64)
        self.conv3 = ComplexConv1d(64, 64, 5, stride=2, padding=2)
        self.bn3 = ComplexBatchNorm1d(64)
        self.act3 = ModReLU(64)
        self.conv4 = ComplexConv1d(64, 64, 3, stride=2, padding=1)
        self.bn4 = ComplexBatchNorm1d(64)
        self.act4 = ModReLU(64)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x_complex):                    # (B, 2, Np) complex
        x = c_to_r(x_complex)                        # (B, 4, Np) real
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))       # (B, 128, Np/8)
        re, im = x[:, :64], x[:, 64:]
        mag = torch.sqrt(re * re + im * im + 1e-12)  # (B, 64, L)
        feat = torch.cat([mag.mean(dim=2), mag.std(dim=2)], dim=1)
        return self.fc2(F.relu(self.fc1(feat))).squeeze(-1)


class CNNRealIQ(nn.Module):
    """Real-valued I/Q CNN: matched capacity, I/Q stacked as 4 real channels."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x_complex):                    # (B, 2, Np) complex
        x = c_to_r(x_complex)                        # (B, 4, Np) real
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        feat = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return self.fc2(F.relu(self.fc1(feat))).squeeze(-1)


def main():
    t00 = time.time()
    torch.set_num_threads(int(CFG["runtime"].get("torch_threads", 6)))
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)

    netC, netR = CNNComplex(), CNNRealIQ()
    nC = sum(p.numel() for p in netC.parameters())
    nR = sum(p.numel() for p in netR.parameters())
    print(f"[d40] params: complex {nC:,}, real-IQ {nR:,}", flush=True)

    split = load_split(CFG["split"]["seed"])
    sampler = UnitSampler(split["train"], Np, HOP, pol=CFG["data"]["primary_pol"],
                          seed=SAMPLER_SEED)
    cells_excl = target_cells_map()
    bands = {}
    for key, cube in sampler.data.items():
        f, p = estimate_psd(cube[:, min(3, cube.shape[1] - 1)], PRF, 512)
        bands[key] = clear_doppler_band(p, f, guard_bins=8)
    print(f"[d40] train units: {len(sampler):,} from {len(sampler.data)} files", flush=True)

    optC = torch.optim.AdamW(netC.parameters(), lr=LR, weight_decay=WD)
    optR = torch.optim.AdamW(netR.parameters(), lr=LR, weight_decay=WD)
    schC = torch.optim.lr_scheduler.CosineAnnealingLR(optC, T_max=EPOCHS_CAP, eta_min=LR_MIN)
    schR = torch.optim.lr_scheduler.CosineAnnealingLR(optR, T_max=EPOCHS_CAP, eta_min=LR_MIN)
    lossf = nn.BCEWithLogitsLoss()

    history, bestC, bestR, patC, patR = [], np.inf, np.inf, 0, 0
    rng = np.random.default_rng(RNG_SEED)   # one advancing stream, S1 convention
    logf = open(LOG, "a")
    stop = False
    for epoch in range(EPOCHS_CAP):
        netC.train(); netR.train()
        trC = trR = 0.0
        for _ in range(STEPS_PER_EPOCH):
            xb, yb = make_batch(sampler, bands, cells_excl, rng)
            xt = torch.from_numpy(xb)
            yt = torch.from_numpy(yb)
            lc = lossf(netC(xt), yt)
            optC.zero_grad(); lc.backward()
            torch.nn.utils.clip_grad_norm_(netC.parameters(), 5.0); optC.step()
            lr_ = lossf(netR(xt), yt)
            optR.zero_grad(); lr_.backward()
            torch.nn.utils.clip_grad_norm_(netR.parameters(), 5.0); optR.step()
            trC += float(lc); trR += float(lr_)
        schC.step(); schR.step()

        netC.eval(); netR.eval()
        with torch.no_grad():
            yv, svC, svR = [], [], []
            for vb in range(5):
                xb, yb = make_batch(sampler, bands, cells_excl, np.random.default_rng(
                    CFG["seed"] + 900 + epoch * 10 + vb), B=BATCH)
                xt = torch.from_numpy(xb)
                svC.append(netC(xt).numpy()); svR.append(netR(xt).numpy())
                yv.append(yb)
            yv = np.concatenate(yv); svC = np.concatenate(svC); svR = np.concatenate(svR)
        neg, pos = yv == 0, yv == 1
        vC = float(lossf(torch.from_numpy(svC), torch.from_numpy(yv)))
        vR = float(lossf(torch.from_numpy(svR), torch.from_numpy(yv)))
        aC = auc_from_scores(svC[neg], svC[pos])
        aR = auc_from_scores(svR[neg], svR[pos])
        history.append({"epoch": epoch + 1, "train_bce_C": trC / STEPS_PER_EPOCH,
                        "train_bce_R": trR / STEPS_PER_EPOCH,
                        "val_bce_C": vC, "val_bce_R": vR,
                        "val_auc_C": float(aC), "val_auc_R": float(aR)})
        mark = ""
        if vC < bestC - 1e-4:
            bestC, patC = vC, 0
            torch.save(netC.state_dict(), CKPT_C); mark += " *C"
        else:
            patC += 1
        if vR < bestR - 1e-4:
            bestR, patR = vR, 0
            torch.save(netR.state_dict(), CKPT_R); mark += " *R"
        else:
            patR += 1
        line = (f"[d40] ep {epoch+1}/{EPOCHS_CAP} trC {trC/STEPS_PER_EPOCH:.3f} "
                f"trR {trR/STEPS_PER_EPOCH:.3f} valC {vC:.3f} valR {vR:.3f} "
                f"aucC {aC:.3f} aucR {aR:.3f} {int(time.time()-t00)}s{mark}")
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
        if patC >= PATIENCE and patR >= PATIENCE:
            stop = True
            logf.write("[d40] early stopping (both nets)\n")
            break
    logf.close()

    out = {
        "config": {
            "protocol": "S1 real clutter + injected Swerling-0, polarimetric geometry "
                        "(s_T in VV only; VH clutter-only), target cells excluded, "
                        "P_target=0.5, SCR~U(-5,15) dB, f_D from clear-Doppler band "
                        "(guard 8), Np=512 hop=256",
            "sampler_seed": SAMPLER_SEED, "rng_seed": RNG_SEED,
            "torch_seed": TORCH_SEED,
            "normalization": "per-window RMS (both channels, both nets, train+eval)",
            "loss": "BCEWithLogits", "optimizer": "AdamW lr 1e-3 wd 1e-4, cosine to 1e-5",
            "budget": {"epochs_cap": EPOCHS_CAP, "steps_per_epoch": STEPS_PER_EPOCH,
                       "batch": BATCH, "max_window_draws": EPOCHS_CAP * STEPS_PER_EPOCH * BATCH,
                       "early_stop_patience": PATIENCE},
            "params": {"complex": nC, "real_iq": nR},
            "architecture": {
                "complex": "ComplexConv1d 2->32 k7 / 32->64 k5 s2 / 64->64 k5 s2 / "
                           "64->64 k3 s2, CBN + ModReLU, magnitude mean+std head 128->64->1",
                "real_iq": "Conv1d 4->32 k7 / 32->64 k5 s2 / 64->64 k5 s2 / 64->64 k3 s2, "
                           "BN + GELU, mean+std head 128->64->1"},
        },
        "history": history,
        "final": {"best_val_bce_C": float(bestC), "best_val_bce_R": float(bestR),
                  "epochs_run": len(history), "early_stopped": stop,
                  "wall_s": int(time.time() - t00)},
    }
    OUTJ.write_text(json.dumps(out, indent=2))
    print(f"[d40] done in {int(time.time()-t00)}s; best val BCE C {bestC:.4f} / "
          f"IQ {bestR:.4f} -> {CKPT_C.name}, {CKPT_R.name}", flush=True)


if __name__ == "__main__":
    main()
