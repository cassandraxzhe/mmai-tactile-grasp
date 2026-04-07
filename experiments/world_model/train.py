"""
Training loop for video diffusion world model.
Task: given N pre-contact egocentric frames, diffuse/denoise the frame at peak contact.

Run: python train.py --data_root /path/to/opentouch/data
"""
import argparse, io, os
import h5py, numpy as np, pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers.optimization import get_cosine_schedule_with_warmup
from model import ContextEncoder, build_unet, build_scheduler

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE    = 64        # keep small for speed
N_FRAMES    = 8         # pre-contact context frames
BATCH       = 8
EPOCHS      = 30
LR          = 1e-4
GRAD_ACCUM  = 2
T_STEPS     = 1000      # diffusion timesteps
DEVICE      = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")

SPLITS_DIR  = Path(__file__).resolve().parents[2] / "data" / "splits"

# ── Dataset ───────────────────────────────────────────────────────────────────
class GraspVideoDataset(Dataset):
    def __init__(self, csv_path, data_root, n_frames=N_FRAMES):
        self.df        = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.n_frames  = n_frames
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),  # → [-1, 1]
        ])

    def __len__(self): return len(self.df)

    def _load(self, hdf5, demo_id, indices):
        with h5py.File(hdf5, "r") as f:
            jpegs = f["data"][demo_id]["rgb_images_jpeg"]
            frames = [self.tf(Image.open(io.BytesIO(bytes(jpegs[i]))).convert("RGB"))
                      for i in indices]
        return torch.stack(frames)  # (N, 3, H, W)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        hdf5    = self.data_root / Path(row["hdf5_path"]).name
        onset   = int(row["onset_idx"])
        peak    = int(row["peak_idx"])

        # Last n_frames before onset (pad at start if clip is short)
        idxs = list(range(max(0, onset - self.n_frames), onset))
        while len(idxs) < self.n_frames:
            idxs = [idxs[0]] + idxs
        idxs = idxs[-self.n_frames:]

        try:
            context = self._load(hdf5, row["demo_id"], idxs)       # (N, 3, H, W)
            target  = self._load(hdf5, row["demo_id"], [peak])[0]  # (3, H, W)
        except Exception:
            z = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            return torch.zeros(self.n_frames, 3, IMG_SIZE, IMG_SIZE), z

        return context, target  # context conditions the diffusion; target is x_0

# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print(f"Device: {DEVICE}  |  img_size: {IMG_SIZE}  |  T: {T_STEPS}")

    train_ds = GraspVideoDataset(SPLITS_DIR / "train.csv", args.data_root)
    val_ds   = GraspVideoDataset(SPLITS_DIR / "val.csv",   args.data_root)
    train_dl = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=4, pin_memory=True)

    cross_attn_dim = 512

    # Build model components
    unet      = build_unet(IMG_SIZE, cross_attn_dim).to(DEVICE)
    ctx_enc   = ContextEncoder(N_FRAMES, cross_attn_dim).to(DEVICE)
    scheduler = build_scheduler(T_STEPS)

    params = list(unet.parameters()) + list(ctx_enc.proj.parameters())
    opt    = torch.optim.AdamW(params, lr=LR)
    lr_sch = get_cosine_schedule_with_warmup(opt, 500, len(train_dl) * EPOCHS // GRAD_ACCUM)

    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        unet.train()
        ctx_enc.train()
        train_losses = []
        opt.zero_grad()

        for step, (ctx, tgt) in enumerate(train_dl):
            ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)   # (B,N,3,H,W), (B,3,H,W)

            # Sample random timesteps and add noise to target (forward process)
            t     = torch.randint(0, T_STEPS, (tgt.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(tgt)
            noisy = scheduler.add_noise(tgt, noise, t)

            # Encode context frames → cross-attention conditioning
            encoder_hidden = ctx_enc(ctx)  # (B, N, cross_attn_dim)

            # Predict noise (standard DDPM objective)
            noise_pred = unet(noisy, t, encoder_hidden_states=encoder_hidden).sample
            loss = F.mse_loss(noise_pred, noise) / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                lr_sch.step()
                opt.zero_grad()

            train_losses.append(loss.item() * GRAD_ACCUM)

        # ── val: run full denoising and measure MSE against clean target ──
        unet.eval()
        ctx_enc.eval()
        val_mse, n = 0.0, 0
        with torch.no_grad():
            for ctx, tgt in val_dl:
                ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)
                encoder_hidden = ctx_enc(ctx)

                # Single-step noise prediction at a fixed mid-level t for fast eval
                t_eval = torch.full((tgt.shape[0],), T_STEPS // 2, device=DEVICE).long()
                noise  = torch.randn_like(tgt)
                noisy  = scheduler.add_noise(tgt, noise, t_eval)
                pred   = unet(noisy, t_eval, encoder_hidden_states=encoder_hidden).sample
                val_mse += F.mse_loss(pred, noise).item() * len(tgt)
                n += len(tgt)

        val_mse /= n
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={np.mean(train_losses):.4f}"
              f"  val_noise_mse={val_mse:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({"unet": unet.state_dict(),
                        "ctx_enc": ctx_enc.state_dict()}, f"{args.out}/best.pt")
            print(f"  ✓ saved (val_mse={best_val:.4f})")

    print("Done. Best val noise-MSE:", best_val)

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Directory containing the .hdf5 files")
    p.add_argument("--out", default="checkpoints",
                   help="Directory to save checkpoints")
    train(p.parse_args())
