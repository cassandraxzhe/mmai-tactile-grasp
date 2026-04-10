"""
Generate sample predictions from the trained world model.
Saves a grid of (context strip | ground truth | generated) for N val examples.

Run: python generate_samples.py --checkpoint checkpoints/world_model_20260407_090605/best.pt
                                 --data_root /home/akshatat/mmai-tactile-grasp/opentouch/data
                                 --n 8
"""
import argparse, io
import h5py, numpy as np, pandas as pd
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

from model import ContextEncoder, build_unet, build_scheduler, generate

IMG_SIZE   = 64
N_FRAMES   = 8
SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def load_frames(hdf5, demo_id, indices):
    with h5py.File(hdf5, "r") as f:
        jpegs = f["data"][demo_id]["rgb_images_jpeg"]
        frames = [tf(Image.open(io.BytesIO(bytes(jpegs[i]))).convert("RGB"))
                  for i in indices]
    return torch.stack(frames)  # (N, 3, H, W)

def denorm(t):
    """[-1,1] → [0,1]"""
    return (t.clamp(-1, 1) + 1) / 2

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    unet    = build_unet(IMG_SIZE, 512).to(device)
    ctx_enc = ContextEncoder(N_FRAMES, 512).to(device)
    unet.load_state_dict(ckpt["unet"])
    ctx_enc.load_state_dict(ckpt["ctx_enc"])
    scheduler = build_scheduler()

    df = pd.read_csv(SPLITS_DIR / "val.csv")
    df = df.sample(min(args.n, len(df)), random_state=42).reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        hdf5  = Path(args.data_root) / Path(row["hdf5_path"]).name
        onset = int(row["onset_idx"])
        peak  = int(row["peak_idx"])

        idxs = list(range(max(0, onset - N_FRAMES), onset))
        while len(idxs) < N_FRAMES:
            idxs = [idxs[0]] + idxs
        idxs = idxs[-N_FRAMES:]

        try:
            ctx = load_frames(hdf5, row["demo_id"], idxs).to(device)   # (N,3,H,W)
            tgt = load_frames(hdf5, row["demo_id"], [peak])[0].to(device)  # (3,H,W)
        except Exception as e:
            print(f"Skipping {row['demo_id']}: {e}")
            continue

        gen = generate(ctx.unsqueeze(0), unet, ctx_enc, scheduler,
                       n_steps=args.steps, device=device)  # (3,H,W)

        # Each sample: (last context, ground truth, generated)
        last_ctx = denorm(ctx[-1])
        rows.append((last_ctx, denorm(tgt), denorm(gen)))
        print(f"  {row['demo_id']}  object={row['object_name']}")

    if not rows:
        print("No samples generated.")
        return

    # Landscape layout: 3 rows (context / GT / generated) × N columns (samples)
    contexts  = torch.stack([r[0] for r in rows])
    targets   = torch.stack([r[1] for r in rows])
    generated = torch.stack([r[2] for r in rows])
    grid_imgs = torch.cat([contexts, targets, generated], dim=0)  # (3*N, 3, H, W)

    grid = make_grid(grid_imgs, nrow=len(rows), padding=2)
    save_image(grid, args.out)
    print(f"\nSaved grid ({len(rows)} samples) → {args.out}")
    print("Rows: [last context frame] / [ground truth peak] / [generated peak]")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root",  default="/home/akshatat/mmai-tactile-grasp/opentouch/data")
    p.add_argument("--n",          type=int, default=8, help="number of samples")
    p.add_argument("--steps",      type=int, default=50, help="DDPM denoising steps")
    p.add_argument("--out",        default="samples.png")
    main(p.parse_args())
