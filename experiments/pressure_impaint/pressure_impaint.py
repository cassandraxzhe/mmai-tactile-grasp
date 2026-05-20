import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PRESSURE_MAX = 3072.0


# 11 fingertip cells in the OpenTouch 16x16 grid.
FINGERTIP_CELLS = [
    (13, 14), (13, 15), (14, 13),  # thumb tip
    (0, 10),  (0, 11),             # index tip
    (0, 6),   (1, 6),              # middle tip
    (0, 5),   (1, 5),              # ring tip
    (0, 1),   (1, 1),              # pinky tip
]


def create_fingertip_mask(shape=(16, 16), fingertip_cells=FINGERTIP_CELLS):
    mask = np.zeros(shape, dtype=np.float32)
    for (r, c) in fingertip_cells:
        mask[r, c] = 1.0
    return mask


class InpaintPoseEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class InpaintVisualEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PressureInpaintNet(nn.Module):
    def __init__(self, visual_dim=128, pose_dim=64):
        super().__init__()
        self.visual_enc = InpaintVisualEncoder(out_dim=visual_dim)
        self.pose_enc   = InpaintPoseEncoder(out_dim=pose_dim)

        cond_dim = visual_dim + pose_dim  # 192

        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )

        self.cond_proj = nn.Linear(cond_dim, 128)

        self.dec3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, pressure_masked, mask, rgb, landmarks):
        """
        pressure_masked: (B, 16, 16) — sensor readings × mask
        mask:            (B, 16, 16) — 1.0 at observed cells, 0 elsewhere
        rgb:             (B, 3, 224, 224)
        landmarks:       (B, 63) — flattened (21, 3)
        Returns:         (B, 16, 16) refined pressure map (ReLU clamped)
        """
        vis_feat  = self.visual_enc(rgb)
        pose_feat = self.pose_enc(landmarks)
        cond      = torch.cat([vis_feat, pose_feat], dim=1)
        cond_vec  = self.cond_proj(cond)

        x  = torch.stack([pressure_masked, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        cond_spatial = cond_vec[:, :, None, None].expand_as(e3)
        e3 = e3 + cond_spatial

        d3  = F.interpolate(self.dec3(e3), scale_factor=2)
        d2  = F.interpolate(self.dec2(torch.cat([d3, e2], 1)), scale_factor=2)
        d1  = self.dec1(torch.cat([d2, e1], 1))
        out = self.out_conv(d1).squeeze(1)

        return F.relu(out)


def compute_loss(pred, target, mask):
    mse        = F.mse_loss(pred, target)
    unobserved = (1 - mask)
    mae_unobs  = (torch.abs(pred - target) * unobserved).sum() / (unobserved.sum() + 1e-6)
    return mse + 0.5 * mae_unobs


def compute_metrics(pred, target, mask):
    unobserved = (1 - mask)
    mae_unobs  = (torch.abs(pred - target) * unobserved).sum() / (unobserved.sum() + 1e-6)

    pred_contact   = (pred > 0.5).float()
    target_contact = (target > 0.5).float()

    tp = (pred_contact * target_contact * unobserved).sum()
    fp = (pred_contact * (1 - target_contact) * unobserved).sum()
    fn = ((1 - pred_contact) * target_contact * unobserved).sum()

    precision   = tp / (tp + fp + 1e-6)
    recall      = tp / (tp + fn + 1e-6)
    f1          = 2 * precision * recall / (precision + recall + 1e-6)
    contact_acc = ((pred_contact == target_contact) * unobserved).sum() / (unobserved.sum() + 1e-6)

    return {
        'mae_unobserved': mae_unobs.item(),
        'f1_unobserved':  f1.item(),
        'precision':      precision.item(),
        'recall':         recall.item(),
        'contact_acc':    contact_acc.item(),
    }


# ─── Training script — only runs when invoked directly ──────────────────────
if __name__ == "__main__":
    import argparse
    import glob
    from collections import defaultdict

    import h5py
    import matplotlib.pyplot as plt
    import wandb
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    p = argparse.ArgumentParser()
    p.add_argument(
        "--hdf5-glob",
        default="/orcd/data/edboyden/002/edward77/touch/opentouch/data/*.hdf5",
    )
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--wandb-project", default="opentouch")
    p.add_argument(
        "--wandb-entity",
        default="edwardchen5414-massachusetts-institute-of-technology",
    )
    p.add_argument("--wandb-run-id", default=None)
    p.add_argument("--no-wandb",     action="store_true")
    args = p.parse_args()


    class PressureInpaintDataset(Dataset):
        def __init__(self, hdf5_paths, img_size=224, min_total_pressure=0.5):
            self.samples = []
            self.img_size = img_size
            self.mask = create_fingertip_mask()

            for path in hdf5_paths:
                with h5py.File(path, 'r') as f:
                    for demo_id in f['data'].keys():
                        demo = f['data'][demo_id]
                        pressures = demo['right_pressure'][:]
                        for i in range(len(pressures)):
                            if (pressures[i] / PRESSURE_MAX).sum() < min_total_pressure:
                                continue
                            self.samples.append({
                                'hdf5_path': path,
                                'demo_id':   demo_id,
                                'frame_idx': i,
                            })
            print(f"Dataset: {len(self.samples)} frames with contact")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            with h5py.File(s['hdf5_path'], 'r') as f:
                demo = f['data'][s['demo_id']]
                i    = s['frame_idx']

                pressure_gt = torch.tensor(
                    demo['right_pressure'][i], dtype=torch.float32
                ) / PRESSURE_MAX

                mask            = torch.tensor(self.mask, dtype=torch.float32)
                pressure_masked = pressure_gt * mask

                jpeg_bytes = bytes(demo['rgb_images_jpeg'][i])
                img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
                img = img.resize((self.img_size, self.img_size))
                img_tensor = torch.tensor(
                    np.array(img) / 255.0, dtype=torch.float32
                ).permute(2, 0, 1)

                landmarks = torch.tensor(
                    demo['right_hand_landmarks'][i].flatten(), dtype=torch.float32
                )

            return {
                'pressure_gt':     pressure_gt,
                'pressure_masked': pressure_masked,
                'mask':            mask,
                'rgb':             img_tensor,
                'landmarks':       landmarks,
            }


    if not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config={
                "learning_rate": args.lr,
                "architecture":  "PressureInpaintNet",
                "batch_size":    args.batch_size,
                "epochs":        args.epochs,
                "pressure_max":  PRESSURE_MAX,
                "optimizer":     "Adam",
                "scheduler":     "CosineAnnealingLR",
            },
        )

    hdf5_files = sorted(glob.glob(args.hdf5_glob))
    print(f"Found {len(hdf5_files)} HDF5 files")
    n_train     = int(0.8 * len(hdf5_files))
    train_files = hdf5_files[:n_train]
    val_files   = hdf5_files[n_train:]
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    train_ds = PressureInpaintDataset(train_files)
    val_ds   = PressureInpaintDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model     = PressureInpaintNet().to(device)
    optim_    = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0
    checkpoint_files = sorted(glob.glob("checkpoint_epoch*.pt"))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        print(f"Found checkpoint: {latest_ckpt}. Resuming...")
        ck = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ck['model_state'])
        optim_.load_state_dict(ck['optim_state'])
        start_epoch = ck['epoch']
        if 'scheduler_state' in ck:
            scheduler.load_state_dict(ck['scheduler_state'])
        else:
            for _ in range(start_epoch):
                scheduler.step()
    else:
        print("No local checkpoints found. Starting from scratch.")

    fixed_val = {k: v.to(device) for k, v in next(iter(val_loader)).items()}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pm  = batch['pressure_masked'].to(device)
            gt  = batch['pressure_gt'].to(device)
            msk = batch['mask'].to(device)
            rgb = batch['rgb'].to(device)
            lmk = batch['landmarks'].to(device)
            pred = model(pm, msk, rgb, lmk)
            loss = compute_loss(pred, gt, msk)
            optim_.zero_grad()
            loss.backward()
            optim_.step()
            train_loss += loss.item()

        model.eval()
        val_loss    = 0.0
        all_metrics = defaultdict(float)
        with torch.no_grad():
            for batch in val_loader:
                pm  = batch['pressure_masked'].to(device)
                gt  = batch['pressure_gt'].to(device)
                msk = batch['mask'].to(device)
                rgb = batch['rgb'].to(device)
                lmk = batch['landmarks'].to(device)
                pred = model(pm, msk, rgb, lmk)
                val_loss += compute_loss(pred, gt, msk).item()
                m = compute_metrics(pred, gt, msk)
                for k, v in m.items():
                    all_metrics[k] += v
        scheduler.step()

        avg_train  = train_loss / len(train_loader)
        avg_val    = val_loss   / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train: {avg_train:.4f} | val: {avg_val:.4f} | lr: {current_lr:.2e}")

        metrics = {
            "epoch":      epoch + 1,
            "train/loss": avg_train,
            "val/loss":   avg_val,
            "train/lr":   current_lr,
        }
        for k, v in all_metrics.items():
            metrics[f'val/{k}'] = v / len(val_loader)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_viz = model(
                    fixed_val['pressure_masked'],
                    fixed_val['mask'],
                    fixed_val['rgb'],
                    fixed_val['landmarks'],
                )
            n = 4
            fig, axes = plt.subplots(n, 4, figsize=(12, n * 3))
            for i in range(n):
                gt_np   = fixed_val['pressure_gt'][i].cpu().numpy()
                pm_np   = fixed_val['pressure_masked'][i].cpu().numpy()
                pred_np = pred_viz[i].cpu().numpy()
                err_np  = np.abs(pred_np - gt_np)
                vmax = max(gt_np.max(), 0.01)
                axes[i, 0].imshow(gt_np,   cmap='hot',  vmin=0, vmax=vmax)
                axes[i, 1].imshow(pm_np,   cmap='hot',  vmin=0, vmax=vmax)
                axes[i, 2].imshow(pred_np, cmap='hot',  vmin=0, vmax=vmax)
                axes[i, 3].imshow(err_np,  cmap='Reds', vmin=0, vmax=vmax)
                for ax in axes[i]:
                    ax.axis('off')
            for ax, title in zip(axes[0], ['GT', 'Masked', 'Pred', 'Err']):
                ax.set_title(title, fontsize=12)
            plt.suptitle(f'Epoch {epoch + 1}', fontsize=14)
            plt.tight_layout()
            if not args.no_wandb:
                metrics['visuals/comparison'] = wandb.Image(fig)
            plt.close(fig)

            ckpt_path = f"checkpoint_epoch{epoch+1:03d}.pt"
            torch.save({
                'epoch':           epoch + 1,
                'model_state':     model.state_dict(),
                'optim_state':     optim_.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_loss':        avg_val,
            }, ckpt_path)
            if not args.no_wandb:
                wandb.save(ckpt_path)

        if not args.no_wandb:
            wandb.log(metrics)

    if not args.no_wandb:
        wandb.finish()
