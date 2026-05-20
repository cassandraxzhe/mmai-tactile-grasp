import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.pressure_impaint.pressure_impaint import (
    InpaintPoseEncoder,
    InpaintVisualEncoder,
    PRESSURE_MAX,
    create_fingertip_mask,
)


CONTACT_EPS = 0.01       
SAT_THRESH  = 0.95       
BIN_EDGES   = [0.0, CONTACT_EPS, 0.33, SAT_THRESH, 1.01]
BIN_NAMES   = ['zero', 'light', 'moderate', 'saturated']


class PressureInpaintNetDualHead(nn.Module):
    def __init__(self, visual_dim=128, pose_dim=64):
        super().__init__()
        self.visual_enc = InpaintVisualEncoder(out_dim=visual_dim)
        self.pose_enc   = InpaintPoseEncoder(out_dim=pose_dim)
        cond_dim = visual_dim + pose_dim

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

        self.out_contact   = nn.Conv2d(16, 1, 1)
        self.out_intensity = nn.Conv2d(16, 1, 1)

    def forward(self, pressure_masked, mask, rgb, landmarks, detach_contact=False):
        vis_feat  = self.visual_enc(rgb)
        pose_feat = self.pose_enc(landmarks)
        cond_vec  = self.cond_proj(torch.cat([vis_feat, pose_feat], dim=1))

        x  = torch.stack([pressure_masked, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2) + cond_vec[:, :, None, None]

        d3 = F.interpolate(self.dec3(e3),                     scale_factor=2)
        d2 = F.interpolate(self.dec2(torch.cat([d3, e2], 1)), scale_factor=2)
        d1 = self.dec1(torch.cat([d2, e1], 1))

        contact_in     = d1.detach() if detach_contact else d1
        contact_logits = self.out_contact(contact_in).squeeze(1)
        intensity      = F.relu(self.out_intensity(d1).squeeze(1))
        return contact_logits, intensity


def predict(contact_logits, intensity):
    return (torch.sigmoid(contact_logits) * intensity).clamp(0.0, 1.0)


def compute_loss_dual(contact_logits, intensity, target, mask,
                      contact_weight=1.0, intensity_weight=1.0,
                      mask_saturated=False, saturated_weight=1.0):
    contact_target = (target > CONTACT_EPS).float()
    bce = F.binary_cross_entropy_with_logits(contact_logits, contact_target)

    int_mask = contact_target
    if mask_saturated:
        int_mask = int_mask * (target < SAT_THRESH).float()
    if saturated_weight != 1.0:
        cell_w = torch.where(
            target >= SAT_THRESH,
            torch.full_like(target, saturated_weight),
            torch.ones_like(target),
        )
        int_mask = int_mask * cell_w
    int_sq   = (intensity - target) ** 2
    int_loss = (int_sq * int_mask).sum() / (int_mask.sum() + 1e-6)

    total = contact_weight * bce + intensity_weight * int_loss
    return total, {'bce': bce.item(), 'intensity_mse': int_loss.item()}


def compute_metrics_dual(contact_logits, intensity, target, mask):
    contact_prob = torch.sigmoid(contact_logits)
    pred         = predict(contact_logits, intensity)
    unobs        = (1 - mask)

    target_contact = (target > CONTACT_EPS).float()
    pred_contact   = (contact_prob > 0.5).float()

    tp = (pred_contact * target_contact * unobs).sum()
    fp = (pred_contact * (1 - target_contact) * unobs).sum()
    fn = ((1 - pred_contact) * target_contact * unobs).sum()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    abs_err   = (pred - target).abs()
    unobs_n   = unobs.sum().item() + 1e-6
    out = {
        'mae_unobs':       (abs_err * unobs).sum().item() / unobs_n,
        'precision':       precision.item(),
        'recall':          recall.item(),
        'f1':              f1.item(),
        'unobs_err_sum':   (abs_err * unobs).sum().item(),
        'unobs_count':     unobs.sum().item(),
    }
    for name, lo, hi in zip(BIN_NAMES, BIN_EDGES[:-1], BIN_EDGES[1:]):
        in_bin = ((target >= lo) & (target < hi)).float() * unobs
        out[f'err_sum_{name}'] = (abs_err * in_bin).sum().item()
        out[f'count_{name}']   = in_bin.sum().item()
    return out, pred

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
    p.add_argument("--hdf5-glob",
        default="/orcd/data/edboyden/002/edward77/touch/opentouch/data/*.hdf5")
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--contact-weight",   type=float, default=1.0)
    p.add_argument("--intensity-weight", type=float, default=1.0)
    p.add_argument("--mask-saturated",   action="store_true",
        help=f"Exclude cells with GT > {SAT_THRESH} from intensity loss.")
    p.add_argument("--saturated-weight", type=float, default=1.0,
        help=f"Multiplier on intensity loss for cells with GT >= {SAT_THRESH}. "
             "Use 3-5 to push the head harder on saturated cells. "
             "Has no effect when --mask-saturated is set.")
    p.add_argument("--intensity-lr",     type=float, default=None,
        help="Separate LR for the intensity head (out_intensity). "
             "Defaults to --lr. Try lr*0.1 if the intensity MAE curves oscillate.")
    p.add_argument("--detach-contact-after-epoch", type=int, default=-1,
        help="After this epoch (0-indexed), detach the contact head from the "
             "shared decoder so BCE no longer pulls the encoder. -1 disables.")
    p.add_argument("--ckpt-dir",         default="./checkpoints_dualhead")
    p.add_argument("--ckpt-every",       type=int,   default=5)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--wandb-project",    default="opentouch")
    p.add_argument("--wandb-entity",
        default="edwardchen5414-massachusetts-institute-of-technology")
    p.add_argument("--wandb-run-id",     default=None,
        help="Resume an existing wandb run by id.")
    p.add_argument("--run-name",         default="dualhead")
    p.add_argument("--no-wandb",         action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)


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
                    demo['right_hand_landmarks'][i].flatten(),
                    dtype=torch.float32,
                )
            return {
                'pressure_gt':     pressure_gt,
                'pressure_masked': pressure_masked,
                'mask':            mask,
                'rgb':             img_tensor,
                'landmarks':       landmarks,
            }


    intensity_lr = args.intensity_lr if args.intensity_lr is not None else args.lr

    if not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_run_id,
            name=args.run_name,
            resume="allow" if args.wandb_run_id else None,
            config={
                "learning_rate":    args.lr,
                "intensity_lr":     intensity_lr,
                "architecture":     "PressureInpaintNetDualHead",
                "batch_size":       args.batch_size,
                "epochs":           args.epochs,
                "pressure_max":     PRESSURE_MAX,
                "contact_weight":   args.contact_weight,
                "intensity_weight": args.intensity_weight,
                "mask_saturated":   args.mask_saturated,
                "saturated_weight": args.saturated_weight,
                "detach_contact_after_epoch": args.detach_contact_after_epoch,
                "contact_eps":      CONTACT_EPS,
                "sat_thresh":       SAT_THRESH,
                "seed":             args.seed,
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
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model     = PressureInpaintNetDualHead().to(device)
    intensity_params    = list(model.out_intensity.parameters())
    intensity_param_ids = {id(p) for p in intensity_params}
    other_params        = [p for p in model.parameters()
                           if id(p) not in intensity_param_ids]
    optim_    = torch.optim.Adam([
        {'params': other_params,     'lr': args.lr},
        {'params': intensity_params, 'lr': intensity_lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_, T_max=args.epochs, eta_min=1e-6,
    )

    start_epoch   = 0
    best_val_loss = float('inf')
    latest_path   = os.path.join(args.ckpt_dir, "latest.pt")
    if os.path.exists(latest_path):
        print(f"Resuming from {latest_path}")
        ck = torch.load(latest_path, map_location=device)
        model.load_state_dict(ck['model_state'])
        start_epoch   = ck['epoch']
        best_val_loss = ck.get('best_val_loss', float('inf'))
        try:
            optim_.load_state_dict(ck['optim_state'])
            scheduler.load_state_dict(ck['scheduler_state'])
        except ValueError as e:
            print(f"WARN: optimizer/scheduler state incompatible ({e}). "
                  f"Continuing with fresh optimizer; advancing scheduler "
                  f"to epoch {start_epoch}.")
            for _ in range(start_epoch):
                scheduler.step()
    else:
        print("Starting from scratch.")

    fixed_val = {k: v.to(device) for k, v in next(iter(val_loader)).items()}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = train_bce = train_int = 0.0
        detach_contact = (
            args.detach_contact_after_epoch >= 0
            and epoch >= args.detach_contact_after_epoch
        )
        for batch in train_loader:
            pm  = batch['pressure_masked'].to(device)
            gt  = batch['pressure_gt'].to(device)
            msk = batch['mask'].to(device)
            rgb = batch['rgb'].to(device)
            lmk = batch['landmarks'].to(device)
            cl, it = model(pm, msk, rgb, lmk, detach_contact=detach_contact)
            loss, parts = compute_loss_dual(
                cl, it, gt, msk,
                contact_weight=args.contact_weight,
                intensity_weight=args.intensity_weight,
                mask_saturated=args.mask_saturated,
                saturated_weight=args.saturated_weight,
            )
            optim_.zero_grad()
            loss.backward()
            optim_.step()
            train_loss += loss.item()
            train_bce  += parts['bce']
            train_int  += parts['intensity_mse']

        model.eval()
        val_loss   = 0.0
        agg        = defaultdict(float)
        with torch.no_grad():
            for batch in val_loader:
                pm  = batch['pressure_masked'].to(device)
                gt  = batch['pressure_gt'].to(device)
                msk = batch['mask'].to(device)
                rgb = batch['rgb'].to(device)
                lmk = batch['landmarks'].to(device)
                cl, it = model(pm, msk, rgb, lmk)
                loss, _ = compute_loss_dual(
                    cl, it, gt, msk,
                    contact_weight=args.contact_weight,
                    intensity_weight=args.intensity_weight,
                    mask_saturated=args.mask_saturated,
                    saturated_weight=args.saturated_weight,
                )
                val_loss += loss.item()
                m, _ = compute_metrics_dual(cl, it, gt, msk)
                for k, v in m.items():
                    agg[k] += v

        scheduler.step()
        n_train_b  = len(train_loader)
        n_val_b    = len(val_loader)
        avg_train  = train_loss / n_train_b
        avg_val    = val_loss   / n_val_b
        current_lr = scheduler.get_last_lr()[0]

        # cell-weighted overall MAE
        mae_unobs_global = agg['unobs_err_sum'] / max(agg['unobs_count'], 1.0)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train: {avg_train:.4f} (bce {train_bce/n_train_b:.4f}, "
              f"int {train_int/n_train_b:.4f}) | "
              f"val: {avg_val:.4f} | mae(unobs): {mae_unobs_global:.4f} | "
              f"lr: {current_lr:.2e}")

        metrics = {
            "epoch":               epoch + 1,
            "train/loss":          avg_train,
            "train/bce":           train_bce / n_train_b,
            "train/intensity_mse": train_int / n_train_b,
            "val/loss":            avg_val,
            "train/lr":            current_lr,
            "val/mae_unobs":       mae_unobs_global,
            "val/precision":       agg['precision']     / n_val_b,
            "val/recall":          agg['recall']        / n_val_b,
            "val/f1":              agg['f1']            / n_val_b,
        }
        for name in BIN_NAMES:
            cnt = agg[f'count_{name}']
            metrics[f'val/mae_{name}']   = (agg[f'err_sum_{name}'] / cnt) if cnt > 0 else float('nan')
            metrics[f'val/count_{name}'] = cnt

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                cl, it = model(
                    fixed_val['pressure_masked'], fixed_val['mask'],
                    fixed_val['rgb'],             fixed_val['landmarks'],
                )
                cp = torch.sigmoid(cl)
                pv = predict(cl, it)

            n = 4
            fig, axes = plt.subplots(n, 5, figsize=(15, n * 3))
            for i in range(n):
                gt_np  = fixed_val['pressure_gt'][i].cpu().numpy()
                pm_np  = fixed_val['pressure_masked'][i].cpu().numpy()
                cp_np  = cp[i].cpu().numpy()
                pv_np  = pv[i].cpu().numpy()
                err_np = np.abs(pv_np - gt_np)
                vmax   = max(gt_np.max(), 0.01)
                axes[i, 0].imshow(gt_np,  cmap='hot',  vmin=0, vmax=vmax)
                axes[i, 1].imshow(pm_np,  cmap='hot',  vmin=0, vmax=vmax)
                axes[i, 2].imshow(cp_np,  cmap='gray', vmin=0, vmax=1)
                axes[i, 3].imshow(pv_np,  cmap='hot',  vmin=0, vmax=vmax)
                axes[i, 4].imshow(err_np, cmap='Reds', vmin=0, vmax=vmax)
                for ax in axes[i]:
                    ax.axis('off')
            for ax, t in zip(axes[0], ['GT', 'Masked', 'Contact p', 'Pred', 'Err']):
                ax.set_title(t, fontsize=12)
            plt.suptitle(f'Epoch {epoch + 1}', fontsize=14)
            plt.tight_layout()
            if not args.no_wandb:
                metrics['visuals/comparison'] = wandb.Image(fig)
            plt.close(fig)

            ckpt = {
                'epoch':           epoch + 1,
                'model_state':     model.state_dict(),
                'optim_state':     optim_.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_loss':        avg_val,
                'best_val_loss':   best_val_loss,
                'args':            vars(args),
            }
            torch.save(ckpt, os.path.join(args.ckpt_dir, f"epoch{epoch+1:03d}.pt"))
            torch.save(ckpt, latest_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'val_loss':    avg_val,
                'args':        vars(args),
            }, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"  -> new best val loss: {best_val_loss:.4f}")

        if not args.no_wandb:
            wandb.log(metrics)

    if not args.no_wandb:
        wandb.finish()
