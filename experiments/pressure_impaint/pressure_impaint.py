import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import glob
import torch
from torch.utils.data import Dataset
import wandb
import os

os.environ["WANDB_API_KEY"] = ""

PRESSURE_MAX = 3072.0

run = wandb.init(
    entity="edwardchen5414-massachusetts-institute-of-technology",
    project="opentouch",
    config={
        "learning_rate": 3e-4,
        "architecture": "PressureInpaintNet",
        "batch_size": 64,
        "epochs": 50,
        "loss_mae_weight": 0.5,
        "optimizer": "Adam"
    }
)

files = glob.glob('/home/edward77/81/touch/opentouch/data/*.hdf5')
print(f"Found {len(files)} HDF5 files:")
for f in sorted(files):
    print(f"  {f}")

all_pressure = []

for fpath in sorted(files):
    f = h5py.File(fpath, 'r')
    demos = [k for k in f['data'].keys() if k.startswith('demo')]
    print(f"{fpath}: {len(demos)} demos")
    for d in demos:
        all_pressure.append(f['data'][d]['right_pressure'][:])
    f.close()

all_pressure = np.concatenate(all_pressure, axis=0)
print(f"\nTotal frames across all files: {all_pressure.shape[0]}")

FINGERTIP_CELLS = [
    (13, 14),   # thumb tip
    (13, 15),   # thumb tip
    (14, 13),   # thumb tip
    (0, 10),   # index tip
    (0, 11),   # index tip
    (0, 6),   # middle tip
    (1, 6),   # middle tip
    (0, 5),   # ring tip
    (1, 5),   # ring tip
    (0, 1),  # pinky tip
    (1, 1),  # pinky tip
]

print("Fingertip cell statistics:")
for (r, c) in FINGERTIP_CELLS:
    vals = all_pressure[:, r, c]
    print(f"  ({r},{c})  mean={vals.mean():.0f}  std={vals.std():.0f}  "
          f"  max={vals.max():.0f}  nonzero={( vals > 0).mean():.2%}")

def create_fingertip_mask(shape=(16,16), fingertip_cells=FINGERTIP_CELLS):
    mask = np.zeros(shape, dtype=np.float32)
    for (r, c) in fingertip_cells:
        mask[r, c] = 1.0
    return mask

class PressureInpaintDataset(Dataset):
    def __init__(self, hdf5_paths, img_size=224, min_total_pressure=0.5):
        self.samples = []
        self.img_size = img_size
        self.mask = create_fingertip_mask()

        for path in hdf5_paths:
            f = h5py.File(path, 'r')
            for demo_id in f['data'].keys():
                demo    = f['data'][demo_id]
                pressures   = demo['right_pressure'][:]    # (N, 16, 16)
                landmarks   = demo['right_hand_landmarks'][:]  
                rgb_frames  = demo['rgb_images_jpeg']     

                for i in range(len(pressures)):
                    if pressures[i].sum() < min_total_pressure:
                        continue
                    self.samples.append({
                        'hdf5_path': path,
                        'demo_id':   demo_id,
                        'frame_idx': i,
                    })
            f.close()

        print(f"Dataset: {len(self.samples)} frames with contact")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        f = h5py.File(s['hdf5_path'], 'r')
        demo = f['data'][s['demo_id']]
        i    = s['frame_idx']

        # Full pressure map 
        pressure_gt = torch.tensor(
            demo['right_pressure'][i], dtype=torch.float32
        ) / PRESSURE_MAX
        pressure_gt = pressure_gt.clamp(0, 1) 

        # Masked input 
        mask = torch.tensor(self.mask, dtype=torch.float32)
        pressure_masked = pressure_gt * mask 

        # RGB frame
        jpeg_bytes = bytes(demo['rgb_images_jpeg'][i])
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_tensor = torch.tensor(
            np.array(img) / 255.0, dtype=torch.float32
        ).permute(2, 0, 1) 

        # Hand landmarks — flatten to 1D vector
        landmarks = torch.tensor(
            demo['right_hand_landmarks'][i].flatten(), dtype=torch.float32
        ) 

        f.close()
        return {
            'pressure_gt':     pressure_gt,    
            'pressure_masked': pressure_masked, 
            'mask':            mask,            
            'rgb':             img_tensor,      
            'landmarks':       landmarks,       
        }

import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class VisualEncoder(nn.Module):
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
    """
    Input: the 16x16 masked pressure map + visual features + pose features
    Output: the 16x16 full pressure map 
    """
    def __init__(self, visual_dim=128, pose_dim=64):
        super().__init__()
        self.visual_enc = VisualEncoder(out_dim=visual_dim)
        self.pose_enc = PoseEncoder(out_dim=pose_dim)

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

        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, pressure_masked, mask, rgb, landmarks):
        vis_feat = self.visual_enc(rgb)
        pose_feat = self.pose_enc(landmarks)
        cond = torch.cat([vis_feat, pose_feat], dim=1)
        cond_vec = self.cond_proj(cond)

        x = torch.stack([pressure_masked, mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        cond_spatial = cond_vec[:, :, None, None].expand_as(e3)
        e3 = e3 + cond_spatial

        d3 = F.interpolate(self.dec3(e3), scale_factor=2)   
        d2 = F.interpolate(self.dec2(torch.cat([d3, e2], 1)), scale_factor=2)  
        d1 = self.dec1(torch.cat([d2, e1], 1))          
    
        out = self.out_conv(d1).squeeze(1)      
        return F.relu(out) 

from torch.utils.data import DataLoader, random_split
import glob

# Find all HDF5 files
hdf5_files = glob.glob('./opentouch/data/*.hdf5')
print(f"Found {len(hdf5_files)} HDF5 files")

# Build dataset
dataset = PressureInpaintDataset(hdf5_files)

# Train-Test Split
n_train = int(0.8 * len(hdf5_files))
train_files = hdf5_files[:n_train]
val_files = hdf5_files[n_train:]
train_ds = PressureInpaintDataset(train_files)
val_ds = PressureInpaintDataset(val_files)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PressureInpaintNet().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

# Add cosine annealing so LR decays gracefully
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=50, eta_min=1e-6
)


def get_viz_batch(loader, device):
    """Always grab the same batch for consistent epoch-to-epoch comparison."""
    batch = next(iter(loader))
    return {k: v.to(device) for k, v in batch.items()}
 
fixed_val = get_viz_batch(val_loader, device)

def compute_loss(pred, target, mask):
    mse = F.mse_loss(pred, target)

    unobserved = (1-mask)
    mae_unobserved = (torch.abs(pred-target) * unobserved).sum() / (unobserved.sum() + 1e-6)
    return mse + 0.5 * mae_unobserved 

for epoch in range(50):
    model.train()
    train_loss = 0
    for batch in train_loader:
        pm  = batch['pressure_masked'].to(device)
        gt  = batch['pressure_gt'].to(device)
        msk = batch['mask'].to(device)
        rgb = batch['rgb'].to(device)
        lmk = batch['landmarks'].to(device)

        pred = model(pm, msk, rgb, lmk)
        loss = compute_loss(pred, gt, msk)

        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pm  = batch['pressure_masked'].to(device)
            gt  = batch['pressure_gt'].to(device)
            msk = batch['mask'].to(device)
            rgb = batch['rgb'].to(device)
            lmk = batch['landmarks'].to(device)
            pred = model(pm, msk, rgb, lmk)
            val_loss += compute_loss(pred, gt, msk).item()

    scheduler.step()

    # print(f"Epoch {epoch+1:3d} | train loss: {train_loss/len(train_loader):.4f} | val loss: {val_loss/len(val_loader):.4f}")

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    metrics = {
        "epoch": epoch + 1,
        "train/loss": avg_train_loss,
        "val/loss": avg_val_loss
    }

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            pred_viz = model(
                fixed_val['pressure_masked'],
                fixed_val['mask'],
                fixed_val['rgb'],
                fixed_val['landmarks'],
            )
 
        idx = 0  
        gt_np   = fixed_val['pressure_gt'][idx].cpu().numpy()
        pm_np   = fixed_val['pressure_masked'][idx].cpu().numpy()
        pred_np = pred_viz[idx].cpu().numpy()
 
        vmax = max(gt_np.max(), 0.01)
 
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, data, title in zip(
            axes,
            [gt_np, pm_np, pred_np],
            ['Ground Truth', 'Masked Input', 'Prediction']
        ):
            ax.imshow(data, cmap='hot', vmin=0, vmax=vmax)
            ax.set_title(title)
            ax.axis('off')
 
        plt.suptitle(f'Epoch {epoch + 1}')
        plt.tight_layout()
 
        metrics["visuals/comparison"] = wandb.Image(
            fig, caption="GT | Masked Input | Prediction"
        )
        plt.close(fig)
 
        # Save checkpoint
        ckpt_path = f"checkpoint_epoch{epoch+1:03d}.pt"
        torch.save({
            'epoch':      epoch + 1,
            'model_state': model.state_dict(),
            'optim_state': optim.state_dict(),
            'val_loss':   avg_val,
        }, ckpt_path)
        wandb.save(ckpt_path)

    wandb.log(metrics)
    print(f"Epoch {epoch+1:3d} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")

wandb.finish()