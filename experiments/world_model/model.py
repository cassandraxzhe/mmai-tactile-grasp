
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler


class ContextEncoder(nn.Module):
    """
    Encode N context frames → sequence of cross-attention tokens for UNet.
    Uses a frozen ResNet18 backbone + a small projection head.
    """
    def __init__(self, n_frames=8, cross_attn_dim=512):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(512, cross_attn_dim),
            nn.ReLU(),
            nn.Linear(cross_attn_dim, cross_attn_dim),
        )
        self.n_frames = n_frames

    def forward(self, x):
        # x: (B, N, 3, H, W)
        B, N, C, H, W = x.shape
        feats = self.backbone(x.view(B*N, C, H, W))          # (B*N, 512, 1, 1)
        feats = feats.view(B, N, 512)                         # (B, N, 512)
        return self.proj(feats)                               # (B, N, cross_attn_dim)


def build_unet(img_size=64, cross_attn_dim=512):
    """Build conditional UNet for diffusion."""
    return UNet2DConditionModel(
        sample_size          = img_size,
        in_channels          = 3,
        out_channels         = 3,
        layers_per_block     = 2,
        block_out_channels   = (64, 128, 256, 256),
        down_block_types     = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types       = ("UpBlock2D", "CrossAttnUpBlock2D",
                                "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim  = cross_attn_dim,
    )


def build_scheduler(num_train_timesteps=1000):
    """Build DDPM scheduler."""
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)


@torch.no_grad()
def generate(ctx_frames_tensor, unet, ctx_enc, scheduler, n_steps=50, device="cpu"):
    # Use DDIM for inference — much better quality at low step counts
    scheduler = DDIMScheduler.from_config(scheduler.config)
    """
    Full DDPM reverse process to generate a predicted contact frame.
    ctx_frames_tensor: (1, N, 3, H, W) on device
    Returns: (3, H, W) tensor in [-1, 1]
    """
    unet.eval()
    ctx_enc.eval()
    encoder_hidden = ctx_enc(ctx_frames_tensor)
    img_size = ctx_frames_tensor.shape[-1]
    x = torch.randn(1, 3, img_size, img_size, device=device)
    scheduler.set_timesteps(n_steps)
    for t in scheduler.timesteps:
        noise_pred = unet(x, t, encoder_hidden_states=encoder_hidden).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
    return x[0]
