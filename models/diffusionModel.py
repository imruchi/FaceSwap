import torch
import torch.nn as nn
from diffusers import UNet2DModel

class DiffusionUNet(nn.Module):
    """
    Wraps the UNet2DModel from Hugging Face diffusers,
    which includes attention blocks in the down/up sampling steps.
    """
    def __init__(self, image_size=256, in_channels=3, out_channels=3):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=image_size,    # Target image resolution
            in_channels=in_channels,   # Input channels (3 for RGB)
            out_channels=out_channels, # Output channels
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=(
                "DownBlock2D",      # Standard downsampling block
                "DownBlock2D",
                "AttnDownBlock2D",  # Downsampling block with self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",    # Upsampling block with self-attention
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, timestep):
        """
        Forward pass through the U-Net.
        
        Args:
            x (torch.Tensor): Noisy input images (B, C, H, W)
            timestep (torch.LongTensor): Diffusion timesteps (B,)
        
        Returns:
            torch.Tensor: Predicted noise (same shape as x)
        """
        return self.unet(x, timestep).sample
