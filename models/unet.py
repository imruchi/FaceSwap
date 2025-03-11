import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Replaces sinusoidal embeddings with a learnable MLP-based embedding"""
    def __init__(self, time_emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(1, time_emb_dim)
        self.fc2 = nn.Linear(time_emb_dim, time_emb_dim)
        self.activation = nn.SiLU()

    def forward(self, t):
        t = t.view(-1, 1)  # Ensure shape is [batch_size, 1]
        t = self.fc1(t)
        t = self.activation(t)
        t = self.fc2(t)
        return t


class ResidualBlock(nn.Module):
    """A basic residual block with GroupNorm and Swish activation."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.activation(self.norm1(x))
        h = self.conv1(h)

        t_emb = self.activation(self.time_emb(t))[:, :, None, None]
        h = h + t_emb

        h = self.activation(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Downsampling layer using strided convolution."""
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    """Upsampling layer using transposed convolution."""
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    """U-Net architecture for diffusion models."""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()

        self.time_mlp = TimeEmbedding(time_emb_dim)  # Replace sinusoidal embedding

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.downsample1 = Downsample(base_channels * 2)

        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.downsample2 = Downsample(base_channels * 4)

        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.upsample1 = Upsample(base_channels * 4)
        self.up1 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.upsample2 = Upsample(base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)  # Uses learned time embeddings

        x = self.init_conv(x)

        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        d2 = self.downsample1(d2)

        d3 = self.down3(d2, t_emb)
        d3 = self.downsample2(d3)

        bottleneck = self.bottleneck(d3, t_emb)

        u1 = self.upsample1(bottleneck)
        u1 = self.up1(u1 + d3, t_emb)

        u2 = self.upsample2(u1)
        u2 = self.up2(u2 + d2, t_emb)

        out = self.final_conv(u2 + d1)

        return out


# Test the model
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=128)
    x = torch.randn(4, 3, 64, 64)  # Batch of 4, 3 channels, 64x64 images
    t = torch.randint(0, 1000, (4,)).float()  # Random timesteps
    y = model(x, t)
    print("Output shape:", y.shape)  # Should be [4, 3, 64, 64]
