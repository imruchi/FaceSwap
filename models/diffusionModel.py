import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDiffusionModel(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        """
        Initialize a simple CNN-based diffusion model.
        
        Args:
            in_channels: Number of input channels (default is 3 for RGB images)
            hidden_channels: Number of hidden channels in the network
        """
        super(SimpleDiffusionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Input image tensor of shape (batch, 3, 256, 256)
            
        Returns:
            Tensor of the same shape as input representing predicted noise.
        """
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out
