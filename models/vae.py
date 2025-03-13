import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder part of the VAE that compresses images into latent representations.
    """
    def __init__(self, in_channels=3, latent_dim=512, hidden_dims=[64, 128, 256, 512]):
        """
        Initialize the encoder network.
        
        Args:
            in_channels (int): Number of input image channels
            latent_dim (int): Dimension of the latent space
            hidden_dims (list): List of hidden dimensions for each layer
        """
        super().__init__()
        
        # Create a list to hold all the encoder modules
        modules = []
        
        # Input channels for the first layer
        in_channels_layer = in_channels
        
        # Add convolutional blocks for each hidden dimension
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_layer, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels_layer = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the output dimensions after all Conv2d layers
        # For each Conv2d layer with stride=2, the spatial dimensions are halved
        # Assuming input size is divisible by 2^(len(hidden_dims))
        
        # Final fully connected layers for mu and log_var
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            tuple: (mu, log_var) for the latent distribution
        """
        # Encode the input
        result = self.encoder(x)
        
        # Flatten the result
        result = torch.flatten(result, start_dim=1)
        
        # Get mu and log_var
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        
        return mu, log_var

class Decoder(nn.Module):
    """
    Decoder part of the VAE that reconstructs images from latent representations.
    """
    def __init__(self, out_channels=3, latent_dim=512, hidden_dims=[512, 256, 128, 64]):
        """
        Initialize the decoder network.
        
        Args:
            out_channels (int): Number of output image channels
            latent_dim (int): Dimension of the latent space
            hidden_dims (list): List of hidden dimensions for each layer
        """
        super().__init__()
        
        # Initial fully connected layer
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)
        
        # Create a list to hold all the decoder modules
        modules = []
        
        # Add transposed convolutional blocks for each hidden dimension
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final layer to output the reconstructed image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()  # Sigmoid to ensure output is in [0, 1] range
        )
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, z):
        """
        Forward pass through the decoder.
        
        Args:
            z (torch.Tensor): Latent vector of shape [B, latent_dim]
            
        Returns:
            torch.Tensor: Reconstructed image
        """
        # Project and reshape from latent space to initial feature map size
        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 4)
        
        # Apply the decoder layers
        result = self.decoder(result)
        
        # Final layer to get the reconstructed image
        reconstruction = self.final_layer(result)
        
        return reconstruction

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for face encoding and reconstruction.
    """
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        """
        Initialize the VAE model.
        
        Args:
            in_channels (int): Number of input image channels
            out_channels (int): Number of output image channels
            latent_dim (int): Dimension of the latent space
        """
        super().__init__()
        
        # Define the encoder and decoder networks
        hidden_dims_encoder = [64, 128, 256, 512]
        hidden_dims_decoder = list(reversed(hidden_dims_encoder))
        
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims_encoder)
        self.decoder = Decoder(out_channels, latent_dim, hidden_dims_decoder)
        
    def encode(self, x):
        """
        Encode an image to latent representation.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            tuple: (mu, log_var) parameters of the latent distribution
        """
        return self.encoder(x)
        
    def decode(self, z):
        """
        Decode a latent representation to an image.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Reconstructed image
        """
        return self.decoder(z)
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian
            log_var (torch.Tensor): Log variance of the latent Gaussian
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            dict: Contains 'reconstruction', 'mu', 'log_var', and 'z'
        """
        # Encode the input to get mu and log_var
        mu, log_var = self.encode(x)
        
        # Sample from the latent distribution using reparameterization
        z = self.reparameterize(mu, log_var)
        
        # Decode the latent vector to get the reconstruction
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
