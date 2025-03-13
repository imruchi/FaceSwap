import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.vae import VAE
from data.celeba_dataset import CelebADataModule

class VAETrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(f"Using device: {self.device}")
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'reconstructions'), exist_ok=True)
        
        # Initialize data module
        self.data_module = CelebADataModule(
            root_dir=args.data_root,
            download=args.download,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            paired=False,  # We don't need paired data for VAE
            seed=args.seed
        )
        self.data_module.setup()
        self.dataloaders = self.data_module.get_dataloaders()
        
        # Initialize model
        self.model = VAE(
            in_channels=3,
            out_channels=3,
            latent_dim=args.latent_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Optionally initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }
        
        self.best_val_loss = float('inf')
    
    def vae_loss(self, recon_x, x, mu, log_var):
        """
        Calculate VAE loss: reconstruction loss + KL divergence
        
        Args:
            recon_x (torch.Tensor): Reconstructed image
            x (torch.Tensor): Original image
            mu (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
            
        Returns:
            tuple: (total_loss, recon_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        
        # Total loss
        total_loss = recon_loss + self.args.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, epoch):
        """
        Train the model for one epoch
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        train_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(data)
            recon_batch = outputs['reconstruction']
            mu = outputs['mu']
            log_var = outputs['log_var']
            
            loss, recon_loss, kl_loss = self.vae_loss(recon_batch, data, mu, log_var)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            })
        
        # Calculate average losses
        avg_loss = train_loss / len(self.dataloaders['train'])
        avg_recon_loss = recon_loss_sum / len(self.dataloaders['train'])
        avg_kl_loss = kl_loss_sum / len(self.dataloaders['train'])
        
        print(f"Epoch {epoch}: Train Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
        
        self.history['train_loss'].append(avg_loss)
        self.history['recon_loss'].append(avg_recon_loss)
        self.history['kl_loss'].append(avg_kl_loss)
        
        return avg_loss
    
    def validate(self, epoch):
        """
        Validate the model
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.dataloaders['val']):
                data = data.to(self.device)
                
                outputs = self.model(data)
                recon_batch = outputs['reconstruction']
                mu = outputs['mu']
                log_var = outputs['log_var']
                
                loss, _, _ = self.vae_loss(recon_batch, data, mu, log_var)
                val_loss += loss.item()
                
                # Save reconstruction example from first batch
                if batch_idx == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([
                        data[:n],
                        recon_batch[:n]
                    ])
                    save_path = os.path.join(self.args.save_dir, 'reconstructions', f'reconstruction_epoch_{epoch}.png')
                    save_image(comparison.cpu(), save_path, nrow=n, normalize=True)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(self.dataloaders['val'])
        print(f"Validation Loss: {avg_val_loss:.6f}")
        
        self.history['val_loss'].append(avg_val_loss)
        
        # Update learning rate scheduler
        self.scheduler.step(avg_val_loss)
        
        # Save model if it has best validation loss
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_model(os.path.join(self.args.save_dir, 'best_vae_model.pt'))
            print(f"Saved best model with validation loss: {avg_val_loss:.6f}")
        
        # Generate new samples from random latent vectors
        if epoch % 5 == 0:
            self.generate_samples(epoch)
        
        return avg_val_loss
    
    def train(self):
        """
        Train the model for the specified number of epochs
        """
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Save checkpoint every checkpoint_interval epochs
            if epoch % self.args.checkpoint_interval == 0:
                self.save_model(os.path.join(self.args.save_dir, f'vae_checkpoint_epoch_{epoch}.pt'))
            
            # Plot and save learning curves
            if epoch % 5 == 0:
                self.plot_learning_curves()
        
        # Save final model
        self.save_model(os.path.join(self.args.save_dir, 'vae_final_model.pt'))
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def save_model(self, path):
        """
        Save model checkpoint
        
        Args:
            path (str): Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'args': self.args,
            'latent_dim': self.args.latent_dim,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model checkpoint
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Model loaded from {path}")
    
    def plot_learning_curves(self):
        """
        Plot and save learning curves
        """
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['recon_loss'], label='Reconstruction Loss')
        plt.plot(self.history['kl_loss'], label='KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Components')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, 'learning_curves.png'))
        plt.close()
    
    def generate_samples(self, epoch):
        """
        Generate and save samples from random latent vectors
        
        Args:
            epoch (int): Current epoch number
        """
        with torch.no_grad():
            # Generate random latent vectors
            sample = torch.randn(64, self.args.latent_dim).to(self.device)
            
            # Decode latent vectors
            generated = self.model.decode(sample).cpu()
            
            # Save generated images
            save_path = os.path.join(self.args.save_dir, 'reconstructions', f'samples_epoch_{epoch}.png')
            save_image(generated, save_path, nrow=8, normalize=True)
            print(f"Generated samples saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE on CelebA dataset')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='./data', help='Data directory')
    parser.add_argument('--download', action='store_true', help='Download dataset if not available')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to use (None for all)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    
    # Model parameters
    parser.add_argument('--latent-dim', type=int, default=512, help='Dimension of latent space')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--kl-weight', type=float, default=0.0025, help='Weight for KL divergence')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Save parameters
    parser.add_argument('--save-dir', type=str, default='./checkpoints/vae', help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    
    trainer = VAETrainer(args)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_model(args.resume)
    
    trainer.train()
