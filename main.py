import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import CelebA
from torch.utils.data import Subset, random_split, DataLoader
from models.diffusionModel import DiffusionUNet
from utils.training import train_epoch, validate_epoch
from torchvision import transforms

def main():
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Define transformations: resize, convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load the aligned CelebA dataset directly from torchvision
    celeba_full = CelebA(root='./data', split='all', target_type='attr', transform=transform, download=True)
    print(f"Total CelebA images available: {len(celeba_full)}")
    
    # Use a smaller subset for quick testing
    num_samples = 350
    celeba_subset = Subset(celeba_full, list(range(num_samples)))
    print(f"Using subset of {num_samples} images.")
    
    # Split the subset into train (70%), validation (15%), and test (15%) sets
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(celeba_subset, [train_size, val_size, test_size])
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize the diffusion model (U-Net with self-attention) and move to device
    model = DiffusionUNet(image_size=256, in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    num_epochs = 10
    train_losses = []
    val_losses = []
    
    # Training loop over multiple epochs
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = validate_epoch(model, val_loader, device, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    # Evaluate on the test set after training
    test_loss = validate_epoch(model, test_loader, device, loss_fn)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training and validation loss curves
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()
    
    # Visualize a batch: display original, noisy, and reconstructed images
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)
        noise = torch.randn_like(images).to(device)
        alpha = 0.9
        alpha_tensor = torch.tensor(alpha, device=device)
        sqrt_alpha = torch.sqrt(alpha_tensor)
        sqrt_one_minus_alpha = torch.sqrt(torch.tensor(1.0, device=device) - alpha_tensor)
        
        # Sample random timesteps for this batch
        timesteps = torch.randint(0, 1000, (images.size(0),), device=device).long()
        
        # Add noise to images (forward process)
        noisy_images = sqrt_alpha * images + sqrt_one_minus_alpha * noise
        
        # Model predicts noise given noisy images and timesteps
        predicted_noise = model(noisy_images, timesteps)
        
        # Reconstruction: invert the noise addition
        reconstructed_images = (noisy_images - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha

    # Denormalize images for display
    def denormalize(x):
        x = x * 0.5 + 0.5
        return x.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    
    orig = denormalize(images)
    noisy = denormalize(noisy_images)
    recon = denormalize(reconstructed_images)
    
    # Plot a few images (original, noisy, reconstructed)
    n = 5  # Number of images to display
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    for i in range(n):
        axes[i, 0].imshow(orig[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(noisy[i])
        axes[i, 1].set_title("Noisy")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(recon[i])
        axes[i, 2].set_title("Reconstructed")
        axes[i, 2].axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
