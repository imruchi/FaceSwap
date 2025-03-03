import torch

def train_epoch(model, dataloader, optimizer, device, loss_fn, num_train_timesteps=1000):
    model.train()
    epoch_loss = 0.0
    count = 0
    for batch in dataloader:
        # CelebA returns a tuple: (images, targets); we only need the images.
        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)
        
        # Sample a random timestep for each image in the batch
        timesteps = torch.randint(0, num_train_timesteps, (images.size(0),), device=device).long()
        
        noise = torch.randn_like(images).to(device)
        alpha = 0.9
        alpha_tensor = torch.tensor(alpha, device=device)
        sqrt_alpha = torch.sqrt(alpha_tensor)
        sqrt_one_minus_alpha = torch.sqrt(torch.tensor(1.0, device=device) - alpha_tensor)
        
        # Add noise to images (forward process)
        noisy_images = sqrt_alpha * images + sqrt_one_minus_alpha * noise
        
        # Forward pass: supply both the noisy images and timesteps
        predicted_noise = model(noisy_images, timesteps)
        
        loss = loss_fn(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
        count += images.size(0)
    avg_loss = epoch_loss / count
    return avg_loss

def validate_epoch(model, dataloader, device, loss_fn, num_train_timesteps=1000):
    model.eval()
    epoch_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)
            timesteps = torch.randint(0, num_train_timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images).to(device)
            alpha = 0.9
            alpha_tensor = torch.tensor(alpha, device=device)
            sqrt_alpha = torch.sqrt(alpha_tensor)
            sqrt_one_minus_alpha = torch.sqrt(torch.tensor(1.0, device=device) - alpha_tensor)
            noisy_images = sqrt_alpha * images + sqrt_one_minus_alpha * noise
            predicted_noise = model(noisy_images, timesteps)
            loss = loss_fn(predicted_noise, noise)
            epoch_loss += loss.item() * images.size(0)
            count += images.size(0)
    avg_loss = epoch_loss / count
    return avg_loss
