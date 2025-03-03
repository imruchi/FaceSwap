import torch

def train_epoch(model, dataloader, optimizer, device, loss_fn):
    model.train()
    epoch_loss = 0.0
    count = 0
    for batch in dataloader:
        # CelebA returns a tuple: (image, target); we only need the images.
        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)
        noise = torch.randn_like(images).to(device)
        alpha = 0.9
        alpha_tensor = torch.tensor(alpha, device=device)
        noisy_images = torch.sqrt(alpha_tensor) * images + torch.sqrt(1 - alpha_tensor) * noise
        predicted_noise = model(noisy_images)
        loss = loss_fn(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
        count += images.size(0)
    avg_loss = epoch_loss / count
    return avg_loss

def validate_epoch(model, dataloader, device, loss_fn):
    model.eval()
    epoch_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)
            noise = torch.randn_like(images).to(device)
            alpha = 0.9
            alpha_tensor = torch.tensor(alpha, device=device)
            noisy_images = torch.sqrt(alpha_tensor) * images + torch.sqrt(1 - alpha_tensor) * noise
            predicted_noise = model(noisy_images)
            loss = loss_fn(predicted_noise, noise)
            epoch_loss += loss.item() * images.size(0)
            count += images.size(0)
    avg_loss = epoch_loss / count
    return avg_loss
