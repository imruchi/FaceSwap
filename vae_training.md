# VAE Training Instructions

## Basic Usage

```bash
python train_vae.py --download
```

## Parameters

- `--data-root`: Data directory (default: ./data)
- `--download`: Download dataset if not available
- `--num-samples`: Number of samples (default: all)
- `--batch-size`: Batch size (default: 64)
- `--latent-dim`: Latent dimension (default: 512)
- `--epochs`: Training epochs (default: 50)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--kl-weight`: KL divergence weight (default: 0.0025)
- `--save-dir`: Save directory (default: ./checkpoints/vae)

## Example

```bash
python train_vae.py --download --num-samples 5000 --epochs 20
```

## Output

- Model checkpoints: best_vae_model.pt, vae_final_model.pt
- Visualizations: learning curves and reconstructions
