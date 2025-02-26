import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

class FaceDatasetPreprocessor:
    """
    Class for preprocessing face dataset images
    """
    def __init__(self, dataset_path, num_samples=2000, seed=42):
        """
        Initialize the preprocessor with dataset path and sampling parameters
        
        Args:
            dataset_path: Path to the CelebA dataset
            num_samples: Number of samples to use (default: 2000)
            seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def get_image_paths(self):
        """
        Get a list of image file paths from the dataset directory
        
        Returns:
            List of image paths limited to num_samples
        """
        all_images = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Sort for reproducibility
        all_images.sort()
        
        # Take only the specified number of samples
        if len(all_images) > self.num_samples:
            all_images = all_images[:self.num_samples]
            
        print(f"Selected {len(all_images)} images from {self.dataset_path}")
        return all_images
    
    def split_dataset(self, image_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split the dataset into training, validation, and test sets
        
        Args:
            image_paths: List of image paths
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples
            test_ratio: Ratio of test samples
            
        Returns:
            Dictionary containing train, val, and test image paths
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        # Shuffle the image paths
        random.shuffle(image_paths)
        
        # Calculate split sizes
        train_size = int(train_ratio * len(image_paths))
        val_size = int(val_ratio * len(image_paths))
        test_size = len(image_paths) - train_size - val_size
        
        # Split the paths
        train_paths = image_paths[:train_size]
        val_paths = image_paths[train_size:train_size + val_size]
        test_paths = image_paths[train_size + val_size:]
        
        print(f"Split dataset: {len(train_paths)} train, {len(val_paths)} validation, {len(test_paths)} test")
        
        return {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        }


class FaceAugmentation:
    """
    Class for creating image augmentations
    """
    def __init__(self):
        """Initialize augmentation transforms"""
        
        # Training transformations with more augmentations
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to uniform size
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.RandomRotation(10),  # Slight random rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])
        
        # Validation/test transformations with minimal processing
        self.val_test_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to uniform size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])
    
    def get_train_transform(self):
        """Get training transforms"""
        return self.train_transform
    
    def get_val_test_transform(self):
        """Get validation/test transforms"""
        return self.val_test_transform


class FaceDataset(Dataset):
    """
    Dataset class for loading and transforming face images
    """
    def __init__(self, image_paths, transform=None):
        """
        Initialize the dataset
        
        Args:
            image_paths: List of image file paths
            transform: Transforms to apply to images
        """
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get an image by index
        
        Args:
            idx: Index of the image
            
        Returns:
            Transformed image (and corresponding index for reference)
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'idx': idx, 'path': img_path}


class FaceDataModule:
    """
    Data module for managing face datasets and dataloaders
    """
    def __init__(self, dataset_path, num_samples=2000, batch_size=32, num_workers=4, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Initialize the data module
        
        Args:
            dataset_path: Path to the CelebA dataset
            num_samples: Number of samples to use
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples
            test_ratio: Ratio of test samples
            seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Initialize other components
        self.preprocessor = FaceDatasetPreprocessor(dataset_path, num_samples, seed)
        self.augmentation = FaceAugmentation()
        
        # Empty placeholders for datasets and dataloaders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
    
    def setup(self):
        """
        Set up datasets and dataloaders
        """
        # Get image paths
        image_paths = self.preprocessor.get_image_paths()
        
        # Split dataset
        splits = self.preprocessor.split_dataset(
            image_paths, 
            self.train_ratio, 
            self.val_ratio, 
            self.test_ratio
        )
        
        # Create datasets with appropriate transforms
        self.train_dataset = FaceDataset(
            splits['train'], 
            transform=self.augmentation.get_train_transform()
        )
        
        self.val_dataset = FaceDataset(
            splits['val'], 
            transform=self.augmentation.get_val_test_transform()
        )
        
        self.test_dataset = FaceDataset(
            splits['test'], 
            transform=self.augmentation.get_val_test_transform()
        )
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"Created dataloaders with batch size {self.batch_size}")
        print(f"Train batches: {len(self.train_dataloader)}")
        print(f"Validation batches: {len(self.val_dataloader)}")
        print(f"Test batches: {len(self.test_dataloader)}")
    
    def get_dataloaders(self):
        """
        Get all dataloaders
        
        Returns:
            Dictionary containing train, val, and test dataloaders
        """
        return {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }
    
    def visualize_batch(self, split='train', num_images=4):
        """
        Visualize a batch of images
        
        Args:
            split: Which split to visualize ('train', 'val', or 'test')
            num_images: Number of images to display
        """
        if split == 'train':
            dataloader = self.train_dataloader
            title = "Training Images"
        elif split == 'val':
            dataloader = self.val_dataloader
            title = "Validation Images"
        else:
            dataloader = self.test_dataloader
            title = "Test Images"
            
        # Get a batch
        batch = next(iter(dataloader))
        images = batch['image']
        
        # Limit to specified number
        images = images[:num_images]
        
        # Create a grid of images
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        fig.suptitle(title)
        
        for i, img in enumerate(images):
            # Denormalize
            img = img * 0.5 + 0.5
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            if len(images) > 1:
                axes[i].imshow(img)
                axes[i].axis('off')
            else:
                axes.imshow(img)
                axes.axis('off')
                
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Path to the CelebA dataset
    dataset_path = "/Users/joannemathew/Desktop/Final DL Project/img_align_celeba"
    
    # Create data module
    data_module = FaceDataModule(
        dataset_path=dataset_path,
        num_samples=2000,
        batch_size=32,
        num_workers=4
    )
    
    # Set up datasets and dataloaders
    data_module.setup()
    
    # Get dataloaders
    dataloaders = data_module.get_dataloaders()
    
    # Visualize a batch from each split
    data_module.visualize_batch(split='train')
    data_module.visualize_batch(split='val')
    data_module.visualize_batch(split='test')
    
    # Example of accessing a batch
    for batch in dataloaders['train']:
        images = batch['image']
        indices = batch['idx']
        paths = batch['path']
        
        print(f"Batch shape: {images.shape}")
        print(f"Indices: {indices[:5]}")
        print(f"Sample paths: {paths[:2]}")
        break