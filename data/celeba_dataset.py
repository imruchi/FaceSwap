import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dataModelling import FaceDatasetPreprocessor

class CelebADatasetHandler:
    """
    Class for handling the CelebA dataset from torchvision.
    Provides functionality to download, preprocess, and prepare CelebA dataset.
    """
    def __init__(self, root_dir, download=True, num_samples=None, seed=42):
        """
        Initialize the CelebA dataset handler.
        
        Args:
            root_dir (str): Root directory for dataset storage
            download (bool): Whether to download dataset if not available
            num_samples (int, optional): Number of samples to use, None uses full dataset
            seed (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.download = download
        self.num_samples = num_samples
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def get_datasets(self, transform_train=None, transform_val=None):
        """
        Get CelebA datasets with specified transforms.
        
        Args:
            transform_train (callable, optional): Transform for training set
            transform_val (callable, optional): Transform for validation/test sets
            
        Returns:
            dict: Dictionary containing 'train', 'val', and 'test' datasets
        """
        # Default transforms if none provided
        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
        if transform_val is None:
            transform_val = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        # Get CelebA dataset splits from torchvision
        train_dataset = datasets.CelebA(
            root=self.root_dir,
            split="train",
            target_type="attr",
            transform=transform_train,
            download=self.download
        )
        
        val_dataset = datasets.CelebA(
            root=self.root_dir,
            split="valid",
            target_type="attr",
            transform=transform_val,
            download=self.download
        )
        
        test_dataset = datasets.CelebA(
            root=self.root_dir,
            split="test",
            target_type="attr",
            transform=transform_val,
            download=self.download
        )
        
        # Limit number of samples if specified
        if self.num_samples is not None:
            # Calculate proportional split sizes
            total = len(train_dataset) + len(val_dataset) + len(test_dataset)
            train_ratio = len(train_dataset) / total
            val_ratio = len(val_dataset) / total
            test_ratio = len(test_dataset) / total
            
            train_size = int(self.num_samples * train_ratio)
            val_size = int(self.num_samples * val_ratio)
            test_size = self.num_samples - train_size - val_size
            
            # Create random samplers
            indices_train = torch.randperm(len(train_dataset))[:train_size]
            indices_val = torch.randperm(len(val_dataset))[:val_size]
            indices_test = torch.randperm(len(test_dataset))[:test_size]
            
            train_dataset = torch.utils.data.Subset(train_dataset, indices_train)
            val_dataset = torch.utils.data.Subset(val_dataset, indices_val)
            test_dataset = torch.utils.data.Subset(test_dataset, indices_test)
            
            print(f"Using limited dataset: {train_size} train, {val_size} val, {test_size} test")
        else:
            print(f"Using full dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

class PairedCelebADataset(Dataset):
    """
    Dataset for paired face images from CelebA for face swapping.
    Creates random pairs of faces for training face swap models.
    """
    def __init__(self, celeba_dataset, transform=None):
        """
        Initialize the paired CelebA dataset.
        
        Args:
            celeba_dataset (Dataset): Base CelebA dataset
            transform (callable, optional): Additional transform for paired images
        """
        self.dataset = celeba_dataset
        self.transform = transform
        self.indices = list(range(len(self.dataset)))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get source face
        source_img, source_attr = self.dataset[idx]
        
        # Random target face (different from source)
        target_idx = random.choice([i for i in self.indices if i != idx])
        target_img, target_attr = self.dataset[target_idx]
        
        sample = {
            'source_image': source_img,
            'target_image': target_img,
            'source_attr': source_attr,
            'target_attr': target_attr,
            'source_idx': idx,
            'target_idx': target_idx
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class CelebADataModule:
    """
    Data module for managing CelebA datasets and dataloaders.
    Integrates with torchvision's CelebA dataset.
    """
    def __init__(self, root_dir, download=True, num_samples=None, batch_size=32, 
                 num_workers=4, paired=True, seed=42):
        """
        Initialize the CelebA data module.
        
        Args:
            root_dir (str): Root directory for dataset storage
            download (bool): Whether to download dataset if not available
            num_samples (int, optional): Number of samples to use, None uses full dataset
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for dataloaders
            paired (bool): Whether to create paired datasets for face swapping
            seed (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.download = download
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.paired = paired
        self.seed = seed
        
        self.handler = CelebADatasetHandler(root_dir, download, num_samples, seed)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        
    def setup(self):
        """
        Set up datasets and dataloaders.
        """
        # Define transforms
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Get datasets
        datasets_dict = self.handler.get_datasets(transform_train, transform_val)
        
        # Create paired datasets if needed
        if self.paired:
            self.train_dataset = PairedCelebADataset(datasets_dict['train'])
            self.val_dataset = PairedCelebADataset(datasets_dict['val'])
            self.test_dataset = PairedCelebADataset(datasets_dict['test'])
        else:
            self.train_dataset = datasets_dict['train']
            self.val_dataset = datasets_dict['val']
            self.test_dataset = datasets_dict['test']
        
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
        Get train, validation, and test dataloaders.
        
        Returns:
            dict: Dictionary containing 'train', 'val', and 'test' dataloaders
        """
        return {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }
    
    def visualize_batch(self, split='train', num_images=4):
        """
        Visualize a batch of images.
        
        Args:
            split (str): Dataset split ('train', 'val', 'test')
            num_images (int): Number of images to visualize
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
            
        batch = next(iter(dataloader))
        
        if self.paired:
            source_images = batch['source_image'][:num_images]
            target_images = batch['target_image'][:num_images]
            
            fig, axes = plt.subplots(2, num_images, figsize=(15, 8))
            fig.suptitle(title)
            
            for i in range(num_images):
                # Source images (top row)
                img_source = source_images[i] * 0.5 + 0.5
                img_source = img_source.permute(1, 2, 0).cpu().numpy()
                img_source = np.clip(img_source, 0, 1)
                axes[0, i].imshow(img_source)
                axes[0, i].set_title(f"Source {i+1}")
                axes[0, i].axis('off')
                
                # Target images (bottom row)
                img_target = target_images[i] * 0.5 + 0.5
                img_target = img_target.permute(1, 2, 0).cpu().numpy()
                img_target = np.clip(img_target, 0, 1)
                axes[1, i].imshow(img_target)
                axes[1, i].set_title(f"Target {i+1}")
                axes[1, i].axis('off')
        else:
            images = batch[0][:num_images]  # CelebA returns (image, attributes)
            
            fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
            fig.suptitle(title)
            
            for i in range(num_images):
                img = images[i] * 0.5 + 0.5
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Compatibility with existing code
class CelebAAdapter(FaceDatasetPreprocessor):
    """
    Adapter class to provide compatibility with existing FaceDatasetPreprocessor.
    """
    def __init__(self, root_dir, num_samples=2000, seed=42, download=True):
        """
        Initialize the CelebA adapter.
        
        Args:
            root_dir (str): Root directory for dataset storage
            num_samples (int): Number of samples to use
            seed (int): Random seed
            download (bool): Whether to download dataset
        """
        super().__init__(root_dir, num_samples, seed)
        self.handler = CelebADatasetHandler(root_dir, download, num_samples, seed)
        
    def get_datasets(self):
        """
        Get CelebA datasets.
        
        Returns:
            dict: Dictionary containing 'train', 'val', and 'test' datasets
        """
        return self.handler.get_datasets()

if __name__ == "__main__":
    # Example usage
    root_dir = "./data"
    data_module = CelebADataModule(
        root_dir=root_dir,
        download=True,
        num_samples=2000,
        batch_size=32,
        num_workers=4,
        paired=True
    )
    
    data_module.setup()
    dataloaders = data_module.get_dataloaders()
    data_module.visualize_batch(split='train')
    
    # Access a batch
    for batch in dataloaders['train']:
        if data_module.paired:
            source_images = batch['source_image']
            target_images = batch['target_image']
            source_indices = batch['source_idx']
            target_indices = batch['target_idx']
            print(f"Source batch shape: {source_images.shape}")
            print(f"Target batch shape: {target_images.shape}")
            print(f"Source indices: {source_indices[:5]}")
            print(f"Target indices: {target_indices[:5]}")
        else:
            images, attributes = batch
            print(f"Images shape: {images.shape}")
            print(f"Attributes shape: {attributes.shape}")
        break
