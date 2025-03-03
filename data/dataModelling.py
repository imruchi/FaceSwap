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
    Class for preprocessing face dataset images.
    """
    def __init__(self, dataset_path, num_samples=2000, seed=42):
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def get_image_paths(self):
        all_images = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
        all_images.sort()
        if len(all_images) > self.num_samples:
            all_images = all_images[:self.num_samples]
        print(f"Selected {len(all_images)} images from {self.dataset_path}")
        return all_images
    
    def split_dataset(self, image_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        random.shuffle(image_paths)
        train_size = int(train_ratio * len(image_paths))
        val_size = int(val_ratio * len(image_paths))
        test_size = len(image_paths) - train_size - val_size
        train_paths = image_paths[:train_size]
        val_paths = image_paths[train_size:train_size + val_size]
        test_paths = image_paths[train_size + val_size:]
        print(f"Split dataset: {len(train_paths)} train, {len(val_paths)} validation, {len(test_paths)} test")
        return {'train': train_paths, 'val': val_paths, 'test': test_paths}

class FaceAugmentation:
    """
    Class for creating image augmentations.
    """
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.val_test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def get_train_transform(self):
        return self.train_transform
    
    def get_val_test_transform(self):
        return self.val_test_transform

class FaceDataset(Dataset):
    """
    Dataset class for loading and transforming face images.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'idx': idx, 'path': img_path}

class FaceDataModule:
    """
    Data module for managing face datasets and dataloaders.
    """
    def __init__(self, dataset_path, num_samples=2000, batch_size=32, num_workers=4, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        self.preprocessor = FaceDatasetPreprocessor(dataset_path, num_samples, seed)
        self.augmentation = FaceAugmentation()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
    
    def setup(self):
        image_paths = self.preprocessor.get_image_paths()
        splits = self.preprocessor.split_dataset(image_paths, self.train_ratio, self.val_ratio, self.test_ratio)
        self.train_dataset = FaceDataset(splits['train'], transform=self.augmentation.get_train_transform())
        self.val_dataset = FaceDataset(splits['val'], transform=self.augmentation.get_val_test_transform())
        self.test_dataset = FaceDataset(splits['test'], transform=self.augmentation.get_val_test_transform())
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        print(f"Created dataloaders with batch size {self.batch_size}")
        print(f"Train batches: {len(self.train_dataloader)}")
        print(f"Validation batches: {len(self.val_dataloader)}")
        print(f"Test batches: {len(self.test_dataloader)}")
    
    def get_dataloaders(self):
        return {'train': self.train_dataloader, 'val': self.val_dataloader, 'test': self.test_dataloader}
    
    def visualize_batch(self, split='train', num_images=4):
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
        images = batch['image']
        images = images[:num_images]
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        fig.suptitle(title)
        for i, img in enumerate(images):
            img = img * 0.5 + 0.5
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dataset_path = "/path/to/your/local/celeba"
    data_module = FaceDataModule(dataset_path=dataset_path, num_samples=2000, batch_size=32, num_workers=4)
    data_module.setup()
    dataloaders = data_module.get_dataloaders()
    data_module.visualize_batch(split='train')
    data_module.visualize_batch(split='val')
    data_module.visualize_batch(split='test')
    
    for batch in dataloaders['train']:
        images = batch['image']
        indices = batch['idx']
        paths = batch['path']
        print(f"Batch shape: {images.shape}")
        print(f"Indices: {indices[:5]}")
        print(f"Sample paths: {paths[:2]}")
        break
