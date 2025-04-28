import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed multimodal data
    """
    def __init__(self, data_path):
        """
        Initialize dataset from a numpy file containing preprocessed data
        
        Args:
            data_path (str): Path to the numpy file with preprocessed data
        """
        # Load data
        data = np.load(data_path)
        
        # Extract features and labels
        self.features = data[:, :-1]  # All columns except the last one
        self.labels = data[:, -1]     # Last column is the label
        
        # Convert to torch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        """Return the total number of samples"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Return a sample at the given index"""
        return self.features[idx], self.labels[idx]


def get_data_loaders(train_path, test_path, batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to testing data
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: Training and testing DataLoaders
    """
    # Create datasets
    train_dataset = MultimodalDataset(train_path)
    test_dataset = MultimodalDataset(test_path)
    
    # Print info about the datasets
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Feature dimensions: {train_dataset.features.shape[1]}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Helps speed up data transfer to GPU
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = get_data_loaders(
        'processed_data/train_data.npy',
        'processed_data/test_data.npy'
    )
    
    # Print a sample batch
    for features, labels in train_loader:
        print(f"Batch features shape: {features.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break