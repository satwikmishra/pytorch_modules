import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10


def download_data(path):
    """Download CIFAR10 dataset form torchvision datasets
    Args:
        path (str, optional): download path for dataset
    Returns:
        torchvision instance: training and testing set
    """
    # We also know that the data that we fetch from the pytorch repo isn't natively in the tensor format. So we need to use dataloaders, normalize the data and have iterators to go over the dataset
    trainset = CIFAR10(root=path, train=True, download=True )
    testset = CIFAR10(root=path, train=False, download=True)

    print('Downloaded CIFAR10 to', path)
    return trainset,testset


class LoadDataset(Dataset):
    """Torch Dataset instance for loading the dataset and transforming it
    """
    def __init__(self, data, transform=False):
        self.data = data
        self.aug = transform
    
    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, i):
        """Read image from dataset and performs transformations
        """
        image, label = self.data[i]
        
        if self.aug:
            image = self.aug(image=np.array(image))['image']
        
        return image, label


      
def get_train_test_loaders(train_transforms, test_transforms, BATCH_SIZE, download_path='/content/data'):
    """Generate Torch instance for Train and Test data loaders
    Args:
        train_transforms (albumentations compose class): training tansformations 
        test_transforms (albumentations compose class): testing tansformations
        BATCH_SIZE (int): Batch size
        download_path (str): download path
    Returns:
        torch instace: train and test data loaders
    """
    trainset, testset = download_data(download_path)

    train_loader = DataLoader(LoadDataset(trainset, train_transforms), batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
    test_loader = DataLoader(LoadDataset(testset, test_transforms), batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=1)

    print("Successfully created the train and the test loaders")
    return train_loader, test_loader
