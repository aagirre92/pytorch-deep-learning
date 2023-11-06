"""
Contains functionality for creating PyTorch DataLoader's for image classification data
"""

import os
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_directory:str,test_directory:str,train_transform:torchvision.transforms.v2.Compose,test_transform:torchvision.transforms.v2.Compose,batch_size:int,num_workers:int=NUM_WORKERS):
  """
  Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns them into PyTorch
  Datasets and then into PyTorch DataLoaders

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    train_transform: torchvision transforms to perform on training data.
    test_transform: torchvision transforms to perform on testing data.
    batch_size: batch size for dataloader (nº of samples per batch for each dataloader).
    num_workers: An integer for number of workerks per DataLoader

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    class_names is a list of of the target classes

    Example usage:
      train_dataloader, test_dataloader, class_names = create_dataloaders(
       train_dir=path/to/train_dir,
       test_dir=path/to/test_dir,
       train_transform=some_transform,
       test_transform=some_other_transform,
       batch_size=32,
       num_workers=4)

  """
  # Use ImageFolder to create Dataset(s)
  train_data = datasets.ImageFolder(root=train_directory, # target folder of images
                                    transform=train_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

  test_data = datasets.ImageFolder(root=test_directory,
                                  transform=test_transform)

  # Class names
  class_names = train_data.classes

  # DataLoaders
  train_dataloader = DataLoader(dataset=train_data,
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True,
                              pin_memory=True) # it enables keeping memory in GPU when possible (when GPU available!!)

  test_dataloader = DataLoader(dataset=test_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False,
                              pin_memory=True) # don't usually need to shuffle testing data

  return train_dataloader, test_dataloader, class_names
