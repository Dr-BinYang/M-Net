from data_provider.data_loader import Dataset_ERA5
from torch.utils.data import DataLoader
import argparse
import torch


def data_provider(args, flag):
    """
    Create a data loader for the specified dataset type (train/val/test)

    Parameters:
    - args: Configuration arguments containing dataset parameters
    - flag: Dataset type ('train', 'val', or 'test')

    Returns:
    - data_loader: PyTorch DataLoader for the specified dataset
    - targetsScaler: Normalizer object for target features (used for inverse transformation)
    """
    # Create dataset instance for the specified type
    data_set = Dataset_ERA5(args, flag)

    # Set DataLoader parameters based on dataset type
    if flag == 'train':
        shuffle_flag = True  # Shuffle training data for better generalization
        drop_last = True  # Drop last incomplete batch
        batch_size = args.batch_size
    else:
        shuffle_flag = False  # Don't shuffle validation/test data
        drop_last = True  # Drop last incomplete batch
        batch_size = args.batch_size

    # Create DataLoader with specified parameters
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)

    # Return DataLoader and target normalizer (for inverse transformation)
    return data_loader, data_set.targetsScaler