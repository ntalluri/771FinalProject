import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import HDF5IterableDataset, generate_row_mask
from MAE import MAEModel
import torch.nn as nn
import random 

def gather_all_file_paths(root_dirs):
    """
    Gather all HDF5 file paths from the immediate subdirectories of the given root directories.

    Args:
        root_dirs (list or str): List of root directories or a single root directory

    Returns:
        list: List of file paths
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    file_paths = []
    for root_dir in root_dirs:
        # List immediate subdirectories in the root directory
        try:
            subdirs = [
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        except FileNotFoundError:
            print(f"Root directory not found: {root_dir}")
            continue

        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            # List files ending with '.h5' in the current subdirectory (non-recursive)
            try:
                files_in_subdir = os.listdir(subdir_path)
            except FileNotFoundError:
                print(f"Subdirectory not found: {subdir_path}")
                continue

            h5_files = [
                os.path.join(subdir_path, f)
                for f in files_in_subdir
                if os.path.isfile(os.path.join(subdir_path, f)) and f.endswith('.h5')
            ]
            file_paths.extend(h5_files)
    return file_paths

root_dirs = ["/media/jesse/sda/Organized_SURF", "/media/jesse/sdb/Organized_SURF"]
all_file_paths = gather_all_file_paths(root_dirs)

# Shuffle and split the file paths
random.shuffle(all_file_paths)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
total_files = len(all_file_paths)
train_end = int(train_ratio * total_files)
val_end = train_end + int(val_ratio * total_files)
train_file_paths = all_file_paths[:train_end]
val_file_paths = all_file_paths[train_end:val_end]
test_file_paths = all_file_paths[val_end:]
print(f"Total files: {total_files}")
print(f"Training files: {len(train_file_paths)}")
print(f"Validation files: {len(val_file_paths)}")
print(f"Testing files: {len(test_file_paths)}")

# Create datasets
train_dataset = HDF5IterableDataset(train_file_paths)
val_dataset = HDF5IterableDataset(val_file_paths)
test_dataset = HDF5IterableDataset(test_file_paths)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for data in train_loader:
    print(data.shape)
    break

for data in val_loader:
    print(data.shape)
    break

for data in test_loader:
    print(data.shape)
    break
