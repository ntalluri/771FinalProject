import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import HDF5IterableDataset, generate_row_mask
from MAE import MAEModel
import torch.nn as nn

root_dirs = ["/media/jesse/sda/Organized_SURF", "/media/jesse/sdb/Organized_SURF"]
exclude_dirs = ['Log'] 

dataset = HDF5IterableDataset(
    root_dirs=root_dirs,
    exclude_dirs=exclude_dirs,
    padding_strategy='zero',
    pos_encoding_method='add'
)

batch_size = 4

loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

for data in loader:
    print(data.shape)  # Processed tensor