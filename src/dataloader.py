import torch
from torch.utils.data import IterableDataset
import h5py
import numpy as np
from scipy import signal
import os

MIN_ROWS = 1357
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000


def generate_row_mask(batch_size, num_rows, mask_ratio=0.25):
    row_mask = torch.zeros((batch_size, num_rows), dtype=torch.bool)
    for i in range(batch_size):
        num_masked = int(num_rows * mask_ratio)
        mask_indices = torch.randperm(num_rows)[:num_masked]
        row_mask[i, mask_indices] = True
    return row_mask


def add_positional_encoding(data):
    rows, cols = data.shape
    position = np.arange(rows)[:, np.newaxis]
    div_term = np.exp(np.arange(0, cols, 2) * -(np.log(10000.0) / cols))

    row_encoding = np.zeros((rows, cols))
    row_encoding[:, 0::2] = np.sin(position * div_term)
    row_encoding[:, 1::2] = np.cos(position * div_term)

    return data + row_encoding


def custom_padding(data, target_rows, strategy='zero'):
    rows, cols = data.shape

    if rows >= target_rows:
        return data

    # rows are < target_rows
    if strategy == 'zero':
        # adds zeros
        padding = np.zeros((target_rows - rows, cols))
    elif strategy == 'noise':
        # adds gaussian noise with mean and std of the existing data
        noise = np.random.normal(data.mean(), data.std(), (target_rows - rows, cols))
        padding = noise
    elif strategy == 'repeat':
        # repeats the last row to fill the missing rows
        last_row = data[-1, :]
        padding = np.tile(last_row, (target_rows - rows, 1))
    else:
        raise ValueError(f"Unknown padding strategy: {strategy}")

    return np.vstack([data, padding])

# inherits from IterableDataset, designed for loading data in a lazy fashion, processing each file one at a time.
class HDF5IterableDataset(IterableDataset):
    def __init__(self, file_dir, padding_strategy='zero'):
        self.file_dir = file_dir
        self.file_paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.h5')]
        self.padding_strategy = padding_strategy

    def __len__(self):
        return len(self.file_paths)

    def process_file(self, file_path):
        with h5py.File(file_path, 'r') as file:
            fs = file['Acquisition/Raw[0]'].attrs["OutputDataRate"]
            raw_data_np = file['Acquisition/Raw[0]/RawData'][:]

            rows, cols = raw_data_np.shape
            # skip files with incorrect dimensions
            # the DataLoader will continue iterating through the generator until it gets enough samples to fill the batch size even if a "bad" image is in the data
            if cols != REQUIRED_COLUMNS or rows < MIN_ROWS or rows > MAX_ROWS:
                print(f"Ignored file due to size: {file_path} (Shape: {raw_data_np.shape})")
                return None
            
            # apply preprocessing transformations for each image
            data_detrend = signal.detrend(raw_data_np, type='linear')
            sos = signal.butter(5, [20, 100], 'bandpass', fs=fs, output='sos')
            data_filtered = signal.sosfilt(sos, data_detrend)
            
            # apply padding strategy
            data_padded = custom_padding(data_filtered, MAX_ROWS, strategy=self.padding_strategy)

            # add positional encoding
            data_with_pos_encoding = add_positional_encoding(data_padded)
            return data_with_pos_encoding

    def __iter__(self):
        for file_path in self.file_paths:
            data_processed = self.process_file(file_path)
## for testing purposes
# # Dataset and DataLoader
# file_dir = f"data/Toy_dataset"

# # creates an instance of the custom dataset class HDF5IterableDataset, initialized with the directory path file_dir.
# dataset = HDF5IterableDataset(file_dir, padding_strategy='zero')
# # only one batch of data is loaded into memory at a time, which helps manage resources when handling extensive data
# data_loader = DataLoader(dataset, batch_size=16, num_workers=0)

# for batch_idx, batch_data in enumerate(data_loader):
#     batch_size, num_rows, num_columns = batch_data.shape
    
#     # Generate separate masks for each sample in the batch
#     row_masks = generate_row_mask(batch_size, num_rows, mask_ratio=0.15)
    
#     print(f"Batch {batch_idx}")
#     print(f"Batch data shape: {batch_data.shape}")
#     print(f"Row masks shape: {row_masks.shape}")
#     print(f"Row masks: \n{row_masks}")
#     break  # Only process one batch for demonstration
            if data_processed is not None:
                yield torch.tensor(data_processed, dtype=torch.float32)