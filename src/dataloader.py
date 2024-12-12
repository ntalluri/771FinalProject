import torch
import h5py
import numpy as np
from scipy import signal
import os
from torch.utils.data import IterableDataset
from torch.multiprocessing import Pool, cpu_count
from itertools import cycle
from functools import partial

# Constants for data dimensions
MIN_ROWS = 1357
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000

def generate_row_mask(batch_size, num_rows, mask_ratio=0.25):
    """
    Generate masks for each batch to indicate which rows should be masked.

    Args:
        batch_size (int): Number of samples in the batch
        num_rows (int): Number of rows in each sample
        mask_ratio (float): Proportion of rows to mask (default: 0.25 as per specs)

    Returns:
        torch.Tensor: Boolean tensor of shape [batch_size, num_rows] where True indicates masked positions
    """
    row_mask = torch.zeros((batch_size, num_rows), dtype=torch.bool)
    for i in range(batch_size):
        num_masked = int(num_rows * mask_ratio)
        mask_indices = torch.randperm(num_rows)[:num_masked]
        row_mask[i, mask_indices] = True
    return row_mask

class HDF5IterableDataset(IterableDataset):
    def __init__(self, file_paths, labels_dict=None, padding_strategy='zero', pos_encoding_method='add', device='cuda'):
        """
        Initialize the dataset with GPU support and optional labels.
        
        Args:
            file_paths (list): List of HDF5 file paths
            labels_dict (dict, optional): Dictionary mapping filenames to labels (0 or 1)
            padding_strategy (str): Strategy for padding rows
            pos_encoding_method (str): Method for positional encoding
            device (str): Device to process data on ('cuda' or 'cpu')
        """
        self.file_paths = file_paths
        self.labels_dict = labels_dict
        if not self.file_paths:
            raise ValueError("No valid HDF5 files found in the provided directories.")
        self.padding_strategy = padding_strategy
        self.pos_encoding_method = pos_encoding_method
        self.device = device
        
        # Pre-compute position encoding matrices for GPU
        self.pos_encoding_cache = self._prepare_pos_encoding()
    
    def _prepare_pos_encoding(self):
        """Precompute position encoding matrices and store them on GPU"""
        rows, cols = MAX_ROWS, REQUIRED_COLUMNS
        position = np.arange(rows)[:, np.newaxis]
        div_term = np.exp(np.arange(0, cols, 2) * -(np.log(10000.0) / cols))
        
        row_encoding = np.zeros((rows, cols))
        row_encoding[:, 0::2] = np.sin(position * div_term)
        row_encoding[:, 1::2] = np.cos(position * div_term)
        
        return torch.tensor(row_encoding, dtype=torch.float32, device=self.device)
    
    def process_file(self, file_path):
        """
        Process a single HDF5 file with GPU acceleration where beneficial.
        CPU is used for I/O and initial signal processing, GPU for later stages.
        Returns both processed data and label if labels_dict is provided.
        """
        try:
            with h5py.File(file_path, 'r') as file:
                fs = file['Acquisition/Raw[0]'].attrs.get("OutputDataRate", None)
                if fs is None:
                    print(f"Missing 'OutputDataRate' attribute in file: {file_path}")
                    return None

                # Keep initial load on CPU
                raw_data_np = file['Acquisition/Raw[0]/RawData'][:]

                rows, cols = raw_data_np.shape
                if cols != REQUIRED_COLUMNS or rows < MIN_ROWS or rows > MAX_ROWS:
                    print(f"Ignored file due to size: {file_path} (Shape: {raw_data_np.shape})")
                    return None

                # Perform CPU-bound operations first
                data_detrend = signal.detrend(raw_data_np, type='linear')
                sos = signal.butter(5, [20, 100], 'bandpass', fs=fs, output='sos')
                data_filtered = signal.sosfilt(sos, data_detrend)
                
                # Convert to tensor and move to GPU for remaining operations
                data_tensor = torch.tensor(data_filtered, dtype=torch.float32, device=self.device)
                
                # Normalize on GPU
                data_normalized = (data_tensor - data_tensor.mean()) / (data_tensor.std() + 1e-8)
                
                # Padding on GPU
                if rows < MAX_ROWS:
                    padding_rows = MAX_ROWS - rows
                    if self.padding_strategy == 'zero':
                        padding = torch.zeros((padding_rows, cols), device=self.device)
                    elif self.padding_strategy == 'noise':
                        mean = data_normalized.mean()
                        std = data_normalized.std()
                        padding = torch.normal(mean, std, (padding_rows, cols), device=self.device)
                    elif self.padding_strategy == 'repeat':
                        padding = data_normalized[-1:].repeat(padding_rows, 1)
                    elif self.padding_strategy == 'mirror':
                        num_rows_to_mirror = min(padding_rows, rows)
                        mirrored_rows = data_normalized[-num_rows_to_mirror:].flip(0)
                        if padding_rows > num_rows_to_mirror:
                            remaining_rows = padding_rows - num_rows_to_mirror
                            padding = torch.cat([
                                mirrored_rows,
                                mirrored_rows[-1:].repeat(remaining_rows, 1)
                            ])
                        else:
                            padding = mirrored_rows[:padding_rows]
                    
                    data_padded = torch.cat([data_normalized, padding], dim=0)
                else:
                    data_padded = data_normalized

                # Apply positional encoding on GPU
                if self.pos_encoding_method == 'add':
                    data_with_pos = data_padded + self.pos_encoding_cache
                else:  # 'concat'
                    data_with_pos = torch.cat([data_padded, self.pos_encoding_cache], dim=1)

                # Get label if labels_dict is provided
                if self.labels_dict is not None:
                    filename = os.path.basename(file_path)
                    label = self.labels_dict.get(filename, 0)  # Default to 0 if not found
                    label_tensor = torch.tensor(label, dtype=torch.float32, device=self.device)
                    return data_with_pos, label_tensor
                
                return data_with_pos

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def __iter__(self):
        """
        Iterate through the dataset, yielding processed tensors and labels if available.
        """
        for file_path in self.file_paths:
            processed = self.process_file(file_path)
            if processed is not None:
                yield processed
