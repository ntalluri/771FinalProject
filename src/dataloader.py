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


# def add_positional_encoding(data, method='add'):
#     """
#     Add positional encoding to the data matrix using either addition or concatenation.

#     The encoding uses sinusoidal functions to create unique position-based values,
#     allowing the model to understand spatial relationships in the data.

#     Args:
#         data (np.ndarray): Input data matrix of shape [rows, cols]
#         method (str): How to combine positional encoding with data:
#             - 'add': Add encoding directly to data (preserves original dimensions)
#             - 'concat': Concatenate encoding as new columns (doubles column dimension)

#     Returns:
#         np.ndarray: Data with positional encoding applied

#     Note:
#         Following the paper specs, we use sinusoidal encoding to provide
#         spatial context while maintaining the ability to extrapolate to
#         different sequence lengths.
#     """
#     rows, cols = data.shape
#     position = np.arange(rows)[:, np.newaxis]
#     div_term = np.exp(np.arange(0, cols, 2) * -(np.log(10000.0) / cols))

#     # Create sinusoidal position encoding
#     row_encoding = np.zeros((rows, cols))
#     row_encoding[:, 0::2] = np.sin(position * div_term)
#     row_encoding[:, 1::2] = np.cos(position * div_term)

#     if method == 'add':
#         # Add encoding directly to preserve original dimensions
#         return data + row_encoding
#     elif method == 'concat':
#         # Concatenate encoding as additional features
#         # This preserves the original data values but increases dimensionality
#         return np.concatenate([data, row_encoding], axis=1)
#     else:
#         raise ValueError(f"Unknown positional encoding method: {method}")


# def custom_padding(data, target_rows, strategy='zero'):
#     """
#     Pad the data matrix to reach target_rows using various strategies.
#     Each strategy offers different benefits for maintaining data characteristics.

#     Args:
#         data (np.ndarray): Input data matrix
#         target_rows (int): Desired number of rows after padding
#         strategy (str): Padding strategy to use:
#             - 'zero': Simple zero padding (baseline approach)
#             - 'noise': Gaussian noise based on data statistics (maintains data distribution)
#             - 'repeat': Repeats last row (maintains edge patterns)
#             - 'mirror': Reflects last rows (maintains local patterns)

#     Returns:
#         np.ndarray: Padded data matrix with target_rows rows

#     Note:
#         As specified in the report, these strategies aim to better simulate
#         real-world conditions and provide the model with different types
#         of context for the padding regions.
#     """
#     rows, cols = data.shape

#     if rows >= target_rows:
#         return data

#     padding_rows = target_rows - rows

#     if strategy == 'zero':
#         # Simple zero padding - baseline approach
#         # Useful when we want to clearly distinguish padding from real data
#         padding = np.zeros((padding_rows, cols))

#     elif strategy == 'noise':
#         # Gaussian noise padding based on data statistics
#         # Maintains the statistical properties of the real data
#         mean = data.mean()
#         std = data.std()
#         padding = np.random.normal(mean, std, (padding_rows, cols))

#     elif strategy == 'repeat':
#         # Repeat last row - useful when we expect continuation of patterns
#         # Good for data where the last row represents a stable state
#         padding = np.tile(data[-1:], (padding_rows, 1))

#     elif strategy == 'mirror':
#         # Mirror padding - maintains local patterns and continuity
#         # Particularly useful for spatial data where local structure is important
#         num_rows_to_mirror = min(padding_rows, rows)
#         mirrored_rows = data[-num_rows_to_mirror:][::-1]
#         if padding_rows > num_rows_to_mirror:
#             # If we need more padding rows than available data rows,
#             # fill the rest with the last mirrored row
#             remaining_rows = padding_rows - num_rows_to_mirror
#             padding = np.vstack([
#                 mirrored_rows,
#                 np.tile(mirrored_rows[-1:], (remaining_rows, 1))
#             ])
#         else:
#             padding = mirrored_rows[:padding_rows]

#     else:
#         raise ValueError(f"Unknown padding strategy: {strategy}")

#     return np.vstack([data, padding])

# # inherits from IterableDataset, designed for loading data in a lazy fashion, processing each file one at a time.
# class HDF5IterableDataset(IterableDataset):
#     """
#     Custom dataset for loading and preprocessing HDF5 DAS data files.
#     Implements lazy loading and on-the-fly preprocessing to manage memory efficiently.

#     Supports multiple root directories and nested directories containing HDF5 files,
#     with the ability to exclude specific folders.
#     """

#     def __init__(self, file_paths, padding_strategy='zero', pos_encoding_method='add'):
#         """
#         Initialize the dataset with customizable preprocessing options.

#         Args:
#             root_dirs (list or str): List of root directories or a single root directory
#             exclude_dirs (list or set): List of folder names to exclude from processing
#             padding_strategy (str): Strategy for padding rows ('zero', 'noise', 'repeat', 'mirror')
#             pos_encoding_method (str): Method for positional encoding ('add', 'concat')
#         """
#         self.file_paths = file_paths
#         if not self.file_paths:
#             raise ValueError(f"No valid HDF5 files found in the provided directories.")
#         self.padding_strategy = padding_strategy
#         self.pos_encoding_method = pos_encoding_method

#     def __len__(self):
#         return len(self.file_paths)

#     def process_file(self, file_path):
#         """
#         Process a single HDF5 file, applying all necessary transformations.

#         Implements the preprocessing pipeline specified in the report:
#         1. Load raw data
#         2. Apply detrending
#         3. Apply bandpass filtering
#         4. Normalize data
#         5. Apply padding
#         6. Add positional encoding

#         Args:
#             file_path (str): Path to HDF5 file

#         Returns:
#             torch.Tensor: Processed data tensor, or None if file is invalid
#         """
#         with h5py.File(file_path, 'r') as file:
#             fs = file['Acquisition/Raw[0]'].attrs["OutputDataRate"]
#             raw_data_np = file['Acquisition/Raw[0]/RawData'][:]

#             rows, cols = raw_data_np.shape
#             if cols != REQUIRED_COLUMNS or rows < MIN_ROWS or rows > MAX_ROWS:
#                 print(f"Ignored file due to size: {file_path} (Shape: {raw_data_np.shape})")
#                 return None

#             # Apply preprocessing transformations as specified in the report
#             data_detrend = signal.detrend(raw_data_np, type='linear')
#             sos = signal.butter(5, [20, 100], 'bandpass', fs=fs, output='sos')
#             data_filtered = signal.sosfilt(sos, data_detrend)

#             # Normalize the filtered data
#             data_normalized = (data_filtered - np.mean(data_filtered)) / (np.std(data_filtered) + 1e-8)

#             # Apply padding with selected strategy
#             data_padded = custom_padding(data_normalized, MAX_ROWS, strategy=self.padding_strategy)

#             # Apply positional encoding with selected method
#             data_with_pos_encoding = add_positional_encoding(data_padded, method=self.pos_encoding_method)

#             return torch.tensor(data_with_pos_encoding, dtype=torch.float32)

#     def __iter__(self):
#         """
#         Iterate through the dataset, yielding processed tensors.
#         Implements lazy loading as specified in the report.
#         """
#         for file_path in self.file_paths:
#             data_processed = self.process_file(file_path)
#             if data_processed is not None:
#                 yield data_processed


class HDF5IterableDataset(IterableDataset):
    def __init__(self, file_paths, padding_strategy='zero', pos_encoding_method='add', device='cuda'):
        """
        Initialize the dataset with GPU support.
        
        Args:
            file_paths (list): List of HDF5 file paths
            padding_strategy (str): Strategy for padding rows
            pos_encoding_method (str): Method for positional encoding
            device (str): Device to process data on ('cuda' or 'cpu')
        """
        self.file_paths = file_paths
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
        """
        with h5py.File(file_path, 'r') as file:
            fs = file['Acquisition/Raw[0]'].attrs["OutputDataRate"]
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

            return data_with_pos

    def __iter__(self):
        """Iterate through the dataset, yielding GPU tensors"""
        for file_path in self.file_paths:
            data_processed = self.process_file(file_path)
            if data_processed is not None:
                yield data_processed
