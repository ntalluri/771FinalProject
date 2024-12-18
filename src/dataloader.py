import torch
import h5py
import numpy as np
from scipy import signal
import os
from torch.utils.data import IterableDataset
from itertools import cycle
import random

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
    def __init__(self, file_paths, labels_dict=None, padding_strategy='zero',
                 pos_encoding_method='add', device='cuda', mode='train',
                 augment_positive=True, augmentations=None):
        """
        Initialize the dataset with GPU support and optional labels.

        Args:
            file_paths (list): List of HDF5 file paths
            labels_dict (dict, optional): Dictionary mapping filenames to labels (0 or 1)
            padding_strategy (str): Strategy for padding rows
            pos_encoding_method (str): Method for positional encoding
            device (str): Device to process data on ('cuda' or 'cpu')
            mode (str): Mode of the dataset ('train', 'val', 'test')
            augment_positive (bool): Whether to apply augmentations to positive samples
            augmentations (dict, optional): Dictionary specifying which augmentations to apply
                Example:
                {
                    'noise': {
                        'apply_prob': 0.5,
                        'level_min': 0.001,
                        'level_max': 0.02
                    },
                    'row_shift': {  # Changed from 'row_swap' to 'row_shift'
                        'apply_prob': 0.5,
                        'shift_max_min': 1,
                        'shift_max_max': 10
                    },
                    'column_shift': {
                        'apply_prob': 0.5,
                        'shift_max_min': 1,
                        'shift_max_max': 10
                    }
                }
        """
        self.file_paths = file_paths
        self.labels_dict = labels_dict
        if not self.file_paths:
            raise ValueError("No valid HDF5 files found in the provided directories.")
        self.padding_strategy = padding_strategy
        self.pos_encoding_method = pos_encoding_method
        self.device = device
        self.mode = mode.lower()
        self.augment_positive = augment_positive
        self.augmentations = augmentations if augmentations is not None else {
            'noise': {
                'apply_prob': 0.5,
                'level_min': 0.001,
                'level_max': 0.02
            },
            'row_shift': {  # Changed from 'row_swap' to 'row_shift'
                'apply_prob': 0.5,
                'shift_max_min': 1,
                'shift_max_max': 10
            },
            'column_shift': {
                'apply_prob': 0.5,
                'shift_max_min': 1,
                'shift_max_max': 10
            }
        }

        # Validate mode
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError("Mode should be one of 'train', 'val', or 'test'.")

        # Separate positive and negative files
        if self.labels_dict is not None:
            self.positive_files = [f for f in self.file_paths if self.labels_dict.get(os.path.basename(f), 0) == 1]
            self.negative_files = [f for f in self.file_paths if self.labels_dict.get(os.path.basename(f), 0) == 0]
            
            if self.mode == 'train':
                if not self.positive_files:
                    raise ValueError("No positive samples found in the provided file paths.")
                if not self.negative_files:
                    raise ValueError("No negative samples found in the provided file paths.")
            else:
                # For validation and test, ensure all positive and negative samples are included
                if not self.positive_files and not self.negative_files:
                    raise ValueError("No samples found in the provided file paths.")
        else:
            self.positive_files = []
            self.negative_files = self.file_paths.copy()

        # Initialize cycling iterator for positive samples if in training mode and oversampling is desired
        if self.mode == 'train' and self.positive_files:
            self.positive_iter = cycle(self.positive_files)
        else:
            self.positive_iter = iter([])  # Empty iterator

        # Precompute position encoding matrices for GPU
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
    
    def add_random_noise(self, data, noise_level):
        """
        Add Gaussian noise to the data.

        Args:
            data (torch.Tensor): Input data tensor
            noise_level (float): Standard deviation of the Gaussian noise

        Returns:
            torch.Tensor: Noisy data
        """
        noise = torch.randn_like(data) * noise_level
        return data + noise

    def row_shift(self, data, shift_max):
        """
        Randomly shift rows up or down by up to shift_max positions.
        Performs a circular shift so that rows shifted off one end reappear on the other.

        Args:
            data (torch.Tensor): Input data tensor
            shift_max (int): Maximum number of positions to shift

        Returns:
            torch.Tensor: Data with circularly shifted rows
        """
        shift = random.randint(-shift_max, shift_max)
        if shift == 0:
            return data
        # Perform circular shift using torch.roll
        shifted_data = torch.roll(data, shifts=shift, dims=0)
        return shifted_data

    def column_shift(self, data, shift_max):
        """
        Randomly shift columns left or right by up to shift_max positions.
        Performs a circular shift so that columns shifted off one end reappear on the other.

        Args:
            data (torch.Tensor): Input data tensor
            shift_max (int): Maximum number of positions to shift

        Returns:
            torch.Tensor: Data with circularly shifted columns
        """
        shift = random.randint(-shift_max, shift_max)
        if shift == 0:
            return data
        # Perform circular shift using torch.roll without zeroing out
        shifted_data = torch.roll(data, shifts=shift, dims=1)
        return shifted_data

    def apply_augmentations(self, data):
        """
        Apply a series of augmentations to the data based on configuration.
        Each augmentation is applied with a certain probability and random parameters within specified ranges.

        Args:
            data (torch.Tensor): Input data tensor

        Returns:
            torch.Tensor: Augmented data
        """
        # Noise Augmentation
        noise_config = self.augmentations.get('noise', {})
        if noise_config.get('apply_prob', 0.0) > 0.0:
            if random.random() < noise_config['apply_prob']:
                noise_level = random.uniform(noise_config['level_min'], noise_config['level_max'])
                data = self.add_random_noise(data, noise_level)

        # Row Shift Augmentation
        row_shift_config = self.augmentations.get('row_shift', {})
        if row_shift_config.get('apply_prob', 0.0) > 0.0:
            if random.random() < row_shift_config['apply_prob']:
                shift_max = random.randint(row_shift_config['shift_max_min'], row_shift_config['shift_max_max'])
                data = self.row_shift(data, shift_max)

        # Column Shift Augmentation
        column_shift_config = self.augmentations.get('column_shift', {})
        if column_shift_config.get('apply_prob', 0.0) > 0.0:
            if random.random() < column_shift_config['apply_prob']:
                shift_max = random.randint(column_shift_config['shift_max_min'], column_shift_config['shift_max_max'])
                data = self.column_shift(data, shift_max)

        return data

    def process_file(self, file_path):
        """
        Process a single HDF5 file with GPU acceleration where beneficial.
        CPU is used for I/O and initial signal processing, GPU for later stages.
        Returns the processed data tensor and label if labels_dict is provided.

        Positional encodings are **not** added here anymore and should be applied after augmentations.

        Args:
            file_path (str): Path to the HDF5 file

        Returns:
            tuple or torch.Tensor: (data_tensor, label) if labels_dict is provided, else data_tensor
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
                    label = self.labels_dict.get(os.path.basename(file_path), 0) if self.labels_dict else 0
                    if self.mode == 'train' and label == 1:
                        raise ValueError(f"Positive file {file_path} does not meet size requirements.")
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

                # **Do not add positional encoding here**
                # data_with_pos = data_padded + self.pos_encoding_cache if self.pos_encoding_method == 'add' else torch.cat([data_padded, self.pos_encoding_cache], dim=1)

                # Get label if labels_dict is provided
                if self.labels_dict is not None:
                    filename = os.path.basename(file_path)
                    label = self.labels_dict.get(filename, 0)  # Default to 0 if not found
                    label_tensor = torch.tensor(label, dtype=torch.float32, device=self.device)
                    return data_padded, label_tensor
                
                return data_padded

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

    def __iter__(self):
        """
        Iterate through the dataset, yielding processed tensors and labels if available.
        Implements oversampling of positive samples and applies augmentations only in training mode.
        Robustly handles exceptions to ensure the iterator continues despite individual file errors.
        Positional encodings are applied after augmentations.

        Yields:
            tuple or torch.Tensor: (data_with_pos, label) if labels_dict is provided, else data_with_pos
        """
        if self.labels_dict is not None:
            if self.mode == 'train':
                # Iterate through negative files
                for neg_file in self.negative_files:
                    try:
                        neg_sample = self.process_file(neg_file)
                        if neg_sample is not None:
                            data, label = neg_sample
                            # No augmentations for negative samples
                            # Add positional encoding
                            if self.pos_encoding_method == 'add':
                                data_with_pos = data + self.pos_encoding_cache
                            else:  # 'concat'
                                data_with_pos = torch.cat([data, self.pos_encoding_cache], dim=1)
                            yield data_with_pos, label
                    except Exception as e:
                        print(f"Skipping negative file {neg_file} due to error: {e}")
                        continue

                    # Oversample positive samples
                    if self.positive_files:
                        try:
                            pos_file = next(self.positive_iter)
                            pos_sample = self.process_file(pos_file)
                            if pos_sample is not None:
                                data, label = pos_sample
                                if self.augment_positive:
                                    data = self.apply_augmentations(data)
                                # Add positional encoding after augmentations
                                if self.pos_encoding_method == 'add':
                                    data_with_pos = data + self.pos_encoding_cache
                                else:  # 'concat'
                                    data_with_pos = torch.cat([data, self.pos_encoding_cache], dim=1)
                                yield data_with_pos, label
                        except Exception as e:
                            print(f"Error processing positive file {pos_file}: {e}")
                            continue
            else:
                # For 'val' and 'test' modes, yield all samples without oversampling or augmentations
                for file_path in self.file_paths:
                    try:
                        sample = self.process_file(file_path)
                        if sample is not None:
                            data = sample[0]
                            label = sample[1] if len(sample) > 1 else None
                            # Add positional encoding
                            if self.pos_encoding_method == 'add':
                                data_with_pos = data + self.pos_encoding_cache
                            else:  # 'concat'
                                data_with_pos = torch.cat([data, self.pos_encoding_cache], dim=1)
                            if label is not None:
                                yield data_with_pos, label
                            else:
                                yield data_with_pos
                    except Exception as e:
                        print(f"Skipping file {file_path} due to error: {e}")
                        continue
        else:
            # If no labels, yield normally
            for file_path in self.file_paths:
                try:
                    data = self.process_file(file_path)
                    if data is not None:
                        # Add positional encoding
                        if self.pos_encoding_method == 'add':
                            data_with_pos = data + self.pos_encoding_cache
                        else:  # 'concat'
                            data_with_pos = torch.cat([data, self.pos_encoding_cache], dim=1)
                        yield data_with_pos
                except Exception as e:
                    print(f"Skipping file {file_path} due to error: {e}")
                    continue
