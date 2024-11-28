import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from scipy import signal
import torch
from torch.utils.data import IterableDataset, DataLoader
import h5py
import numpy as np
from scipy import signal
import os

MIN_ROWS = 1357
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000

def add_positional_encoding(data):
    #TODO currently adds the PE to the data
    # want to try concatting the PE to the data
    
    rows, cols = data.shape

    row_encoding = np.sin(np.arange(rows)[:, np.newaxis] / 10000) + np.cos(np.arange(rows)[:, np.newaxis] / 10000)
    col_encoding = np.sin(np.arange(cols)[np.newaxis, :] / 10000) + np.cos(np.arange(cols)[np.newaxis, :] / 10000)

    # adding not concatting the PE
    data = data + row_encoding[:rows, :] + col_encoding[:, :cols]
    return data


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
            if data_processed is not None and data_processed.shape[:2] == (MAX_ROWS, REQUIRED_COLUMNS):
                yield torch.tensor(data_processed, dtype=torch.float32)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 4), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 4), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 4), padding=(1, 1)),
            nn.ReLU(True),
            # we can sdd more layers if needed
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 4), padding=(1, 1), output_padding=(1, 3)),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 4), padding=(1, 1), output_padding=(1, 3)),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=(2, 4), padding=(1, 1), output_padding=(1, 3)),
            # output layer
            # add the output layer to binary? event or non even
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def apply_row_mask(data, mask_ratio):
    """
    Masks entire rows in the input data.

    data: Tensor of shape [batch_size, 1, H, W]
    mask_ratio: Fraction of rows to mask (e.g., 0.75)
    """
    batch_size, _, H, W = data.shape
    num_rows_to_mask = int(H * mask_ratio)

    # generate random indices for rows to mask
    row_indices = torch.randperm(H)[:num_rows_to_mask]

    # create mask tensor (True for unmasked, False for masked)
    mask = torch.ones((batch_size, 1, H, W), dtype=torch.bool, device=data.device)
    mask[:, :, row_indices, :] = False

    # apply mask
    masked_data = data.clone()
    masked_data[~mask] = 0  # set masked positions to zero

    return masked_data, mask

def masked_row_loss(output, target, mask):
    """
    Computes MSE loss only over masked rows.

    output: Reconstructed data, shape [batch_size, 1, H, W]
    target: Original data, shape [batch_size, 1, H, W]
    mask: Boolean mask tensor
    """
    # compute loss over masked positions
    loss = ((output - target) ** 2)
    masked_loss = loss * (~mask)  # only consider masked positions
    loss = masked_loss.sum() / (~mask).sum()
    return loss

file_dir = f"data/Toy_dataset"
dataset = HDF5IterableDataset(file_dir, padding_strategy='zero')
data_loader = DataLoader(dataset, batch_size=4, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for data in data_loader:
        data = data.unsqueeze(1).to(device)
        masked_data, mask = apply_row_mask(data, mask_ratio=0.75)

        # forward pass
        output = model(masked_data)

        # compute loss
        loss = masked_row_loss(output, data, mask)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
