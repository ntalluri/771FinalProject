import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from dataloader import generate_row_mask, custom_padding, add_positional_encoding
from MAE import MAEModel
import torch.nn as nn
import random
import h5py
import numpy as np
from scipy import signal


class HDF5Dataset(Dataset):
    """
    Dataset class for HDF5 files that supports distributed training
    """

    def __init__(self, file_paths, padding_strategy='zero', pos_encoding_method='add'):
        self.file_paths = file_paths
        self.padding_strategy = padding_strategy
        self.pos_encoding_method = pos_encoding_method
        self.MIN_ROWS = 1357
        self.MAX_ROWS = 1387
        self.REQUIRED_COLUMNS = 30000

    def __len__(self):
        return len(self.file_paths)

    def process_file(self, file_path):
        with h5py.File(file_path, 'r') as file:
            fs = file['Acquisition/Raw[0]'].attrs["OutputDataRate"]
            raw_data_np = file['Acquisition/Raw[0]/RawData'][:]

            rows, cols = raw_data_np.shape
            if cols != self.REQUIRED_COLUMNS or rows < self.MIN_ROWS or rows > self.MAX_ROWS:
                return None

            # Apply preprocessing transformations
            data_detrend = signal.detrend(raw_data_np, type='linear')
            sos = signal.butter(5, [20, 100], 'bandpass', fs=fs, output='sos')
            data_filtered = signal.sosfilt(sos, data_detrend)

            # Normalize the filtered data
            data_normalized = (data_filtered - np.mean(data_filtered)) / (np.std(data_filtered) + 1e-8)

            # Apply padding with selected strategy
            data_padded = custom_padding(data_normalized, self.MAX_ROWS, strategy=self.padding_strategy)

            # Apply positional encoding with selected method
            data_with_pos_encoding = add_positional_encoding(data_padded, method=self.pos_encoding_method)

            return torch.tensor(data_with_pos_encoding, dtype=torch.float32)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            data = self.process_file(file_path)
            if data is None:
                # Return a zero tensor if the file is invalid
                return torch.zeros((self.MAX_ROWS, self.REQUIRED_COLUMNS), dtype=torch.float32)
            return data
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return torch.zeros((self.MAX_ROWS, self.REQUIRED_COLUMNS), dtype=torch.float32)


def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()


def train(rank, world_size, args):
    """
    Training function for each process.
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Parameters
    batch = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    mask_ratio = args.mask_ratio
    padding = args.padding
    positional_encodings = args.positional_encodings
    embedding_dim = args.embedding_dim
    number_heads = args.number_heads
    layers = args.layers
    MAX_ROWS = 1387
    REQUIRED_COLUMNS = 30000

    # Create model and move it to GPU
    model = MAEModel(
        input_dim=REQUIRED_COLUMNS,
        embed_dim=embedding_dim,
        num_heads=number_heads,
        depth=layers
    ).to(rank)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Gather file paths
    root_dirs = ["/media/jesse/sda/Organized_SURF", "/media/jesse/sdb/Organized_SURF"]
    all_file_paths = []
    for root_dir in root_dirs:
        try:
            subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        except FileNotFoundError:
            if rank == 0:
                print(f"Root directory not found: {root_dir}")
            continue

        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            try:
                files_in_subdir = os.listdir(subdir_path)
            except FileNotFoundError:
                continue

            h5_files = [
                os.path.join(subdir_path, f)
                for f in files_in_subdir
                if os.path.isfile(os.path.join(subdir_path, f)) and f.endswith('.h5')
            ]
            all_file_paths.extend(h5_files)

    # Split datasets
    random.seed(42)
    random.shuffle(all_file_paths)
    train_ratio = 0.7
    val_ratio = 0.15
    total_files = len(all_file_paths)
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)
    train_file_paths = all_file_paths[:train_end]
    val_file_paths = all_file_paths[train_end:val_end]
    test_file_paths = all_file_paths[val_end:]

    if rank == 0:
        print(f"Total files: {total_files}")
        print(f"Training files: {len(train_file_paths)}")
        print(f"Validation files: {len(val_file_paths)}")
        print(f"Testing files: {len(test_file_paths)}")

    # Create datasets
    train_dataset = HDF5Dataset(train_file_paths, padding_strategy=padding, pos_encoding_method=positional_encodings)
    val_dataset = HDF5Dataset(val_file_paths, padding_strategy=padding, pos_encoding_method=positional_encodings)
    test_dataset = HDF5Dataset(test_file_paths, padding_strategy=padding, pos_encoding_method=positional_encodings)

    # Create samplers and dataloaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch,
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(rank)
            optimizer.zero_grad()

            row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(rank)
            reconstructed = model(batch_data, row_mask)

            loss = criterion(
                reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Average training loss across all processes
        train_loss = train_loss / num_batches
        train_loss_tensor = torch.tensor(train_loss).to(rank)
        dist.reduce(train_loss_tensor, 0, op=dist.ReduceOp.SUM)

        if rank == 0:
            train_loss = train_loss_tensor.item() / world_size
            print(f"Epoch {epoch + 1} Average Training Loss: {train_loss:.4f}")

            # Validation (only on rank 0)
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    batch_data = batch_data.to(rank)
                    row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(rank)
                    reconstructed = model(batch_data, row_mask)
                    loss = criterion(
                        reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                        batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
                    )
                    val_loss += loss.item()
                    val_batches += 1
            print(f"Epoch {epoch + 1} Validation Loss: {val_loss / val_batches:.4f}")

    # Final test (only on rank 0)
    if rank == 0:
        model.eval()
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(rank)
                row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(rank)
                reconstructed = model(batch_data, row_mask)
                loss = criterion(
                    reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                    batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
                )
                test_loss += loss.item()
                test_batches += 1
        print(f"Final Test Loss: {test_loss / test_batches:.4f}")

        # Save model
        model_save_path = "trained_mae_model.pt"
        torch.save(model.module.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    cleanup()


def main():
    """
    Main function to set up distributed training
    """

    class Args:
        def __init__(self):
            self.batch_size = 4
            self.num_epochs = 2
            self.learning_rate = 1e-4
            self.mask_ratio = 0.25
            self.padding = 'zero'
            self.positional_encodings = 'add'
            self.embedding_dim = 512
            self.number_heads = 8
            self.layers = 6

    args = Args()

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")

    if world_size > 1:
        mp.spawn(
            train,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        train(0, 1, args)


if __name__ == "__main__":
    main()