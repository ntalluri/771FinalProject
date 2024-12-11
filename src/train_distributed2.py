import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from dataloader import HDF5IterableDataset, generate_row_mask
from MAE import MAEModel
import random
import datetime
import argparse

class EarlyStopping:
    """
    Early stops the training if the monitored metric does not improve after a given patience.
    """
    def __init__(self, patience=3, verbose=False, delta=0.0, path='best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, loss, model, optimizer, scheduler=None):
        score = -loss  # Since we want to minimize loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model, optimizer, scheduler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model, optimizer, scheduler)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler=None):
        '''Saves the entire model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler:
            state['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(state, self.path)
        self.best_loss = val_loss

def gather_all_file_paths(root_dirs, exclude_dirs=None):
    """
    Gather all HDF5 file paths from the immediate subdirectories of the given root directories,
    excluding the specified directories.
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    if exclude_dirs is None:
        exclude_dirs = set()
    else:
        # Convert to set for faster lookup
        exclude_dirs = set(os.path.abspath(d) for d in exclude_dirs)

    file_paths = []
    for root_dir in root_dirs:
        root_dir = os.path.abspath(root_dir)
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
            subdir_path = os.path.abspath(subdir_path)
            if subdir_path in exclude_dirs:
                print(f"Excluding directory: {subdir_path}")
                continue
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

def compute_batch_pearson_correlation(reconstructed, batch_data, row_mask):
    """
    Compute the average absolute Pearson correlation coefficient for a batch.
    """
    batch_size = reconstructed.size(0)
    correlations = []
    for i in range(batch_size):
        mask = row_mask[i]
        x = reconstructed[i][mask]
        y = batch_data[i][mask]
        if x.numel() == 0:
            continue
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        x_mean = torch.mean(x_flat)
        y_mean = torch.mean(y_flat)
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered ** 2)) * torch.sqrt(torch.sum(y_centered ** 2))
        correlation = torch.abs(numerator / (denominator + 1e-8))  # Added abs()
        correlations.append(correlation.item())
    
    return sum(correlations) / len(correlations) if correlations else 0.0

def setup_distributed(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def save_encoder(best_model, device, save_path="trained_encoder_state_dict.pt"):
    print("Extracting the encoder from the best model.")
    # Access the encoder
    if isinstance(best_model, nn.parallel.DistributedDataParallel):
        encoder = best_model.module.encoder
    else:
        encoder = best_model.encoder

    # Initialize a new encoder instance (same architecture)
    encoder_model = MAEModel(
        input_dim=best_model.module.input_dim if isinstance(best_model, nn.parallel.DistributedDataParallel) else best_model.input_dim,
        embed_dim=best_model.module.embed_dim if isinstance(best_model, nn.parallel.DistributedDataParallel) else best_model.embed_dim,
        num_heads=best_model.module.num_heads if isinstance(best_model, nn.parallel.DistributedDataParallel) else best_model.num_heads,
        depth=best_model.module.depth if isinstance(best_model, nn.parallel.DistributedDataParallel) else best_model.depth
    ).to(device)

    # Load the encoder's state_dict from the best model
    encoder_model.load_state_dict(encoder.state_dict())
    encoder_model.eval()
    print("Encoder extracted successfully.")

    # Save the encoder's state_dict
    torch.save(encoder_model.state_dict(), save_path)
    print(f"Trained encoder's state_dict saved to {save_path}")

def train_model(rank, world_size, args):
    """
    Distributed training function.
    """
    setup_distributed(rank, world_size)
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Gather all file paths
    all_file_paths = gather_all_file_paths(args.root_dirs, exclude_dirs=args.exclude_dirs)

    # Shuffle and split the file paths
    random.shuffle(all_file_paths)
    len_paths = len(all_file_paths)
    num_keep = int(len_paths * args.file_prop)
    all_file_paths = all_file_paths[:num_keep]
    
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
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
    train_dataset = HDF5IterableDataset(train_file_paths, padding_strategy=args.padding, pos_encoding_method=args.pos_encoding)
    val_dataset = HDF5IterableDataset(val_file_paths, padding_strategy=args.padding, pos_encoding_method=args.pos_encoding)
    test_dataset = HDF5IterableDataset(test_file_paths, padding_strategy=args.padding, pos_encoding_method=args.pos_encoding)

    # Create samplers and dataloaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Initialize the model
    model = MAEModel(
        input_dim=args.required_columns,
        embed_dim=args.embedding_dim,
        num_heads=args.number_heads,
        depth=args.layers
    ).to(device)

    # Wrap the model with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None  # Add scheduler if needed

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=args.delta, path='best_model.pt')

    # Setup TensorBoard
    if rank == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
        logs_dir = os.path.join(project_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_log_dir = os.path.join(logs_dir, f"run_{timestamp}")
        writer = SummaryWriter(log_dir=unique_log_dir)
        print(f"TensorBoard logs will be saved to: {unique_log_dir}")
    else:
        writer = None

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_corr = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            # Generate row masks for the batch
            row_mask = generate_row_mask(batch_data.size(0), args.max_rows, args.mask_ratio).to(device)
            reconstructed = model(batch_data, row_mask)

            # Compute loss
            loss = criterion(
                reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Compute Pearson correlation
            corr = compute_batch_pearson_correlation(reconstructed, batch_data, row_mask)
            train_corr += corr

            num_batches += 1

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Reduce loss and correlation across all processes
        train_loss_tensor = torch.tensor(train_loss).to(device)
        train_corr_tensor = torch.tensor(train_corr).to(device)
        dist.reduce(train_loss_tensor, 0, op=dist.ReduceOp.SUM)
        dist.reduce(train_corr_tensor, 0, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_train_loss = train_loss_tensor.item() / world_size / num_batches
            avg_train_corr = train_corr_tensor.item() / world_size / num_batches
            print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f} Pearson Correlation: {avg_train_corr:.4f}")

            # Log training metrics
            if writer:
                writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
                writer.add_scalar('Correlation/Train', avg_train_corr, epoch + 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_corr = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                row_mask = generate_row_mask(batch_data.size(0), args.max_rows, args.mask_ratio).to(device)
                reconstructed = model(batch_data, row_mask)

                # Compute loss
                loss = criterion(
                    reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                    batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
                )
                val_loss += loss.item()

                # Compute Pearson correlation
                corr = compute_batch_pearson_correlation(reconstructed, batch_data, row_mask)
                val_corr += corr

                val_batches += 1

        # Reduce validation loss and correlation
        val_loss_tensor = torch.tensor(val_loss).to(device)
        val_corr_tensor = torch.tensor(val_corr).to(device)
        dist.reduce(val_loss_tensor, 0, op=dist.ReduceOp.SUM)
        dist.reduce(val_corr_tensor, 0, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_val_loss = val_loss_tensor.item() / world_size / val_batches
            avg_val_corr = val_corr_tensor.item() / world_size / val_batches
            print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f} Pearson Correlation: {avg_val_corr:.4f}")

            # Log validation metrics
            if writer:
                writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
                writer.add_scalar('Correlation/Validation', avg_val_corr, epoch + 1)

            # Early Stopping check
            early_stopping(avg_val_loss, model, optimizer, scheduler)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

    # After training, only rank 0 saves the best model and evaluates on the test set
    if rank == 0:
        print("Training complete. Loading the best saved model for testing.")
        checkpoint = torch.load('best_model.pt', map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Best model loaded successfully.")

        # Final Evaluation on Test Set
        model.eval()
        test_loss = 0.0
        test_corr = 0.0
        test_batches = 0

        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                row_mask = generate_row_mask(batch_data.size(0), args.max_rows, args.mask_ratio).to(device)
                reconstructed = model(batch_data, row_mask)

                # Compute loss
                loss = criterion(
                    reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                    batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
                )
                test_loss += loss.item()

                # Compute Pearson correlation
                corr = compute_batch_pearson_correlation(reconstructed, batch_data, row_mask)
                test_corr += corr

                test_batches += 1

        avg_test_loss = test_loss / test_batches
        avg_test_corr = test_corr / test_batches
        print(f"Final Test Loss: {avg_test_loss:.4f} Pearson Correlation: {avg_test_corr:.4f}")

        # Log test metrics
        if writer:
            writer.add_scalar('Loss/Test', avg_test_loss, epoch + 1)
            writer.add_scalar('Correlation/Test', avg_test_corr, epoch + 1)

        # Save the encoder
        save_encoder(model, device)

        # Close the TensorBoard writer
        if writer:
            writer.close()

    cleanup_distributed()

def main():
    folder_paths = [
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220509_00",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220513_16",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_05",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220509_20",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220513_22",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220519_17",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220509_10",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220511_19",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_15",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230201_05",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220511_06",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_22",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220513_19",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230201_20",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220507_13",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220514_03",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220516_02",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230201_02",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220509_12",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220518_08",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220509_11",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220516_09",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220505_04",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220513_00",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_07",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_17",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_00",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220508_07",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220510_15",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_08",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220518_11",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220510_05",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_08",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220514_12",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220518_06",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220517_06",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230201_22",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220517_00",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220507_11",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220511_20",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_15",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230202_06",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_21",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_11",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_23",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_03",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220518_17",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_14",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230202_09",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220510_23",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220505_23",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230202_18",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220509_07",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220517_15",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220519_08",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220509_09",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220518_19",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220513_10",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_02",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_10",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230201_21",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230202_15",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220519_06",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220517_03",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220507_22",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_20",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220513_02",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_06",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220517_22",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230202_00",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220509_13",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_21",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220511_04",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220511_14",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220511_12",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220517_19",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220508_11",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220517_09",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_17",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220519_21",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220516_14",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220508_02",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220507_20",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220519_10",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_03",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220507_06",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_19",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220518_10",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220505_03",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220514_00",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_12",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220515_18",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_22",
        "/media/jesse/sda/Organized_SURF/seis_sensor_processed_20230201_12",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220512_05",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220518_18",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220514_07",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220507_16",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220513_18",
        "/media/jesse/sdb/Organized_SURF/seis_sensor_processed_20220506_13",
    ]

    parser = argparse.ArgumentParser(description="Distributed MAE Model Training")
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use for training')
    parser.add_argument('--batch_size', type=int, default=6, help='Input batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='Mask ratio for MAE')
    parser.add_argument('--padding', type=str, default='zero', help='Padding strategy')
    parser.add_argument('--pos_encoding', type=str, default='add', help='Positional encoding method')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--number_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--delta', type=float, default=0.0, help='Early stopping delta')
    parser.add_argument('--required_columns', type=int, default=30000, help='Required number of columns')
    parser.add_argument('--max_rows', type=int, default=1387, help='Maximum number of rows')
    parser.add_argument('--root_dirs', type=str, nargs='+', default=["/media/jesse/sda/Organized_SURF", "/media/jesse/sdb/Organized_SURF"], help='Root directories containing data')
    parser.add_argument('--exclude_dirs', type=str, nargs='*', default=folder_paths, help='Directories to exclude')
    parser.add_argument('--file_prop', type=float, default=0.25, help='Proportion of total files to use')
    
    args = parser.parse_args()

    world_size = args.world_size
    if world_size > 1:
        mp.spawn(
            train_model,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        train_model(0, 1, args)

if __name__ == "__main__":
    main()
