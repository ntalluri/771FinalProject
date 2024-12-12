import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter
from sklearn.model_selection import train_test_split
from dataloader import HDF5IterableDataset, generate_row_mask
from MAE import MAEModel, Encoder
import torch.nn as nn
import random 
import datetime
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class EarlyStopping:
    """
    Early stops the training if the monitored metric does not improve after a given patience.
    """
    def __init__(self, patience=3, verbose=False, delta=0.0, path='best_model.pt'):
        """
        Args:
            patience (int): How long to wait after last time monitored metric improved.
            verbose (bool): If True, prints a message for each improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, loss, model):
        score = -loss  # Since we want to minimize loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves the entire model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

def gather_all_file_paths(root_dirs, exclude_dirs=None):
    """
    Gather all HDF5 file paths from the immediate subdirectories of the given root directories,
    excluding the specified directories.

    Args:
        root_dirs (list or str): List of root directories or a single root directory
        exclude_dirs (list or set, optional): List or set of subdirectory paths to exclude

    Returns:
        list: List of HDF5 file paths
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
    
    Args:
        reconstructed (torch.Tensor): Reconstructed data from the model.
        batch_data (torch.Tensor): Original batch data.
        row_mask (torch.Tensor): Mask indicating which rows were processed.
    
    Returns:
        float: Average absolute Pearson correlation coefficient for the batch.
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

# list of folder paths
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

# tunable parameters
batch = 6
num_epochs = 10
learning_rate = 0.0001
mask_ratio = 0.25
padding = 'zero'
positional_encodings = 'add'
embedding_dim = 512
number_heads = 8
layers = 4

file_prop = 0.001

# not tunable parameters 
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000
model = MAEModel(
    input_dim=REQUIRED_COLUMNS,
    embed_dim=embedding_dim,
    num_heads=number_heads,
    depth=layers
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
else:
    print("No GPUs, using CPUs")

model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

root_dirs = ["/media/jesse/sda/Organized_SURF", "/media/jesse/sdb/Organized_SURF"]
all_file_paths = gather_all_file_paths(root_dirs, folder_paths)

# Shuffle and split the file paths
random.shuffle(all_file_paths)
len_paths = len(all_file_paths)
num_keep = int(len_paths * file_prop)
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
print(f"Total files: {total_files}")
print(f"Training files: {len(train_file_paths)}")
print(f"Validation files: {len(val_file_paths)}")
print(f"Testing files: {len(test_file_paths)}")

# Create datasets
train_dataset = HDF5IterableDataset(train_file_paths, padding_strategy = padding, pos_encoding_method=positional_encodings)
val_dataset = HDF5IterableDataset(val_file_paths, padding_strategy = padding, pos_encoding_method=positional_encodings)
test_dataset = HDF5IterableDataset(test_file_paths, padding_strategy = padding, pos_encoding_method=positional_encodings)
 
# Create data loaders
# TODO: is this loading all the data prior to training? 
# TODO: could we load per batch instead. A dataloader per batch?
train_loader = DataLoader(train_dataset, batch_size=batch)
val_loader = DataLoader(val_dataset, batch_size=batch)
test_loader = DataLoader(test_dataset, batch_size=batch)

# Determine the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory (project root)
project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Define the logs directory path
logs_dir = os.path.join(project_dir, 'logs')

# Create the logs directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# Generate a unique subdirectory name using the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
unique_log_dir = os.path.join(logs_dir, f"run_{timestamp}")

# Initialize the TensorBoard SummaryWriter with the unique log directory
writer = SummaryWriter(log_dir=unique_log_dir)

print(f"TensorBoard logs will be saved to: {unique_log_dir}")

# Initialize EarlyStopping
early_stopping = EarlyStopping(patience=3, verbose=True, delta=0.0, path='best_model.pt')

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    
    # training
    model.train()
    train_loss = 0.0
    train_corr = 0.0
    num_batches = 0

    for batch_idx, batch_data in enumerate(train_loader):
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        
        # generate row masks for the batch
        row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(device)
        reconstructed = model(batch_data, row_mask)
        # reconstructed.to(device)
        
        # compute loss
        loss = criterion(
            reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
            batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
        )

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # compute pearson correlation
        corr = compute_batch_pearson_correlation(reconstructed, batch_data, row_mask)
        train_corr += corr
            
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

    train_loss /= max(num_batches, 1)
    train_corr /= max(num_batches, 1)
    
    print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f} Pearson Correlation: {train_corr:.4f}")
    
    
    # Log training loss to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch + 1)
    writer.add_scalar('Correlation/Train', train_corr, epoch + 1)

    model.eval()
    val_loss = 0.0
    val_corr = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device)
            row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(device)
            reconstructed = model(batch_data, row_mask)
            # reconstructed.to(device)
            
            # compute loss
            loss = criterion(
                reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
            )
            val_loss += loss.item()
            
            # compute pearson correlation
            corr = compute_batch_pearson_correlation(reconstructed, batch_data, row_mask)
            val_corr += corr
            
            val_batches += 1

    val_loss /= max(val_batches, 1)
    val_corr /= max(val_batches, 1)
    print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f} Pearson Correlation: {val_corr:.4f}")
    
    # Log validation loss to TensorBoard
    writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
    writer.add_scalar('Correlation/Validation', val_corr, epoch + 1)

    # Early Stopping check
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.")
        break

# After training is complete, load the best saved model
print("Training complete. Loading the best saved model for testing.")

# Initialize a new MAEModel instance (same architecture)
best_model = MAEModel(
    input_dim=REQUIRED_COLUMNS,
    embed_dim=embedding_dim,
    num_heads=number_heads,
    depth=layers
)

# Handle DataParallel wrapping
if torch.cuda.device_count() >= 1:
    best_model = nn.DataParallel(best_model)

best_model.to(device)

# Load the saved state_dict
best_model_path = 'best_model.pt'
best_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
print("Best model loaded successfully.")

# Final Evaluation on Test Set using the loaded best MAE model
best_model.eval()
test_loss = 0.0
test_corr = 0.0
test_batches = 0
with torch.no_grad():
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(device)
        reconstructed = best_model(batch_data, row_mask)
        # reconstructed.to(device)
        
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

test_loss /= max(test_batches, 1)
test_corr /= max(test_batches, 1)
print(f"Final Test Loss: {test_loss:.4f} Pearson Correlation: {test_corr:.4f}")

# Log final test metrics to TensorBoard
writer.add_scalar('Loss/Test', test_loss, epoch + 1)
writer.add_scalar('Correlation/Test', test_corr, epoch + 1)

# Extract the encoder from the loaded best model and save its state_dict separately
print("Extracting the encoder from the best model.")

# Access the encoder
if hasattr(best_model, 'module'):
    best_model = best_model.module
encoder = best_model.encoder

# Initialize a new Encoder instance with the same architecture
encoder_model = Encoder(
    input_dim=best_model.input_dim,
    embed_dim=best_model.embed_dim,
    num_heads=best_model.num_heads,
    depth=best_model.depth
).to(device)

# Load the encoder's state_dict into the encoder_model
encoder_model.load_state_dict(encoder.state_dict())
encoder_model.eval()
print("Encoder extracted successfully.")

# Save the encoder's state_dict
save_path = "trained_encoder_state_dict.pt"
torch.save(encoder_model.state_dict(), save_path)
print(f"Trained encoder's state_dict saved to {save_path}")

# Close the TensorBoard writer
writer.close()
