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

# tunable parameters
# TODO: update this to do a paramter sweep 
batch = 4
num_epochs = 2
learning_rate = 1e-4
mask_ratio = 0.25
padding = 'zero'
positional_encodings = 'add'
embedding_dim = 512
number_heads = 8
layers = 6

# not tunable parameters 
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000
model = MAEModel(
    input_dim=REQUIRED_COLUMNS,
    embed_dim=embedding_dim,
    num_heads=number_heads,
    depth=layers
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# List of folder paths
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


all_file_paths = []
for subdir_path in folder_paths:
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
    all_file_paths.extend(h5_files)

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
train_dataset = HDF5IterableDataset(train_file_paths, padding_strategy = padding, pos_encoding_method=positional_encodings)
val_dataset = HDF5IterableDataset(val_file_paths, padding_strategy = padding, pos_encoding_method=positional_encodings)
test_dataset = HDF5IterableDataset(test_file_paths, padding_strategy = padding, pos_encoding_method=positional_encodings)
 
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch)
val_loader = DataLoader(val_dataset, batch_size=batch)
test_loader = DataLoader(test_dataset, batch_size=batch)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    
    # training
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        
        # generate row masks for the batch
        row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(batch_data.device)
        reconstructed = model(batch_data, row_mask)
        
        # compute loss
        loss = criterion(
            reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
            batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
        )

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

    train_loss /= max(num_batches, 1)
    print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch_data in val_loader:
            row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(batch_data.device)
            reconstructed = model(batch_data, row_mask)
            
            # compute loss
            loss = criterion(
                reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
            )
            val_loss += loss.item()
            val_batches += 1

    val_loss /= max(val_batches, 1)
    print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

model.eval()
test_loss = 0.0
test_batches = 0
with torch.no_grad():
        # generate row masks for the batch
        # forward pass
    for batch_data in test_loader:
        row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(batch_data.device)
        reconstructed = model(batch_data, row_mask)
        
        # compute loss
        loss = criterion(
            reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
            batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
        )
        test_loss += loss.item()
        test_batches += 1

test_loss /= max(test_batches, 1)
print(f"Final Test Loss: {test_loss:.4f}")

model_save_path = "trained_mae_model.pt"
torch.save(model, model_save_path)
print(f"Model saved to {model_save_path}")