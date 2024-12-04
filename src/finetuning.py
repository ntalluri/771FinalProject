import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import HDF5IterableDataset, generate_row_mask
from MAE import MAEModel
import torch.nn as nn
import random 
import itertools
import pandas as pd
import copy

# tunable parameters and their ranges
param_grid = {
    'batch_size': [4, 8],
    'num_epochs': [2],
    'learning_rate': [1e-4, 1e-3],
    'mask_ratio': [0.25, 0.5],
    'padding': ['zero', 'repeat', 'noise', 'mirror'],
    'positional_encodings': ['add', 'concat'],
    'embedding_dim': [256, 512],
    'number_heads': [4, 8],
    'layers': [4, 6]
}

# not tunable parameters 
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000

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

all_file_paths = []
for subdir_path in folder_paths:
    # list files ending with '.h5' in the current subdirectory (non-recursive)
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

# shuffle and split the file paths
random.shuffle(all_file_paths)
# TODO make this work for mutliple gpus
# TODO: we only need train and val, since we are only finetuning
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
total_files = len(all_file_paths)
train_end = int(train_ratio * total_files)
val_end = train_end + int(val_ratio * total_files)
train_file_paths = all_file_paths[:train_end]
val_file_paths = all_file_paths[train_end:val_end]
test_file_paths = all_file_paths[val_end:]

# create datasets
def create_datasets(padding_strategy, pos_encoding_method):
    train_dataset = HDF5IterableDataset(train_file_paths, padding_strategy=padding_strategy, pos_encoding_method=pos_encoding_method)
    val_dataset = HDF5IterableDataset(val_file_paths, padding_strategy=padding_strategy, pos_encoding_method=pos_encoding_method)
    test_dataset = HDF5IterableDataset(test_file_paths, padding_strategy=padding_strategy, pos_encoding_method=pos_encoding_method)
    return train_dataset, val_dataset, test_dataset

# function to compute Pearson correlation coefficient
# TODO: I have no idea if this will work
def compute_batch_pearson_correlation(reconstructed, batch_data, row_mask):
    # reconstructed and batch_data are tensors of shape [batch_size, MAX_ROWS, REQUIRED_COLUMNS]
    # row_mask is of shape [batch_size, MAX_ROWS]
    # We want to compute the Pearson correlation over the masked positions
    # For each sample in the batch, we select the masked rows
    batch_size = reconstructed.size(0)
    correlations = []
    for i in range(batch_size):
        mask = row_mask[i]  # shape: [MAX_ROWS]
        x = reconstructed[i][mask]  # shape: [num_masked_rows, REQUIRED_COLUMNS]
        y = batch_data[i][mask]     # same shape
        if x.numel() == 0:
            continue  # skip samples with no masked rows
        x_flat = x.flatten()
        y_flat = y.flatten()
        # Compute Pearson correlation between x_flat and y_flat
        x_mean = torch.mean(x_flat)
        y_mean = torch.mean(y_flat)
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered ** 2)) * torch.sqrt(torch.sum(y_centered ** 2))
        correlation = numerator / (denominator + 1e-8)  # add epsilon to prevent division by zero
        correlations.append(correlation.item())
    if correlations:
        return sum(correlations) / len(correlations)
    else:
        return 0.0

# Training and evaluation function
def train_and_evaluate(params):
    # unpack parameters
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    mask_ratio = params['mask_ratio']
    padding = params['padding']
    positional_encodings = params['positional_encodings']
    embedding_dim = params['embedding_dim']
    number_heads = params['number_heads']
    layers = params['layers']
    
    print(f"Training with parameters: {params}")
    
    # create datasets and loaders
    # TODO: I want this to only happen onces, not everytime the function is called
    # TODO: is this loading all the data prior to training? 
    # TODO: could we load per batch instead. A dataloader per batch?
    train_dataset, val_dataset, test_dataset = create_datasets(padding, positional_encodings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # init model, criterion, optimizer
    model = MAEModel(
        input_dim=REQUIRED_COLUMNS,
        embed_dim=embedding_dim,
        num_heads=number_heads,
        depth=layers
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_corr = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(batch_data.device)
            reconstructed = model(batch_data, row_mask)
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

        train_loss /= max(num_batches, 1)
        train_corr /= max(num_batches, 1)

        # validation loop
        model.eval()
        val_loss = 0.0
        val_corr = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(batch_data.device)
                reconstructed = model(batch_data, row_mask)
                loss = criterion(
                    reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                    batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
                )
                val_loss += loss.item()

                # compute Pearson correlation
                corr = compute_batch_pearson_correlation(reconstructed, batch_data, row_mask)
                val_corr += corr

                val_batches += 1

        val_loss /= max(val_batches, 1)
        val_corr /= max(val_batches, 1)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Corr: {train_corr:.4f}, Validation Corr: {val_corr:.4f}")
    
    # Testing loop
    model.eval()
    test_loss = 0.0
    test_corr = 0.0
    test_batches = 0
    with torch.no_grad():
        for batch_data in test_loader:
            row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(batch_data.device)
            reconstructed = model(batch_data, row_mask)
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
    print(f"Test Loss: {test_loss:.4f}, Test Corr: {test_corr:.4f}")
    
    return val_loss, test_loss, val_corr, test_corr, copy.deepcopy(model)

# Parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

# Store results
results = []
# best_val_loss = float('inf')
best_val_corr = float('-inf')
best_model = None
best_params = None

# Loop over all parameter combinations
for param_values in param_combinations:
    params = dict(zip(param_names, param_values))
    val_loss, test_loss, val_corr, test_corr, model = train_and_evaluate(params)
    results.append({
        **params,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'val_corr': val_corr,
        'test_corr': test_corr
    })
    
    # Save the best model based on validation loss
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     best_model = model
    #     best_params = params

    # Save the best model based on validation correlation
    if val_corr > best_val_corr:
        best_val_corr = val_corr
        best_model = model
        best_params = params

# Save the best model
model_save_path = "best_trained_mae_model.pt"
torch.save(best_model.state_dict(), model_save_path)
print(f"Best model saved to {model_save_path} with parameters {best_params}")

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('parameter_tuning_results.csv', index=False)
print("Results saved to parameter_tuning_results.csv")