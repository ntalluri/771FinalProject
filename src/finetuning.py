import os
import torch
from torch.utils.data import DataLoader
from dataloader import HDF5IterableDataset, generate_row_mask
from MAE import MAEModel
import torch.nn as nn
import random 
import itertools
import pandas as pd
import copy

# tunable parameters and their ranges
param_grid = {
    'batch_size': [6],
    'num_epochs': [2, 4],
    'learning_rate': [1e-4, 1e-3],
    'mask_ratio': [0.25, 0.5],
    'padding': ['zero'],
    'positional_encodings': ['add'],
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
]

all_file_paths = []
for subdir_path in folder_paths:
    # list files ending with '.h5' in the current subdirectory
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
train_ratio = 0.8
val_ratio = 0.2
total_files = len(all_file_paths)
train_end = int(train_ratio * total_files)
train_file_paths = all_file_paths[:train_end]
val_file_paths = all_file_paths[train_end:]


# function to compute Pearson correlation coefficient
def compute_batch_pearson_correlation(reconstructed, batch_data, row_mask):
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

# training and evaluation function
def train_and_evaluate(params, train_loader, val_loader):
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
    
    # init model, criterion, optimizer
    model = MAEModel(
        input_dim=REQUIRED_COLUMNS,
        embed_dim=embedding_dim,
        num_heads=number_heads,
        depth=layers
    )

    # NEW
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        print("No GPUs, using CPUs")

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")

        model.train()
        train_loss = 0.0
        train_corr = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            print(batch_idx)
            batch_data = batch_data.to(device)  # NEW
            optimizer.zero_grad()
            row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(device) # NEW
            reconstructed = model(batch_data, row_mask)
            reconstructed.to(device)
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
                batch_data = batch_data.to(device)  # New 
                row_mask = generate_row_mask(batch_data.size(0), MAX_ROWS, mask_ratio).to(device) # New
                reconstructed = model(batch_data, row_mask)
                reconstructed.to(device)
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
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Corr: {train_corr:.4f}, Validation Corr: {val_corr:.4f}")
    
    return val_loss, val_corr, copy.deepcopy(model.module) # when saving or returning the model, access the underlying model wrapped by DataParallel

# the parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

# store all the results
results = []
best_val_loss = float('inf')
best_model = None
best_params = None

# cache of the dataloaders for a specific padding and positional_encoding
dataset_cache = {}

# loop over all parameter combinations
for param_values in param_combinations[8:]:  # Skip the first 8 combinations
    params = dict(zip(param_names, param_values))

    # create a unique key for the dataset cache based on padding and positional_encodings
    dataset_key = (params["padding"], params["positional_encodings"])

    if dataset_key not in dataset_cache:
        # create datasets and loaders only once per unique padding and positional_encodings
        train_dataset = HDF5IterableDataset(
            train_file_paths, 
            padding_strategy=params["padding"], 
            pos_encoding_method=params["positional_encodings"]
        )
        val_dataset = HDF5IterableDataset(
            val_file_paths, 
            padding_strategy = params["padding"],
            pos_encoding_method = params["positional_encodings"]
        )

        print("making train loader")
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=0)
        print("making val loader")
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=0)

        # cache the loaders
        print("caching the train and val loader")
        dataset_cache[dataset_key] = (train_loader, val_loader)
    else:
        # retrieve cached loaders
        train_loader, val_loader = dataset_cache[dataset_key]

    # train and evaluate
    print("train and evaluate")
    val_loss, val_corr, model = train_and_evaluate(params, train_loader, val_loader)
    results.append({
        **params,
        'val_loss': val_loss,
        'val_corr': val_corr,
    })

    df = pd.DataFrame(results)
    df.to_csv("parameter_tuning_results.csv", mode='a', index=False)
    print("Results saved to current parameter_tuning_results.csv")
    
    # save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_params = params

# save the best model
model_save_path = "best_trained_mae_model.pt"
torch.save(best_model.state_dict(), model_save_path)
print(f"Best model saved to {model_save_path} with parameters {best_params}")

# save all of the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('all_parameter_tuning_results.csv', index=False)
print("Results saved to parameter_tuning_results.csv")