import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataloader import HDF5IterableDataset, generate_row_mask
from mae import MAEModel
import torch.nn as nn

# parameters
file_dir = "data/Toy_dataset"
batch_size = 16
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000
num_epochs = 1
mask_ratio = 0.15
train_ratio = 0.7
val_ratio = 0.15  # the test ratio will be 1 - (train_ratio + val_ratio)

# get all file paths
all_file_paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.h5')]

# shuffle and split the dataset
train_paths, test_paths = train_test_split(all_file_paths, test_size=(1 - train_ratio), random_state=42)
val_paths, test_paths = train_test_split(test_paths, test_size=(1 - val_ratio / (1 - train_ratio)), random_state=42)

# datasets for each split
train_dataset = HDF5IterableDataset(file_dir=file_dir)
train_dataset.file_paths = train_paths
val_dataset = HDF5IterableDataset(file_dir=file_dir)
val_dataset.file_paths = val_paths
test_dataset = HDF5IterableDataset(file_dir=file_dir)
test_dataset.file_paths = test_paths

# create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

# training loop
model = MAEModel(input_dim=REQUIRED_COLUMNS, embed_dim=512, num_heads=8, depth=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    
    # training
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        
        # generate row masks for the batch
        # TODO: the error is happening here beacause of the batch_size I have present for the parameters being used, insterad of it being the batcha data index 0 size I think
        row_mask = generate_row_mask(batch_size, MAX_ROWS, mask_ratio=mask_ratio).to(batch_data.device)
        
        # forward pass
        reconstructed = model(batch_data, row_mask)
        
        # compute loss
        loss = criterion(
            reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
            batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
        )
        
         # back pass
        loss.backward()
        optimizer.step()
        print(f"  Train Batch {batch_idx}, Loss: {loss.item()}")

    # val
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            # generate row masks for the batch
            row_mask = generate_row_mask(batch_size, MAX_ROWS, mask_ratio=mask_ratio).to(batch_data.device)
            
            # forward pass
            reconstructed = model(batch_data, row_mask)
            
            # compute loss
            loss = criterion(
                reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
                batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
            )
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"  Validation Loss: {val_loss}")

# test
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_loader):
        # generate row masks for the batch
        row_mask = generate_row_mask(batch_size, MAX_ROWS, mask_ratio=mask_ratio).to(batch_data.device)
        
        # forward pass
        reconstructed = model(batch_data, row_mask)
        
        # compute loss
        loss = criterion(
            reconstructed[row_mask.unsqueeze(-1).expand_as(reconstructed)],
            batch_data[row_mask.unsqueeze(-1).expand_as(batch_data)]
        )
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss}")