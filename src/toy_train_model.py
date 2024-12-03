import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import HDF5IterableDataset, generate_row_mask
from MAE import MAEModel
import torch.nn as nn

file_dir = "../data/Toy_dataset"
batch_size = 4 # I run out of memory with higher, but it probably should be higher
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000
num_epochs = 2
mask_ratio = 0.25 # from our midterm doc
train_ratio = 0.7
val_ratio = 0.15  # the test ratio will be 1 - (train_ratio + val_ratio)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

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

model = MAEModel(
    input_dim=REQUIRED_COLUMNS,
    embed_dim=512,
    num_heads=8,
    depth=6
)


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    
    # training
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        
        # generate row masks for the batch
        # TODO: the error is happening here beacause of the batch_size I have present for the parameters being used, insterad of it being the batcha data index 0 size I think

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