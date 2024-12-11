import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import HDF5IterableDataset, generate_row_mask
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# take the trained encoder, add a binary classifier nn on top to classify 1 for event 0 for no event
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Constants for data dimensions
MIN_ROWS = 1357
MAX_ROWS = 1387
REQUIRED_COLUMNS = 30000

def get_all_file_paths(folder_paths):
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
    return all_file_paths

def load_labels(csv_path):
    """
    Load labels from CSV file
    """
    labels_df = pd.read_csv('labeled_filenames.csv')
    return dict(zip(labels_df['Filename'], labels_df['Label']))

def compute_metrics(y_true, y_pred):
   acc = accuracy_score(y_true, y_pred)
   f1 = f1_score(y_true, y_pred)
   recall = recall_score(y_true, y_pred)
   precision = precision_score(y_true, y_pred)
   return acc, f1, recall, precision

class BinaryClassifier(nn.Module):
    def __init__(self, encoder, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # get encoder output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, MAX_ROWS, REQUIRED_COLUMNS).to(device)
            encoder_output = self.encoder(dummy_input)
            encoder_output_dim = encoder_output.shape[-1]
            # TODO: figure out what this is and hopefully the output shape is the same for all the embeddings
            # Delete after
            print(f"encoder_output shape: {encoder_output.shape}")
            print(f"encoder_output_dim: {encoder_output_dim}")
        
        # TODO: we can finetune this to change the layers
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if isinstance(self.encoder, nn.DataParallel):
            encoded = self.encoder.module(x)
        else:
            encoded = self.encoder(x)
        return self.classifier(encoded).squeeze()

def train_classifier(model, train_loader, val_loader, num_epochs=10):
   criterion = nn.BCELoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-4)
   writer = SummaryWriter('runs/binary_classifier')
   
   for epoch in range(num_epochs):
       # training
       model.train()
       train_loss = 0.0
       train_preds = []
       train_labels = []
       
       for data, labels in train_loader:
           data, labels = data.to(device), labels.to(device)
           optimizer.zero_grad()
           outputs = model(data)
           loss = criterion(outputs, labels.float())
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
           predictions = (outputs >= 0.5).float()
           
           train_preds.extend(predictions.cpu().numpy())
           train_labels.extend(labels.cpu().numpy())
       
       train_loss = train_loss / len(train_loader)
       train_acc, train_f1, train_recall, train_precision = compute_metrics(train_labels, train_preds)
       
       # validation
       model.eval()
       val_loss = 0.0
       val_preds = []
       val_labels = []
       
       with torch.no_grad():
           for data, labels in val_loader:
               data, labels = data.to(device), labels.to(device)
               outputs = model(data)
               loss = criterion(outputs, labels.float())
               val_loss += loss.item()
               predictions = (outputs >= 0.5).float()
               
               val_preds.extend(predictions.cpu().numpy())
               val_labels.extend(labels.cpu().numpy())
       
       val_loss = val_loss / len(val_loader)
       val_acc, val_f1, val_recall, val_precision = compute_metrics(val_labels, val_preds)
       
       # log metrics
       writer.add_scalar('Loss/Train', train_loss, epoch)
       writer.add_scalar('Loss/Val', val_loss, epoch)
       writer.add_scalar('Accuracy/Train', train_acc, epoch)
       writer.add_scalar('Accuracy/Val', val_acc, epoch)
       writer.add_scalar('F1/Train', train_f1, epoch)
       writer.add_scalar('F1/Val', val_f1, epoch)
       writer.add_scalar('Recall/Train', train_recall, epoch)
       writer.add_scalar('Recall/Val', val_recall, epoch)
       writer.add_scalar('Precision/Train', train_precision, epoch)
       writer.add_scalar('Precision/Val', val_precision, epoch)
       
       print(f"Epoch {epoch+1}/{num_epochs}:")
       print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}")
       print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}\n")
   
   writer.close()
   return model

# grab the labels dict
labels_dict = load_labels("labeled_filenames.csv")

# grab the subfolders
folder_paths = []
with open('labeled_folders.txt', 'r') as file:
    folder_paths = file.read().splitlines()

# get all the H5 files 
all_file_paths = get_all_file_paths(folder_paths)

# 70% train, 15% val, 15% test 
# TODO: decide if we need val; if we finetune, we need val
train_files, temp_files = train_test_split(all_file_paths, test_size=0.3)
val_files, test_files = train_test_split(temp_files, test_size=0.5)

# TODO: delete later after confirming split is what we want is is right
print(f"Total files: {len(all_file_paths)}")
print(f"Train files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Test files: {len(test_files)}")

# grab all files, and create x, y pairs? (x is the input, y is the label) using the labeled_filenames.csv
# TODO: update to use the chosen parameters; currently set to use all the defaults
train_dataset = HDF5IterableDataset(file_paths=train_files, labels_dict=labels_dict, device=device)
val_dataset = HDF5IterableDataset(file_paths=val_files, labels_dict=labels_dict, device=device)
test_dataset = HDF5IterableDataset(file_paths=test_files, labels_dict=labels_dict, device=device)

train_loader = DataLoader(train_dataset, batch_size=4)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

# TODO: test one of the loaders and delete
for i, (data, labels) in enumerate(train_loader):
    print(f"\nBatch {i}:")
    print(f"Data shape: {data.shape}")
    print(f"Labels: {labels}")
    
    if i >= 2:  # Just look at first few batches
        break

# model
# grab the encoder from the saved MAE encoder
# TODO: The encoder part still needs to be tested
encoder_state_dict = torch.load('trained_encoder_state_dict.pt', map_location=device)

# initialize a new encoder with the same architecture
encoder_model = MAEModel(
    input_dim=REQUIRED_COLUMNS,
    embed_dim=embedding_dim,
    num_heads=number_heads,
    depth=layers
).encoder

# load the state dict
encoder_model.load_state_dict(encoder_state_dict)

if torch.cuda.device_count() >= 1:
    encoder_model = nn.DataParallel(encoder_model)

encoder_model.to(device)

# TODO: test the encoder; then delete this code
for data, labels in train_loader:
    data = data.to(device)
    encoded = encoder_model(data)
    print(f"Input shape: {data.shape}")
    print(f"Encoded shape: {encoded.shape}")
    break

model = BinaryClassifier(encoder_model, freeze_encoder=True)
if torch.cuda.device_count() >= 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
else:
    print("Using single GPU or CPU")
model.to(device)
trained_model = train_classifier(model, train_loader, val_loader, num_epochs=10)

if isinstance(trained_model, nn.DataParallel):
    torch.save(trained_model.module.state_dict(), 'binary_classifier.pt')
else:
    torch.save(trained_model.state_dict(), 'binary_classifier.pt')

# Test final model
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
   for data, labels in test_loader:
       data, labels = data.to(device), labels.to(device)
       outputs = model(data)
       predictions = (outputs >= 0.5).float()
       
       test_preds.extend(predictions.cpu().numpy())
       test_labels.extend(labels.cpu().numpy())

test_acc, test_f1, test_recall, test_precision = compute_metrics(test_labels, test_preds)
print("\nTest Results:")
print(f"Accuracy: {test_acc:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"Precision: {test_precision:.4f}")