import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import HDF5IterableDataset  # Ensure correct import path
from MAE import Encoder  # Ensure correct import path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import pandas as pd
import random
import datetime
import numpy as np

# ---------------------------
# 1. Seed Setting for Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------
# 2. Early Stopping Implementation
# ---------------------------
class EarlyStopping:
    """
    Early stops the training if the monitored metric does not improve after a given patience.
    """
    def __init__(self, patience=3, verbose=False, delta=0.0, path='best_classifier.pt', mode='max'):
        """
        Args:
            patience (int): How long to wait after last time monitored metric improved.
            verbose (bool): If True, prints a message for each improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model.
            mode (str): 'min' for metrics like loss, 'max' for metrics like F1 score
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = -float('inf') if mode == 'max' else float('inf')

    def __call__(self, metric, model):
        if self.mode == 'max':
            score = metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(metric, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(metric, model)
                self.counter = 0
        else:
            # Existing 'min' mode for loss
            score = metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(metric, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(metric, model)
                self.counter = 0

    def save_checkpoint(self, metric, model):
        '''Saves the model when the monitored metric improves.'''
        if self.verbose:
            print(f'Validation metric improved to {metric:.6f}. Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_metric = metric

# ---------------------------
# 3. Binary Classifier Definition
# ---------------------------
class BinaryClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, freeze_encoder=True):
        super(BinaryClassifier, self).__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # Pass input through the encoder
        encoded = self.encoder(x)  # Shape: (batch_size, num_rows, embed_dim)

        # Apply mean pooling over the sequence dimension to get (batch_size, embed_dim)
        pooled = encoded.mean(dim=1)

        # Pass the pooled vector through the classifier
        out = self.classifier(pooled).squeeze(1)
        # out = out.view(-1)  # Flatten to (batch_size,)
        return out

# ---------------------------
# 4. Utility Functions
# ---------------------------
def load_labels(csv_path):
    """
    Load labels from CSV file
    """
    labels_df = pd.read_csv(csv_path)
    return dict(zip(labels_df['Filename'], labels_df['Label']))

def get_all_file_paths(folder_paths):
    all_file_paths = []
    for subdir_path in folder_paths:
        # List files ending with '.h5' in the current subdirectory
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

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return acc, f1, recall, precision

# ---------------------------
# 5. Training Function
# ---------------------------
def train_classifier(model, train_loader, val_loader, num_epochs=10, patience=3, pos_weight_tensor=None, writer=None):
    """
    Train the binary classifier.

    Args:
        model (nn.Module): The binary classification model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        patience (int): Patience for early stopping.
        pos_weight_tensor (torch.Tensor, optional): Weight tensor for positive class in loss.
        writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging.
    
    Returns:
        nn.Module: The trained model.
    """
    if pos_weight_tensor is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # Initialize EarlyStopping to monitor F1 score
    early_stopping = EarlyStopping(patience=patience, verbose=True, 
                                   delta=0.0, path='best_classifier.pt', mode='min')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_batches = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            try:
                data, labels = data.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = (torch.sigmoid(outputs) >= 0.5).float()

                train_preds.extend(predictions.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                train_batches += 1

                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"Error during training batch {batch_idx}: {e}")
                continue

        avg_train_loss = train_loss / max(train_batches, 1)
        train_acc, train_f1, train_recall, train_precision = compute_metrics(train_labels, train_preds)

        print(f"  Training Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"F1: {train_f1:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_batches = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                try:
                    data, labels = data.to(device), labels.to(device).float()
                    outputs = model(data)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predictions = (torch.sigmoid(outputs) >= 0.5).float()

                    val_preds.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_batches += 1
                except Exception as e:
                    print(f"Error during validation batch {batch_idx}: {e}")
                    continue

        avg_val_loss = val_loss / max(val_batches, 1)
        val_acc, val_f1, val_recall, val_precision = compute_metrics(val_labels, val_preds)

        print(f"  Validation Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}")

        # Logging to TensorBoard
        if writer:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
            writer.add_scalar('Accuracy/Train', train_acc, epoch + 1)
            writer.add_scalar('F1/Train', train_f1, epoch + 1)
            writer.add_scalar('Recall/Train', train_recall, epoch + 1)
            writer.add_scalar('Precision/Train', train_precision, epoch + 1)

            writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch + 1)
            writer.add_scalar('F1/Validation', val_f1, epoch + 1)
            writer.add_scalar('Recall/Validation', val_recall, epoch + 1)
            writer.add_scalar('Precision/Validation', val_precision, epoch + 1)

        # Early Stopping Check based on F1 score
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Load the best model
    print("Loading the best model from early stopping...")
    model.load_state_dict(torch.load('best_classifier.pt'))
    return model

# ---------------------------
# 6. Evaluation Function
# ---------------------------
def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        tuple: (accuracy, f1_score, recall, precision)
    """
    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            try:
                data, labels = data.to(device), labels.to(device).float()
                outputs = model(data)
                predictions = (torch.sigmoid(outputs) >= 0.5).float()

                test_preds.extend(predictions.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Error during test batch {batch_idx}: {e}")
                continue

    test_acc, test_f1, test_recall, test_precision = compute_metrics(test_labels, test_preds)
    print("\nTest Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Precision: {test_precision:.4f}")

    return test_acc, test_f1, test_recall, test_precision

# ---------------------------
# 7. Main Execution
# ---------------------------
if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Constants for data dimensions
    MIN_ROWS = 1357
    MAX_ROWS = 1387
    REQUIRED_COLUMNS = 30000

    # Hyperparameters
    batch_size = 6
    num_epochs = 10
    learning_rate = 1e-4
    embedding_dim = 512    # Ensure this matches your trained encoder
    number_heads = 8       # Ensure this matches your trained encoder
    layers = 4             # Ensure this matches your trained encoder
    
    file_prop = 0.25  # Proportion of negative samples to keep

    # Paths
    labels_csv_path = "labeled_filenames.csv"
    folders_txt_path = 'labeled_folders.txt'
    encoder_state_path = 'trained_encoder_state_dict.pt'
    best_classifier_path = 'best_classifier.pt'

    # 1. Load Labels
    labels_dict = load_labels(labels_csv_path)

    # 2. Load Folder Paths
    with open(folders_txt_path, 'r') as file:
        folder_paths = file.read().splitlines()

    # 3. Get All H5 File Paths
    all_file_paths = get_all_file_paths(folder_paths)
    print(f"Total files found: {len(all_file_paths)}")
    
    random.shuffle(all_file_paths)

    # 4. Separate Positive and Negative Files
    positive_files = [f for f in all_file_paths if labels_dict.get(os.path.basename(f), 0) == 1]
    negative_files = [f for f in all_file_paths if labels_dict.get(os.path.basename(f), 0) == 0]

    print(f"Total positive files: {len(positive_files)}")
    print(f"Total negative files: {len(negative_files)}")

    # Validate the number of positive samples
    assert len(positive_files) == 9, f"Expected 9 positive samples, found {len(positive_files)}"

    # Shuffle the positive and negative files
    random.shuffle(positive_files)
    random.shuffle(negative_files)
    
    # Limit negative samples based on file_prop
    len_paths = len(negative_files)
    num_keep = int(len_paths * file_prop)
    negative_files = negative_files[:num_keep]

    # 5. Split Positive Files into Train/Val/Test
    train_pos = positive_files[:4]
    val_pos = positive_files[4:6]
    test_pos = positive_files[6:]

    print(f"Training positive samples: {len(train_pos)}")
    print(f"Validation positive samples: {len(val_pos)}")
    print(f"Testing positive samples: {len(test_pos)}")

    # 6. Split Negative Files into Train/Val/Test based on proportions
    train_neg, temp_neg = train_test_split(negative_files, test_size=5/9, random_state=42)
    val_neg, test_neg = train_test_split(temp_neg, test_size=3/5, random_state=42)

    print(f"Training negative samples: {len(train_neg)}")
    print(f"Validation negative samples: {len(val_neg)}")
    print(f"Testing negative samples: {len(test_neg)}")

    # 7. Combine Positive and Negative Files for Each Split
    train_files = train_pos + train_neg
    val_files = val_pos + val_neg
    test_files = test_pos + test_neg

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    print(f"Total training files: {len(train_files)}")
    print(f"Total validation files: {len(val_files)}")
    print(f"Total testing files: {len(test_files)}")
    
    # 8. Define augmentation configuration with randomness
    augmentations = {
        'noise': {
            'apply_prob': 0.5,        # 50% chance to apply noise
            'level_min': 0.001,       # Minimum noise level
            'level_max': 0.02         # Maximum noise level
        },
        'row_swap': {
            'apply_prob': 0.5,        # 50% chance to apply row swap
            'swap_prob_min': 0.3,     # Minimum swap probability
            'swap_prob_max': 0.7      # Maximum swap probability
        },
        'column_shift': {
            'apply_prob': 0.5,        # 50% chance to apply column shift
            'shift_max_min': 1000,       # Minimum shift value
            'shift_max_max': 15000       # Maximum shift value
        }
    }

    # 9. Create Datasets with appropriate modes
    train_dataset = HDF5IterableDataset(
        file_paths=train_files,
        labels_dict=labels_dict,
        device=device,
        mode='train',
        augment_positive=True,
        augmentations=augmentations
    )

    val_dataset = HDF5IterableDataset(
        file_paths=val_files,
        labels_dict=labels_dict,
        device=device,
        mode='val',
        augment_positive=False,  # No augmentations in validation
        augmentations=augmentations  # Augmentations won't be applied as mode='val'
    )

    test_dataset = HDF5IterableDataset(
        file_paths=test_files,
        labels_dict=labels_dict,
        device=device,
        mode='test',
        augment_positive=False,  # No augmentations in test
        augmentations=augmentations  # Augmentations won't be applied as mode='test'
    )

    # 10. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Datasets and DataLoaders are set up successfully.")

    # 11. Compute Class Weights for the Training Set
    train_labels = [labels_dict.get(os.path.basename(f), 0) for f in train_files]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos

    # Avoid division by zero
    if num_pos == 0:
        raise ValueError("No positive samples in the training set.")

    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    # 12. Initialize the Encoder
    encoder_model = Encoder(
        input_dim=REQUIRED_COLUMNS,
        embed_dim=embedding_dim,
        num_heads=number_heads,
        depth=layers
    )
    
    # Load the trained encoder's state dictionary
    encoder_state_dict = torch.load(encoder_state_path, map_location=device)
    encoder_model.load_state_dict(encoder_state_dict)
    print("Encoder loaded successfully.")
    
    encoder_model.to(device)
    encoder_model.eval()  # Set encoder to evaluation mode
    
    # Initialize the Binary Classifier
    classifier_model = BinaryClassifier(
        encoder=encoder_model,
        embed_dim=embedding_dim,
        freeze_encoder=False  # Set to False if you want to fine-tune the encoder
    )
    
    # If using multiple GPUs, wrap only the classifier_model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for the classifier!")
        classifier_model = nn.DataParallel(classifier_model)
    else:
        print("Using single GPU or CPU for the classifier.")
    
    classifier_model.to(device)

    # 13. Initialize TensorBoard SummaryWriter with Unique Log Directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    logs_dir = os.path.join(project_dir, 'logs', 'binary_classifier')

    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_log_dir = os.path.join(logs_dir, f"run_{timestamp}")
    writer = SummaryWriter(log_dir=unique_log_dir)
    print(f"TensorBoard logs will be saved to: {unique_log_dir}")

    # 14. Train the Classifier
    trained_classifier = train_classifier(
        model=classifier_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        patience=3,  # Adjust patience as needed
        pos_weight_tensor=pos_weight,  # Pass the computed pos_weight
        writer=writer  # Pass the TensorBoard writer
    )

    # 15. Save the Trained Classifier
    if isinstance(trained_classifier, nn.DataParallel):
        torch.save(trained_classifier.module.state_dict(), best_classifier_path)
    else:
        torch.save(trained_classifier.state_dict(), best_classifier_path)
    print(f"Trained classifier saved to {best_classifier_path}")

    # 16. Evaluate on Test Set
    evaluate_model(trained_classifier, test_loader)

    # 17. Close the TensorBoard writer
    writer.close()
    print("Training and evaluation complete.")
