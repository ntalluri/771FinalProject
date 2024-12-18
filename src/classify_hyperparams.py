import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import HDF5IterableDataset  # Ensure correct import path
from MAE import Encoder  # Ensure correct import path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
import pandas as pd
import random
import datetime
import numpy as np
import itertools
import copy

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
    def __init__(self, patience=3, verbose=False, delta=0.0, path='best_val_loss_model.pt', mode='min'):
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
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        
        self.optimal_threshold = 0.5

    def __call__(self, metric, model, optimal_threshold=0.5):
        if self.mode == 'min':
            score = metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(metric, model, optimal_threshold)
            elif score > self.best_score - self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(metric, model, optimal_threshold)
                self.counter = 0
        else:
            # Existing 'max' mode for metrics like F1 score
            score = metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(metric, model, optimal_threshold)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(metric, model, optimal_threshold)
                self.counter = 0

    def save_checkpoint(self, metric, model, optimal_threshold):
        '''Saves the model when the monitored metric improves.'''
        if self.verbose:
            print(f'Validation metric improved to {metric:.6f}. Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_metric = metric
        self.optimal_threshold = optimal_threshold

# ---------------------------
# 3. Binary Classifier Definition
# ---------------------------
class BinaryClassifier(nn.Module):
    def __init__(self, encoder, input_dim, embeddings, freeze_encoder=True):
        super(BinaryClassifier, self).__init__()
        self.encoder = encoder

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encoder

        layers = []
        for hidden_unit in embeddings:
            layers.append(nn.Linear(input_dim, hidden_unit))
            layers.append(nn.ReLU())
            input_dim = hidden_unit
        layers.append(nn.Linear(input_dim, 1))  # Final output layer
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Pass input through the encoder
        encoded = self.encoder(x)  # Shape: (batch_size, num_rows, embed_dim)

        # Apply mean pooling over the sequence dimension to get (batch_size, embed_dim)
        pooled = encoded.mean(dim=1)

        # Pass the pooled vector through the classifier
        out = self.classifier(pooled).squeeze(1)
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

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[optimal_idx] if optimal_idx < len(f1_scores) else f1_scores[-1]
    return optimal_threshold, best_f1

# ---------------------------
# 5. Training Function
# ---------------------------
def train_classifier(model, train_loader, val_loader, num_epochs=10, patience=3, pos_weight_tensor=None, weight_decay=0.0, lr=1e-4, writer=None):
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
        print("Not using pos_weight")
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    # Initialize EarlyStopping to monitor validation loss
    early_stopping = EarlyStopping(patience=patience, verbose=True, 
                                   delta=0.0, path='best_val_loss_model.pt', mode='min')
    
    # Variables to track the best F1 score
    best_f1 = -float('inf')
    best_f1_threshold = 0.5
    best_f1_model_path = 'best_f1_model.pt'

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

                if batch_idx % 10 == 0:
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
        val_probs = []
        val_labels = []
        val_batches = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                try:
                    data, labels = data.to(device), labels.to(device).float()
                    outputs = model(data)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    prob = torch.sigmoid(outputs)
                    predictions = (prob >= 0.5).float()

                    val_preds.extend(predictions.cpu().numpy())
                    val_probs.extend(prob.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_batches += 1
                except Exception as e:
                    print(f"Error during validation batch {batch_idx}: {e}")
                    continue

        optimal_threshold, current_best_f1 = find_optimal_threshold(val_labels, val_probs)
        print(f"Optimal Threshold: {optimal_threshold}, Best F1 Score: {current_best_f1}")

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
            
            writer.add_scalar('F1/Best', current_best_f1, epoch + 1)
            writer.add_scalar('Optimal_Threshold', optimal_threshold, epoch + 1)

        # Early Stopping Check based on validation loss
        early_stopping(avg_val_loss, model, optimal_threshold)

        # Check if current F1 is the best
        if current_best_f1 > best_f1:
            best_f1 = current_best_f1
            best_f1_threshold = optimal_threshold
            torch.save(model.state_dict(), best_f1_model_path)
            print(f"New best F1 score: {best_f1:.4f}. Model saved to {best_f1_model_path}")

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Load the best validation loss model
    print("Loading the best model based on validation loss...")
    model.load_state_dict(torch.load('best_val_loss_model.pt'))
    best_val_loss_threshold = early_stopping.optimal_threshold
    print(f"Best validation loss model threshold: {best_val_loss_threshold}")

    return model, best_val_loss_threshold, best_f1_threshold, best_f1_model_path

# ---------------------------
# 6. Evaluation Function
# ---------------------------
def evaluate_model(model, test_loader, threshold=0.5):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        threshold (float): Threshold for converting probabilities to binary predictions.

    Returns:
        tuple: (accuracy, f1_score, recall, precision)
    """
    print(f"Testing with a threshold of {threshold}")
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            try:
                data, labels = data.to(device), labels.to(device).float()
                outputs = model(data)
                prob = torch.sigmoid(outputs)
                predictions = (prob >= threshold).float()

                test_preds.extend(predictions.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(prob.cpu().numpy())
            except Exception as e:
                print(f"Error during test batch {batch_idx}: {e}")
                continue
            
    optimal_threshold, best_f1 = find_optimal_threshold(test_labels, test_probs)
    print(f"Optimal Threshold: {optimal_threshold}, Best F1 Score: {best_f1}")        

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
    input_dim = 512    # Ensure this matches your trained encoder
    number_heads = 8       # Ensure this matches your trained encoder
    layers = 4             # Ensure this matches your trained encoder
    
    file_prop = 0.01  # Proportion of negative samples to keep

    param_grid = {
      'learning_rate': [1e-4],
      'freeze_encoder': [True, False],
      'weight_decay': [1e-4],
      'use_pos_weight': [True, False],
      'embeddings': [
          [128, 64],
          [256],
          [256, 128],
          [256, 128, 64],
        ]
      }
      
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    print(f"Total hyperparameter combinations to try: {len(param_combinations)}")

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
        'row_shift': {  # Changed from 'row_swap' to 'row_shift'
                'apply_prob': 0.5,
                'shift_max_min': 1,
                'shift_max_max': 693
            },
        'column_shift': {
            'apply_prob': 0.5,        # 50% chance to apply column shift
            'shift_max_min': 1,       # Minimum shift value
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

    # 11. Compute Class Weights for the Training Set
    train_labels = [labels_dict.get(os.path.basename(f), 0) for f in train_files]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos

    # Avoid division by zero
    if num_pos == 0:
        raise ValueError("No positive samples in the training set.")

    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    # 11. Initialize a list to store all results
    results = []
    best_test_f1 = -float('inf')
    best_model_state = None
    best_params = None

    # 12. Generate all hyperparameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    print(f"Starting hyperparameter tuning over {len(param_combinations)} combinations.")

    # 13. Loop over all parameter combinations
    for idx, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        print(f"\n=== Hyperparameter Combination {idx + 1}/{len(param_combinations)} ===")
        print(f"Parameters: {params}")

        learning_rate = params['learning_rate']
        freeze_encoder = params['freeze_encoder']
        weight_decay = params['weight_decay']
        use_pos_weight = params['use_pos_weight']
        embeddings = params['embeddings']

        # 10. Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
        print("Datasets and DataLoaders are set up successfully.")

        # 12. Initialize the Encoder
        encoder_model = Encoder(
            input_dim=REQUIRED_COLUMNS,
            embed_dim=input_dim,
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
            input_dim = input_dim,
            embeddings=embeddings,
            freeze_encoder=freeze_encoder  # Set to False if you want to fine-tune the encoder
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

        if not use_pos_weight:
            pos_weight_tensor = None
        else:
            pos_weight_tensor = pos_weight

        # 14. Train the Classifier
        trained_classifier, best_val_loss_threshold, best_f1_threshold, best_f1_model_path = train_classifier(
            model=classifier_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            patience=5,  # Adjust patience as needed
            pos_weight_tensor=pos_weight_tensor,  # Pass the computed pos_weight
            weight_decay = weight_decay,
            lr = learning_rate,
            writer=writer  # Pass the TensorBoard writer
        )

        # Evaluate the model with the best validation loss
        print("\nEvaluating model with best validation loss:")
        val_loss_model = copy.deepcopy(trained_classifier)
        val_loss_model.load_state_dict(torch.load('best_val_loss_model.pt'))
        val_loss_model.to(device)
        test_acc_val_loss, test_f1_val_loss, test_recall_val_loss, test_precision_val_loss = evaluate_model(
            val_loss_model, test_loader, threshold=best_val_loss_threshold
        )

        # Evaluate the model with the best F1 score
        print("\nEvaluating model with best F1 score:")
        best_f1_model = copy.deepcopy(trained_classifier)
        best_f1_model.load_state_dict(torch.load(best_f1_model_path))
        best_f1_model.to(device)
        test_acc_f1, test_f1_f1, test_recall_f1, test_precision_f1 = evaluate_model(
            best_f1_model, test_loader, threshold=best_f1_threshold
        )

        # 17. Close the TensorBoard writer
        writer.close()
        print("Training and evaluation complete.")

        # Append the results
        results.append({
            **params,
            'test_acc_val_loss': test_acc_val_loss,
            'test_f1_val_loss': test_f1_val_loss,
            'test_recall_val_loss': test_recall_val_loss,
            'test_precision_val_loss': test_precision_val_loss,
            'test_acc_f1': test_acc_f1,
            'test_f1_f1': test_f1_f1,
            'test_recall_f1': test_recall_f1,
            'test_precision_f1': test_precision_f1
        })

        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_df.to_csv("classifier_parameter_tuning_results.csv", mode='a', header=not os.path.exists("classifier_parameter_tuning_results.csv"), index=False)
        print("Results saved to classifier_parameter_tuning_results.csv")

        # Update best model if current run has higher validation F1
        if test_f1_f1 > best_test_f1:
            best_test_f1 = test_f1_f1
            best_model_state = copy.deepcopy(trained_classifier.state_dict())
            best_params = params

   # After all combinations
    if best_model_state is not None:
        # Save the best model
        torch.save(best_model_state, best_classifier_path)
        print(f"\nBest model saved to {best_classifier_path} with parameters:")
        print(best_params)
    else:
        print("No model was trained.")

    # Save all of the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('all_classifier_parameter_tuning_results.csv', index=False)
    print("All results saved to all_classifier_parameter_tuning_results.csv")
