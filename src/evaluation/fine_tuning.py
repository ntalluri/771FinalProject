import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import json
from datetime import datetime


class EventDetector(nn.Module):
    """
    Neural network head for event detection that sits on top of the encoder
    """

    def __init__(self, encoder_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class FineTuner:
    """
    Framework for fine-tuning the encoder for event detection
    """

    def __init__(self, base_model_path, results_dir='fine_tuning_results'):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                self.multi_gpu = True
                # Initialize process group if not already initialized
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl')
                self.device = torch.device(f'cuda:{dist.get_rank()}')
            else:
                print("Using single GPU!")
                self.multi_gpu = False
                self.device = torch.device('cuda')
        else:
            print("Using CPU!")
            self.multi_gpu = False
            self.device = torch.device('cpu')

        self.base_model = torch.load(base_model_path)
        self.base_model.to(self.device)
        if self.multi_gpu:
            self.base_model = DDP(self.base_model, device_ids=[dist.get_rank()])

        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def create_event_detector(self, train_loader, val_loader, hyperparams):
        """
        Fine-tune the encoder for event detection
        """
        # Create event detection head
        detector = EventDetector(encoder_dim=512).to(self.device)
        if self.multi_gpu:
            detector = DDP(detector, device_ids=[dist.get_rank()])

        # Create distributed sampler for training data if using multiple GPUs
        if self.multi_gpu:
            train_sampler = DistributedSampler(train_loader.dataset)
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=train_sampler,
                num_workers=train_loader.num_workers,
                pin_memory=True
            )

        # Freeze encoder weights initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Training setup
        optimizer = torch.optim.Adam([
            {'params': detector.parameters(), 'lr': hyperparams['lr']},
            {'params': self.base_model.parameters(), 'lr': hyperparams['lr'] * 0.1}
        ])
        criterion = nn.BCELoss()

        # Training loop
        best_val_f1 = 0
        best_detector = None
        training_history = []

        for epoch in range(hyperparams['epochs']):
            # Set epoch for distributed sampler
            if self.multi_gpu:
                train_sampler.set_epoch(epoch)

            # Training phase
            detector.train()
            train_loss = 0

            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                # Get encoder features
                with torch.no_grad():
                    embedded = self.base_model.module.embedding(batch_data) if self.multi_gpu else self.base_model.embedding(batch_data)
                    encoded = self.base_model.module.encoder(embedded) if self.multi_gpu else self.base_model.encoder(embedded)
                    features = torch.mean(encoded, dim=1)

                # Forward pass through detector
                predictions = detector(features)
                loss = criterion(predictions.squeeze(), batch_labels.float())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Average loss across GPUs if using distributed training
            if self.multi_gpu:
                train_loss_tensor = torch.tensor(train_loss).to(self.device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                train_loss = train_loss_tensor.item() / dist.get_world_size()

            # Validation phase (only on main process if using multiple GPUs)
            if not self.multi_gpu or (self.multi_gpu and dist.get_rank() == 0):
                detector.eval()
                val_predictions = []
                val_labels = []

                with torch.no_grad():
                    for batch_data, batch_labels in val_loader:
                        batch_data = batch_data.to(self.device)

                        embedded = self.base_model.module.embedding(batch_data) if self.multi_gpu else self.base_model.embedding(batch_data)
                        encoded = self.base_model.module.encoder(embedded) if self.multi_gpu else self.base_model.encoder(embedded)
                        features = torch.mean(encoded, dim=1)

                        predictions = detector(features)

                        val_predictions.extend(predictions.cpu().numpy())
                        val_labels.extend(batch_labels.numpy())

                # Calculate metrics
                val_predictions = np.array(val_predictions) > 0.5
                precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='binary')
                accuracy = accuracy_score(val_labels, val_predictions)

                # Save training history
                epoch_results = {
                    'epoch': epoch,
                    'train_loss': train_loss / len(train_loader),
                    'val_precision': float(precision),
                    'val_recall': float(recall),
                    'val_f1': float(f1),
                    'val_accuracy': float(accuracy)
                }
                training_history.append(epoch_results)

                # Save best model
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_detector = detector.module.state_dict() if self.multi_gpu else detector.state_dict()

                print(f"Epoch {epoch}: Train Loss = {train_loss / len(train_loader):.4f}, Val F1 = {f1:.4f}")

        # Save training history (only on main process if using multiple GPUs)
        if not self.multi_gpu or (self.multi_gpu and dist.get_rank() == 0):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = os.path.join(self.results_dir, f'training_history_{timestamp}.json')
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=4)

        # Load best model
        if best_detector is not None:
            if self.multi_gpu:
                detector.module.load_state_dict(best_detector)
            else:
                detector.load_state_dict(best_detector)

        return detector, training_history

    def evaluate_detector(self, detector, test_loader):
        """
        Evaluate the fine-tuned detector on test data
        """
        detector.eval()
        test_predictions = []
        test_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)

                embedded = self.base_model.module.embedding(batch_data) if self.multi_gpu else self.base_model.embedding(batch_data)
                encoded = self.base_model.module.encoder(embedded) if self.multi_gpu else self.base_model.encoder(embedded)
                features = torch.mean(encoded, dim=1)

                predictions = detector(features)
                test_predictions.extend(predictions.cpu().numpy())
                test_labels.extend(batch_labels.numpy())

        # Convert predictions to binary
        test_predictions = np.array(test_predictions) > 0.5

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_predictions, average='binary')
        accuracy = accuracy_score(test_labels, test_predictions)

        results = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy)
        }

        return results