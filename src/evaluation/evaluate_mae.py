import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from encoder_analysis import EncoderAnalysis
from fine_tuning import FineTuner
from src.dataloader import HDF5IterableDataset


def prepare_data_for_fine_tuning(data_dir, label_path):
    """
    Prepare data with labels for fine-tuning
    """
    dataset = HDF5IterableDataset(data_dir)

    with open(label_path, 'r') as f:
        labels = json.load(f)

    # Create datasets
    train_data, train_labels = [], []
    val_data, val_labels = [], []
    test_data, test_labels = [], []

    # TODO - Split data into train/val/test sets


    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_data), torch.tensor(train_labels)),
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_data), torch.tensor(val_labels)),
        batch_size=32
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_data), torch.tensor(test_labels)),
        batch_size=32
    )

    return train_loader, val_loader, test_loader


def main():
    # Paths and hyperparameters
    model_path = "trained_mae_model.pt"
    data_dir = "../data/Toy_dataset"
    label_path = "path_to_labels.json"
    results_dir = "fine_tuning_results"

    # Initialize analysis and fine-tuning
    encoder_analysis = EncoderAnalysis(model_path)
    fine_tuner = FineTuner(model_path, results_dir)

    # Prepare data
    train_loader, val_loader, test_loader = prepare_data_for_fine_tuning(data_dir, label_path)

    # Analyze encoder representations
    sample_data = next(iter(train_loader))[0]
    encoder_analysis.visualize_encodings(
        sample_data,
        save_path=os.path.join(results_dir, 'encoder_visualization.png')
    )

    # Fine-tune for event detection
    hyperparams = {
        'lr': 1e-4,
        'epochs': 20,
        'batch_size': 32
    }

    detector, history = fine_tuner.create_event_detector(
        train_loader,
        val_loader,
        hyperparams
    )

    # Evaluate on test set
    test_results = fine_tuner.evaluate_detector(detector, test_loader)

    # Save test results
    results_path = os.path.join(results_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)


if __name__ == "__main__":
    main()