import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class EncoderAnalysis:
    """
    Framework for analyzing and visualizing encoder representations
    """

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torch.load(model_path)
        self.model.to(device)
        self.model.eval()

    def extract_features(self, data):
        """
        Extract features from the encoder for a batch of data
        """
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)

            # Get embeddings
            embedded = self.model.embedding(data)

            # Get encoder representations
            encoded = self.model.encoder(embedded)

            # Average across temporal dimension to get a single vector per sample
            features = torch.mean(encoded, dim=1)

        return features.cpu().numpy()

    def visualize_encodings(self, data, labels=None, save_path=None):
        """
        Create visualization of encoder representations using PCA
        """
        # Extract features
        features = self.extract_features(data)

        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # Plot
        plt.figure(figsize=(10, 8))
        if labels is not None:
            plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
            plt.colorbar(label='Event/No Event')
        else:
            plt.scatter(features_2d[:, 0], features_2d[:, 1])

        plt.title('PCA of Encoder Representations')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')

        if save_path:
            plt.savefig(save_path)
        plt.close()