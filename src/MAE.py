import torch
import torch.nn as nn
from dataloader import HDF5IterableDataset, generate_row_mask
from torch.utils.data import DataLoader

class MAEModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth, mask_token_value=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.mask_token_value = mask_token_value

        # embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads), num_layers=depth
        )
        
        # transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads), num_layers=depth
        )

        # reconstruction head
        self.reconstruction_head = nn.Linear(embed_dim, input_dim)

    def forward(self, x, row_mask):
        """
        Forward pass for the MAE model.
        Args:
            x (torch.Tensor): Input data of shape [batch_size, MAX_ROWS, REQUIRED_COLUMNS].
            row_mask (torch.Tensor): Mask indicating which rows to mask (shape: [batch_size, MAX_ROWS]).
        Returns:
            torch.Tensor: Reconstructed data of shape [batch_size, MAX_ROWS, REQUIRED_COLUMNS].
        """
        batch_size, num_rows, num_cols = x.size()
      
        row_mask = row_mask.unsqueeze(-1).expand(batch_size, num_rows, num_cols)  # shape: [batch_size, MAX_ROWS, REQUIRED_COLUMNS]
        x[row_mask] = self.mask_token_value

        x = self.embedding(x)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        reconstructed = self.reconstruction_head(decoded)

        return reconstructed

