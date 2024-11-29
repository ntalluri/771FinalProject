import torch
import torch.nn as nn


class MAEModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth, mask_token_value=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.mask_token_value = mask_token_value

        # embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Transformer encoder for unmasked rows
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Transformer decoder with placeholders for masked rows
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

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

        # Mask rows as per report (25-50% of rows)
        row_mask = row_mask.unsqueeze(-1).expand(-1, -1, num_cols)
        x_masked = x.clone()
        x_masked[row_mask] = self.mask_token_value

        # Embed and encode
        embedded = self.embedding(x_masked)
        encoded = self.encoder(embedded)

        # Decode with masked tokens
        decoded = self.decoder(encoded, encoded)

        # Reconstruct full matrix
        reconstructed = self.reconstruction_head(decoded)

        return reconstructed