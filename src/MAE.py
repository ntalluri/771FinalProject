import torch
import torch.nn as nn


# class MAEModel(nn.Module):
#     def __init__(self, input_dim, embed_dim, num_heads, depth, mask_token_value=0.0):
#         super().__init__()
#         self.input_dim = input_dim
#         self.mask_token_value = mask_token_value
#         self.embed_dim = embed_dim
#         self.depth = depth
#         self.num_heads = num_heads

#         # embedding layer
#         self.embedding = nn.Linear(input_dim, embed_dim)

#         # Transformer encoder for unmasked rows
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=4 * embed_dim,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         # Transformer decoder with placeholders for masked rows
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=4 * embed_dim,
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

#         # reconstruction head
#         self.reconstruction_head = nn.Linear(embed_dim, input_dim)

#     def forward(self, x, row_mask):
#         """
#         Forward pass for the MAE model.
#         Args:
#             x (torch.Tensor): Input data of shape [batch_size, MAX_ROWS, REQUIRED_COLUMNS].
#             row_mask (torch.Tensor): Mask indicating which rows to mask (shape: [batch_size, MAX_ROWS]).
#         Returns:
#             torch.Tensor: Reconstructed data of shape [batch_size, MAX_ROWS, REQUIRED_COLUMNS].
#         """
#         batch_size, num_rows, num_cols = x.size()

#         # Mask rows as per report (25-50% of rows)
#         row_mask = row_mask.unsqueeze(-1).expand(-1, -1, num_cols)
#         x_masked = x.clone()
#         x_masked[row_mask] = self.mask_token_value

#         # Embed and encode
#         embedded = self.embedding(x_masked)
#         encoded = self.encoder(embedded)

#         # Decode with masked tokens
#         decoded = self.decoder(encoded, encoded)

#         # Reconstruct full matrix
#         reconstructed = self.reconstruction_head(decoded)

#         return reconstructed

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        embedded = self.embedding(x)
        src_key_padding_mask = ~mask  # Invert mask: True for positions to pad (masked)
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        encoded = self.norm(encoded)
        return encoded

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        #self.reconstruction_head = nn.Linear(embed_dim, embed_dim)  # Adjust as needed

    def forward(self, encoded, target, tgt_mask=None, tgt_key_padding_mask=None):
        decoded = self.transformer_decoder(target, encoded, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        decoded = self.norm(decoded)
        #reconstructed = self.reconstruction_head(decoded)
        #return reconstructed
        return decoded

class MAEModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth, mask_token_value=0.0):
        super(MAEModel, self).__init__()
        self.input_dim = input_dim
        self.mask_token_value = mask_token_value
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Initialize Encoder and Decoder
        self.encoder = Encoder(input_dim, embed_dim, num_heads, depth)
        self.decoder = Decoder(embed_dim, num_heads, depth)

        # Learned mask token
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))  # Shape: [embed_dim]

        # Reconstruction head to map back to input dimensions
        self.reconstruction_head = nn.Linear(embed_dim, input_dim)  # Final reconstruction to input_dim

    def forward(self, x, row_mask):
        batch_size, num_rows, num_cols = x.size()

        # Clone input and apply mask
        x_masked = x.clone()
        x_masked[row_mask] = self.mask_token_value  # Mask the input data

        # Embed the masked input
        embedded = self.encoder.embedding(x_masked)  # Shape: [batch_size, num_rows, embed_dim]
        
        # Replace masked embeddings with the mask token
        embedded[row_mask] = self.mask_token  # Properly replacing masked embeddings
        
        # Encode the masked input
        encoded = self.encoder(x_masked, row_mask)

        # Decode using the modified embeddings
        decoded = self.decoder(encoded, embedded)

        # Reconstruct the original input dimension
        reconstructed = self.reconstruction_head(decoded)

        return reconstructed
