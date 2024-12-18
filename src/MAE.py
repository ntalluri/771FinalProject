import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer_encoder(embedded)
        return encoded

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, input_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.reconstruction_head = nn.Linear(embed_dim, input_dim)

    def forward(self, encoded, memory):
        decoded = self.transformer_decoder(encoded, memory)
        reconstructed = self.reconstruction_head(decoded)
        return reconstructed

class MAEModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth, mask_token_value=0.0):
        super(MAEModel, self).__init__()
        self.input_dim = input_dim
        self.mask_token_value = mask_token_value
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth

        self.encoder = Encoder(input_dim, embed_dim, num_heads, depth)
        self.decoder = Decoder(embed_dim, num_heads, depth, input_dim)

    def forward(self, x, row_mask):
        batch_size, num_rows, num_cols = x.size()

        # Apply row mask
        row_mask_expanded = row_mask.unsqueeze(-1).expand(-1, -1, num_cols)
        x_masked = x.clone()
        x_masked[row_mask_expanded] = self.mask_token_value

        # Encode the masked input
        encoded = self.encoder(x_masked)

        # Decode using the encoded representations
        decoded = self.decoder(encoded, encoded)

        return decoded