import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the MAE model
class MAE(nn.Module):
    def __init__(self, img_size=(MAX_ROWS, REQUIRED_COLUMNS), patch_size=16, embed_dim=768, encoder_layers=12, decoder_layers=4, num_heads=12, mask_ratio=0.75):
        super(MAE, self).__init__()
        self.img_size = img_size  # Image dimensions
        self.patch_size = patch_size  # Size of each patch
        self.embed_dim = embed_dim  # Embedding dimension
        self.mask_ratio = mask_ratio  # Ratio of patches to mask
        
        # Calculate the number of patches
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        
        # Reconstruction layer
        self.reconstruction_layer = nn.Linear(embed_dim, patch_size * patch_size)
        
    def forward(self, x):
        # x shape: [batch_size, MAX_ROWS, REQUIRED_COLUMNS]
        
        # Add a channel dimension
        x = x.unsqueeze(1)  # [batch_size, 1, MAX_ROWS, REQUIRED_COLUMNS]
        
        # Create patches and flatten
        patches = self.patch_embed(x)  # [batch_size, embed_dim, H_patches, W_patches]
        patches = patches.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        
        batch_size, num_patches, _ = patches.size()
        
        # Add positional encoding
        patches = patches + self.pos_embed
        
        # Create masks per sample
        num_masked = int(self.mask_ratio * num_patches)
        
        keep_indices = []
        mask_indices = []
        
        for _ in range(batch_size):
            idx = torch.randperm(num_patches)
            idx_keep = idx[num_masked:]
            idx_mask = idx[:num_masked]
            keep_indices.append(idx_keep)
            mask_indices.append(idx_mask)
        
        # Stack indices
        keep_indices = torch.stack(keep_indices)  # [batch_size, num_patches - num_masked]
        mask_indices = torch.stack(mask_indices)  # [batch_size, num_masked]
        
        # Unmasked patches
        patches_unmasked = []
        for i in range(batch_size):
            patches_unmasked.append(patches[i, keep_indices[i], :])
        patches_unmasked = torch.cat(patches_unmasked, dim=0)  # [batch_size * num_unmasked_patches, embed_dim]
        num_unmasked_patches = patches_unmasked.size(0) // batch_size
        
        # Encode
        patches_unmasked = patches_unmasked.view(batch_size, num_unmasked_patches, -1).transpose(0, 1)  # [num_unmasked_patches, batch_size, embed_dim]
        encoded = self.encoder(patches_unmasked)
        encoded = encoded.transpose(0, 1).reshape(-1, self.embed_dim)  # [batch_size * num_unmasked_patches, embed_dim]
        
        # Prepare for decoding
        # Create mask tokens
        mask_tokens = self.mask_token.repeat(batch_size * num_masked, 1)  # [batch_size * num_masked, embed_dim]
        
        # Combine encoded unmasked patches and mask tokens
        patches_decoded = torch.zeros(batch_size * num_patches, self.embed_dim, device=x.device)
        idx_keep = torch.cat([i * num_patches + keep_indices[i] for i in range(batch_size)])
        idx_mask = torch.cat([i * num_patches + mask_indices[i] for i in range(batch_size)])
        patches_decoded[idx_keep] = encoded
        patches_decoded[idx_mask] = mask_tokens
        
        # Reshape back to [batch_size, num_patches, embed_dim]
        patches_decoded = patches_decoded.view(batch_size, num_patches, self.embed_dim)
        
        # Add positional encoding
        patches_decoded = patches_decoded + self.pos_embed
        
        # Decode
        patches_decoded = patches_decoded.transpose(0, 1)  # [num_patches, batch_size, embed_dim]
        decoded = self.decoder(patches_decoded)
        decoded = decoded.transpose(0, 1)  # [batch_size, num_patches, embed_dim]
        
        # Reconstruct
        reconstructed = self.reconstruction_layer(decoded)  # [batch_size, num_patches, patch_size*patch_size]
        
        return reconstructed, mask_indices

import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, and optimizer
model = MAE().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for data in data_loader:
        data = data.to(device)
        batch_size = data.size(0)
        
        # Forward pass
        reconstructed_patches, mask_indices = model(data)
        
        # Get the original patches
        x = data.unsqueeze(1)  # [batch_size, 1, MAX_ROWS, REQUIRED_COLUMNS]
        patches = model.patch_embed(x)  # [batch_size, embed_dim, H_patches, W_patches]
        patches = patches.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        patches = patches.view(batch_size, model.num_patches, -1)  # [batch_size, num_patches, patch_size*patch_size]
        
        # Compute loss on masked patches
        loss = 0
        for i in range(batch_size):
            original_masked_patches = patches[i, mask_indices[i], :]  # [num_masked, patch_size*patch_size]
            reconstructed_masked_patches = reconstructed_patches[i, mask_indices[i], :]  # [num_masked, patch_size*patch_size]
            loss += loss_fn(reconstructed_masked_patches, original_masked_patches)
        
        loss = loss / batch_size
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# https://chatgpt.com/share/674283df-d478-8011-8bf9-b6c17aab56aa


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import ViTMAEConfig, ViTMAEForPreTraining

patch_dim = 64 
pad_height = (patch_dim - (1387 % patch_dim)) % patch_dim
pad_width = (patch_dim - (30000 % patch_dim)) % patch_dim

config = ViTMAEConfig(
    image_size=(1387 + pad_height, 30000 + pad_width),  # Padded data dimensions
    patch_size=(patch_dim, patch_dim),   # Chosen patch size
    num_channels=1,            # Single-channel data
    hidden_size=768,           # Embedding dimension
    num_hidden_layers=12,      # Number of transformer layers
    num_attention_heads=12,    # Number of attention heads
    intermediate_size=3072,    # Size of feed-forward layers
    decoder_num_attention_heads=12,
    decoder_hidden_size=512,
    decoder_num_hidden_layers=8,
    decoder_intermediate_size=2048,
    mask_ratio=0.75,           # Ratio of patches to mask
    norm_pix_loss=False        # Use unnormalized pixel loss
)


model = ViTMAEForPreTraining(config)
model.to(device)

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 5
total_steps = len(data_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        data = torch.nn.functional.pad(data, (0, pad_width, 0, pad_height))
        data = data.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
        data = (data - data.min()) / (data.max() - data.min())

        outputs = model(pixel_values=data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Step [{batch_idx+1}/{len(data_loader)}], "
                f"Loss: {loss.item():.4f}"
            )
