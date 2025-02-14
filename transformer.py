import torch
import torch.nn as nn

from datasets import train_loader, test_loader
from params import * 
from device import get_device

device = get_device()

torch.manual_seed(42)

# Full Encoder-Decoder Architecture:
# Added a complete decoder stack
# Implemented causal masking for autoregressive decoding
# Added target embeddings and positional encodings
# Sequence Generation:
# The model now handles sequential data
# Uses a start token followed by the target class
# Could be extended to generate longer sequences
# Training Process:
# Modified to handle the encoder-decoder structure
# Implements teacher forcing during training
# Uses causal masking to prevent looking at future tokens
# This implementation is more complex but more powerful. It can be extended to:
# Generate sequences of tokens
# Handle variable-length outputs
# Perform more complex tasks than simple classification
# Note that for MNIST classification, this is overkill - the encoder-only version would be more efficient. However, this architecture would be useful for more complex tasks like image captioning or semantic segmentation.

class FullTransformer(nn.Module):
    def __init__(self, image_size=56, patch_size=7, embedding_dim=64, output_dim=10, nhead=8, 
                 num_encoder_layers=2, num_decoder_layers=2, max_seq_length=17):
        super(FullTransformer, self).__init__()
        
        # Encoder parts
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 64 patches for 56x56 image with 7x7 patches
        self.patch_dim = patch_size * patch_size  # 49 for 7x7 patches
        
        # Patch embedding: from patch_dim to embedding_dim
        self.patch_embedding = nn.Linear(self.patch_dim, embedding_dim)
        
        # Position embeddings for patches + CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Target embedding and positional encoding
        self.target_embedding = nn.Embedding(output_dim, embedding_dim)
        self.target_pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, embedding_dim))
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, output_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, x):
        batch_size = x.shape[0]
        # print("Initial shape:", x.shape)
        
        # Remove channel dimension
        x = x.squeeze(1)
        # print("After squeeze:", x.shape)
        
        # Create patches
        patches = x.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        # print("After unfold:", patches.shape)
        
        # Reshape to [batch_size, num_patches, patch_dim]
        patches = patches.permute(0, 1, 2, 3, 4).contiguous()
        # print("After permute:", patches.shape)
        patches = patches.view(batch_size, -1, self.patch_size * self.patch_size)
        # print("After view:", patches.shape)
        
        # Embed patches
        x = self.patch_embedding(patches)
        # print("After embedding:", x.shape)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print("After CLS token:", x.shape)
        
        # Add position embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]
        # print("Final shape before encoder:", x.shape)
        
        return self.encoder(x)

    def decode(self, memory, tgt):
        batch_size = tgt.shape[0]
        seq_len = tgt.shape[1]
        
        # print("Memory shape:", memory.shape)
        # print("Target shape before embedding:", tgt.shape)
        
        # Create target mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        # Embed target sequence and ensure 3D shape
        tgt = self.target_embedding(tgt).squeeze(-2)  # Remove extra dimension
        # print("Target shape after embedding:", tgt.shape)
        
        # Add position embeddings
        tgt = tgt + self.target_pos_embedding[:, :seq_len, :]
        # print("Target shape after position embedding:", tgt.shape)
        
        # Apply decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        # print("Output shape:", output.shape)
        
        return output

    def forward(self, x, tgt):
        # Encode
        memory = self.encode(x)
        # Decode
        output = self.decode(memory, tgt)
        # Project to vocabulary size
        return self.output_projection(output)

# Training setup
model = FullTransformer(
    image_size=56,
    patch_size=patch_size,
    embedding_dim=embedding_dim,
    output_dim=output_dim,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2
).to(device)

# Modified training loop
def train_step(model, images, labels, optimizer, criterion):
    batch_size = labels.shape[0]
    total_loss = 0
    
    for i in range(4):  # Process each position in the 2x2 grid
        # Convert to long type for embedding layer
        start_token = torch.zeros(batch_size, 1, dtype=torch.long).to(labels.device)
        current_label = labels[:, i:i+1].long()  # Convert to long type
        tgt = torch.stack([start_token, current_label], dim=1)
        
        outputs = model(images, tgt)
        loss = criterion(outputs[:, 0, :], current_label.squeeze())
        total_loss += loss
    
    loss = total_loss / 4
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        loss = train_step(model, images, labels, optimizer, criterion)
        
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.shape[0]
        
        for i in range(4):
            start_token = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
            current_label = labels[:, i:i+1].long()
            tgt = torch.stack([start_token, current_label], dim=1)
            
            outputs = model(images, tgt)
            _, predicted = torch.max(outputs[:, 0, :], 1)
            total += current_label.numel()
            correct += (predicted == current_label.squeeze()).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")