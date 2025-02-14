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
    def __init__(self, image_size=28, patch_size=7, embedding_dim=64, output_dim=10, nhead=8, 
                 num_encoder_layers=2, num_decoder_layers=2, max_seq_length=17):  # 16 patches + 1 cls token
        super(FullTransformer, self).__init__()
        # Encoder parts (same as before)
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        self.patch_embedding = nn.Linear(self.patch_dim, embedding_dim)
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
        
        # Save dimensions for later
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, x):
        batch_size = x.shape[0]
        # Split image into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_dim)
        # Embed patches
        x = self.patch_embedding(x)
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add position embeddings
        x = x + self.pos_embedding
        # Apply encoder
        return self.encoder(x)

    def decode(self, memory, tgt):
        batch_size = tgt.shape[0]
        # Create target mask
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        # Embed target sequence
        tgt = self.target_embedding(tgt)
        # Add position embeddings
        tgt = tgt + self.target_pos_embedding[:, :tgt.shape[1], :]
        # Apply decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
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
    image_size=28,
    patch_size=patch_size,
    embedding_dim=embedding_dim,
    output_dim=output_dim,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2
).to(device)

# Modified training loop
def train_step(model, images, labels, optimizer, criterion):
    # Create target sequence (start token + label)
    start_token = torch.zeros_like(labels)  # 0 as start token
    tgt = torch.stack([start_token, labels], dim=1)
    
    # Forward pass
    outputs = model(images, tgt)
    # We only care about the prediction after the start token
    loss = criterion(outputs[:, 0, :], labels)
    
    # Backward and optimize
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
        start_token = torch.zeros_like(labels)
        tgt = torch.stack([start_token, labels], dim=1)
        outputs = model(images, tgt)
        _, predicted = torch.max(outputs[:, 0, :], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")