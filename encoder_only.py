import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Architecture
#    Image (28x28)
#       ↓
#    Split into patches (16 patches of 7x7)
#       ↓
#    Embed patches (patch_dim → embedding_dim)
#       ↓
#    Add CLS token
#       ↓
#    Add position embeddings
#       ↓
#    Transformer layers
#       ↓
#    Take CLS token output
#       ↓
#    Classification head



from datasets import train_loader, test_loader
from params import * 
from device import get_device

# Set the seed for reproducibility
torch.manual_seed(0)
device = get_device()

# Define the neural network model
class ImageClassifier(nn.Module):
    def __init__(self, image_size=28, patch_size=7, embedding_dim=64, output_dim=10, nhead=8, num_layers=2):
        super(ImageClassifier, self).__init__()
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        
        # Patch embedding layer
        self.patch_embedding = nn.Linear(self.patch_dim, embedding_dim)
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split image into patches: (batch_size, num_patches, patch_dim)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_dim)
        # print(x.shape)
        # Embed patches
        x = self.patch_embedding(x)
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add position embeddings
        x = x + self.pos_embedding
        # Apply transformer
        x = self.transformer(x)
        # Use CLS token for classification
        x = x[:, 0]
        # Output layer
        output = self.output(x)
        return output

# Initialize the model, loss function, and optimizer
model = ImageClassifier(
    image_size=28,
    patch_size=patch_size,
    embedding_dim=embedding_dim,
    output_dim=output_dim,
    nhead=8,
    num_layers=2
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move the data to the device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the loss at each 100 mini-batches
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")