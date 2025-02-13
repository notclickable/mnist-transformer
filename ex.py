import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Set the seed for reproducibility
torch.manual_seed(0)

# Define the device (GPU or CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device("mps") if torch.backends.mps.is_available() else device

# Hyperparameters
input_dim = 784  # 28x28 images
embedding_dim = 64
output_dim = 10
batch_size = 100
num_epochs = 10
learning_rate = 0.001

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define the neural network model
class ImageClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, nhead=8):
        super(ImageClassifier, self).__init__()
        # Embedding layer (using nn.Linear)
        self.embedding = nn.Linear(input_dim, embedding_dim)
        
        # Add Transformer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        # Output layer
        self.output = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, input_dim)
        # Embed the input
        x = self.embedding(x)
        # Reshape for transformer: (batch_size, sequence_length=1, embedding_dim)
        x = x.unsqueeze(1)
        # Pass through transformer
        x = self.transformer(x)
        # Squeeze back: (batch_size, embedding_dim)
        x = x.squeeze(1)
        # Output layer
        output = self.output(x)
        return output

# Initialize the model, loss function, and optimizer
model = ImageClassifier(input_dim, embedding_dim, output_dim).to(device)
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