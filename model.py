import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Embedding layers
        self.embedding1 = nn.Linear(784, 196)  # 784 input features (28x28), 196 output features
        self.embedding2 = nn.Linear(196, 64)   # 196 input features, 64 output features
        # Attention mechanism
        self.attention = nn.Linear(64, 1)      # Compute attention scores
        # Decoder
        self.decoder = nn.Linear(64, 10)       # 64 input features, 10 output classes
        
    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 784)  # [batch_size, 784]
        
        # Pass through embedding layers
        x = F.relu(self.embedding1(x))  # [batch_size, 196]
        x = F.relu(self.embedding2(x))  # [batch_size, 64]
        
        # Compute attention scores
        attention_scores = self.attention(x).squeeze(-1)  # [batch_size, 64] -> [batch_size]
        attention_weights = F.softmax(attention_scores, dim=0)  # [batch_size]
        
        # Apply attention weights to embedded features
        context = torch.sum(x * attention_weights.unsqueeze(-1), dim=0)  # [batch_size, 64]
        
        # Pass through decoder
        output = self.decoder(context)  # [batch_size, 10]
        
        return F.log_softmax(output, dim=1)