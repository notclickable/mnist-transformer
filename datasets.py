import torchvision.transforms as transforms
import torchvision
import torch
import torch.utils.data
import random

batch_size = 100

class GridMNIST(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self.num_samples = len(base_dataset)
    
    def __len__(self):
        return self.num_samples // 4  # Since we use 4 images per grid
    
    def __getitem__(self, idx):
        # Get 4 random indices
        indices = random.sample(range(self.num_samples), 4)
        images = []
        labels = []
        
        # Get the images and labels
        for i in indices:
            img, label = self.dataset[i]
            images.append(img)
            labels.append(label)
        
        # Create 2x2 grid
        top = torch.cat([images[0], images[1]], dim=2)  # Concatenate horizontally
        bottom = torch.cat([images[2], images[3]], dim=2)
        grid = torch.cat([top, bottom], dim=1)  # Concatenate vertically
        # Stack labels into tensor
        labels = torch.tensor(labels)
        
        return grid, labels

# Load the base MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
base_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
base_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create grid datasets
train_dataset = GridMNIST(base_train_dataset)
test_dataset = GridMNIST(base_test_dataset)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)