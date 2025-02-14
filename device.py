import torch

# Define the device (GPU or CPU)
def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    return device

