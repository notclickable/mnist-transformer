from datasets import train_dataset

# view the first output of the dataset as an image
import matplotlib.pyplot as plt

# Get the first image and label from the dataset
image, label = train_dataset[0]

# Display the image
plt.figure(figsize=(8,8))
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Grid of MNIST digits with labels: {label.tolist()}")
plt.axis('off')
plt.show()





