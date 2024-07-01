# Import Library
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import os

#preprocessing
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),             
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
])

data_dir = 'PetImages'  #dataset folder directory

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Train, val, test split dataset size
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Create a generator for reproducibility
manual_seed = 42
generator = torch.Generator().manual_seed(manual_seed)

# Train, val, test split dataset size
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split dataset with the manual seed generator
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# Save the datasets with confirmation
try:
    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(val_dataset, 'val_dataset.pt')
    torch.save(test_dataset, 'test_dataset.pt')
    print('The subset has been saved successfully.')
except Exception as e:
    print(f'Error occurred while saving the subset: {e}')