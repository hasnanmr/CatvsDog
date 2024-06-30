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
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),                      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),   
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])        
])

data_dir = 'PetImages'

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Train, val, test split dataset size
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Apply transformations to the split dataset
train_dataset = Subset(datasets.ImageFolder(data_dir, transform=transform), train_dataset.indices)
val_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_transform),val_dataset.indices)
test_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_transform),test_dataset.indices)

# Data loader for training later
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Show the length of each train, val, and test 
print(f'train dataset count are : {len(train_dataset)}')
print(f'test dataset count are : {len(test_dataset)}')
print(f'val dataset count are : {len(val_dataset)}')

# Function for visualize dataset
def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{dataset.classes[labels[j]]:5s}' for j in range(32)))