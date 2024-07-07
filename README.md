# CatvsDog Image Classification
**This is a repository of image preparation before use in training**. 
This repo contains preprocessing and a subset of a dataset that has been saved with a .pth file.
Preprocessing of images ensures the images are easy to use and fed into a neural network model to train.

The dataset is downloaded from [CatvsDog Kaggle dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset/data)
The datasets consist of the following images:
1. Dog --> 12490 images
2. Cat --> 12470 images

## Preprocessing step
1. Resize the images to **224x224**
2. Transform into tensor type
3. Normalize the tensor using mean and STD
4. Split the dataset into train evaluation and testing datasets (80, 10, 10) with generator seed to ensure the split always generates the exact number
5. Save the subset into **.pt** file so it can be loaded next for training the model
6. The sizes of the subsets for data training, evaluation, and testing are **19966**, **2497**, and **2495**, respectively.

## How to use saved subset
1. Load the subset using the following code:
'''bash
train_dataset = torch.load('train_dataset.pt')
val_dataset = torch.load('val_dataset.pt')
test_dataset = torch.load('test_dataset.pt')
'''
2. Use the dataset in the training model 
'''bash
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
'''