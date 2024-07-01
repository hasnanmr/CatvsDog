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
5. Save the subset into **.pth** file so it can be loaded next for training the model
6. The sizes of the subsets for data training, evaluation, and testing are **19966**, **2497**, and **2495**, respectively.