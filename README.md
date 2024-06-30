# CatvsDog Image Classification
**This is a project of MLOps for image classification**. The project will be used pytorch framework and transformer based for image classification task Vision Transformer which is achieved novelty in computer vision task.

The dataset is downloaded from [CatvsDog Kaggle dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset/data)
The datasets are consist of:
1. Dog --> 12490 images
2. Cat --> 12470 images

## Preprocessing step
1. Resize the images to 224x224
2. Transform into tensor type
3. Normalize the tensor using mean and STD
4. Split the dataset into train evaluation and testing dataset (80, 10, 10)
5. Create dataloader for train, evaluation and testing dataset using batch size of 32