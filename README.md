# Deep Learning on Imagenette

This project implements an image classification pipeline using a custom Convolutional Neural Network (CNN) trained on the [Imagenette dataset](https://github.com/fastai/imagenette). Built with PyTorch Lightning, the model learns to classify images using data augmentation, regularization, and efficient training callbacks.



## Overview

- Dataset: Imagenette (10-class subset of ImageNet)
- Model: 3-layer CNN with batch norm, ReLU, and max pooling
- Tools: PyTorch Lightning, Torchvision, Torchmetrics
- Training Enhancements:
  - Data Augmentation
  - Early Stopping
  - Model Checkpointing



## Features

- CNN with increasing filters: 16 → 32 → 64
- Fully connected layers with ReLU and dropout
- Custom `LightningModule` for training/validation logic
- Accuracy metric tracking
- Saved best model weights (`model_reg_weights.pth`)

