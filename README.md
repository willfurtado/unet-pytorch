## U-Net for Flood Segmentation

Houses a PyTorch implementation of the U-Net model architecture used for flood segmentation prediction. 

### Architecture

Original Paper: https://arxiv.org/abs/1505.04597

### `U-Net` Package

A Python module, `unet`, is provided to handle all aspects of training and evaluation. The package includes the following modules:
- `dataset`: Train and validation dataset classes, used to organize and load flood area images and masks
- `eval`: Model evaluation classes used to calculate training and validation metrics
- `main`: Main launch script for a training run, utilizing all other modules
- `model`: PyTorch implementation of `U-Net` model architecture
- `params`: Dataclass containing all experiment hyperparameters, logged to Weights & Biases
- `trainer`: Trainer class used to launch, monitor, and log training experiments
- `utils`: Miscellaneous data loading and visualization functions

### Results

Following are the experimental results for the flood segmentation dataset:

Weights & Biases Project: https://wandb.ai/willfurtado/unet

