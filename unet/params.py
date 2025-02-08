"""
Stores training hyperparameters
"""

from dataclasses import dataclass, field

import torch


@dataclass
class TrainingParams:
    """
    Hyperparameters used for model training
    """

    wandb_project_name: str = "unet"
    architecture: str = "U-Net"
    dataset: str = "https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation"

    in_channels: int = 3
    out_channels: int = 1
    layer_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512])

    resize_height: int = 256

    num_epochs: int = 5
    batch_size: int = 2
    learning_rate: float = 1e-3
    device: torch.device = torch.device("mps")
    train_examples_path: str = "data/train.csv"
    val_examples_path: str = "data/val.csv"
    image_dir: str = "data/image"
    mask_dir: str = "data/mask"
    apply_augmentations: bool = True
    verbose: bool = False
    verbose: bool = False
