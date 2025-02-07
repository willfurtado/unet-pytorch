"""
Flood segmentation dataset and dataloader classes
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from abc import ABC, abstractmethod


class BaseFloodDataset(Dataset, ABC):
    """
    Base dataset implementation for flood segmentation
    """

    def __init__(self, examples_path: str, image_dir: str, mask_dir: str):
        """
        Creates an instance of the `BaseFloodDataset` class

        Parameters:
            examples_path (str):
            image_dir (str):
            mask_dir (str):
        """
        self.examples_path = examples_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Read examples CSV file into DataFrame format
        examples_df = pd.read_csv(self.examples_path)

        self.image_filenames = [
            os.path.join(self.image_dir, name) for name in examples_df["image"]
        ]
        self.mask_filenames = [
            os.path.join(self.mask_dir, name) for name in examples_df["mask"]
        ]

    @abstractmethod
    def _prepare_datapoint(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads image, mask pair for training

        Parameters:
            idx (int): Index used to access example

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Image, mask pair as PyTorch tensors
        """
        pass

    @abstractmethod
    def _get_transforms(self):
        """
        Returns an albumentations transform
        """
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a training example at the desired `idx`

        Parameters:
            idx (int): Index used to access example

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Image, mask pair as PyTorch tensors
        """
        return self._prepare_datapoint(idx=idx)

    def __len__(self) -> int:
        """
        Returns the number of examples in the dataset
        """
        return len(self.image_filenames)

    def __repr__(self) -> str:
        """
        Returns the string representation of the dataset
        """
        return f"{self.__class__.__name__}(examples={self.examples_path})"


class TrainFloodDataset(BaseFloodDataset):
    """
    Train-specific flood dataset
    """

    def _prepare_datapoint(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads image, mask pair for training

        Parameters:
            idx (int): Index used to access example

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Image, mask pair as PyTorch tensors
        """
        pass

    def _get_transforms(self):
        """
        Returns an albumentations transform
        """
        pass


class ValFloodDataset(BaseFloodDataset):
    """
    Validation-specific flood dataset
    """

    def _prepare_datapoint(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads image, mask pair for validation

        Parameters:
            idx (int): Index used to access example

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Image, mask pair as PyTorch tensors
        """
        image_path = self.image_filenames[idx]
        mask_path = self.mask_filenames[idx]

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

    def _get_transforms(self):
        """
        Returns an albumentations transform
        """
        pass
