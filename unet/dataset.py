"""
Flood segmentation dataset and dataloader classes
"""

import os
from abc import ABC, abstractmethod

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseFloodDataset(Dataset, ABC):
    """
    Base dataset implementation for flood segmentation
    """

    def __init__(
        self,
        examples_path: str,
        image_dir: str,
        mask_dir: str,
        resize_height: int,
        apply_augmentations: bool = False,
    ):
        """
        Creates an instance of the `BaseFloodDataset` class

        Parameters:
            examples_path (str):
            image_dir (str):
            mask_dir (str):
            apply_augmentations (bool):
        """
        self.examples_path = examples_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resize_height = resize_height
        self.apply_augmentations = apply_augmentations

        # Read examples CSV file into DataFrame format
        examples_df = pd.read_csv(self.examples_path)

        self.image_filenames = [
            os.path.join(self.image_dir, name) for name in examples_df["image"]
        ]
        self.mask_filenames = [
            os.path.join(self.mask_dir, name) for name in examples_df["mask"]
        ]

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

        image = np.array(Image.open(image_path).convert("RGB"))

        # Convert binary mask to float, then scale between 0 and 1
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0

        transforms = self._get_transforms()

        return transforms(image=image, mask=mask)

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
        try:
            datapoint = self._prepare_datapoint(idx=idx)

            return datapoint["image"], datapoint["mask"].unsqueeze(0).float()
        except ValueError:
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self) -> int:
        """
        Returns the number of examples in the dataset
        """
        return len(self.image_filenames)

    def __repr__(self) -> str:
        """
        Returns the string representation of the dataset
        """
        return f"{self.__class__.__name__}(examples='{self.examples_path}')"


class TrainFloodDataset(BaseFloodDataset):
    """
    Train-specific flood dataset
    """

    # Constants for `HorizontalFlip` augmentation
    HORIZONTAL_FLIP_P: float = 0.5

    # Constants for `Affine` augmentation
    AFFINE_P: float = 0.7
    TRANSLATE_RANGE: tuple[int] = (-0.0625, 0.0625)
    SCALE_RANGE: tuple[int] = (1.1, 1.3)
    ROTATE_RANGE: tuple[int] = (-15, 15)

    # Constants for `ColorJitter` augmentation
    COLOR_JITTER_P: float = 0.5
    BRIGHTNESS_RANGE: tuple[int] = (0.8, 1.2)
    CONTRAST_RANGE: tuple[int] = (0.8, 1.2)
    SATURATION_RANGE: tuple[int] = (0.8, 1.2)
    HUE_RANGE: tuple[int] = (-0.5, 0.5)

    def _get_transforms(self) -> A.Compose:
        """
        Returns an albumentations transform
        """
        transforms = [
            A.LongestMaxSize(max_size=self.resize_height),
            A.PadIfNeeded(min_height=self.resize_height, min_width=self.resize_height),
        ]

        if self.apply_augmentations:
            data_augmentations = [
                A.HorizontalFlip(p=self.HORIZONTAL_FLIP_P),
                A.Affine(
                    scale=self.SCALE_RANGE,
                    translate_percent=self.TRANSLATE_RANGE,
                    rotate=self.ROTATE_RANGE,
                    mask_interpolation=cv2.INTER_LINEAR,
                    p=self.AFFINE_P,
                ),
                A.ColorJitter(
                    brightness=self.BRIGHTNESS_RANGE,
                    contrast=self.CONTRAST_RANGE,
                    saturation=self.SATURATION_RANGE,
                    hue=self.HUE_RANGE,
                    p=self.COLOR_JITTER_P,
                ),
            ]
            transforms.extend(data_augmentations)

        # In all cases, we normalize and convert to PyTorch tensor
        transforms.extend([A.Normalize(), A.pytorch.ToTensorV2()])

        return A.Compose(transforms)


class ValFloodDataset(BaseFloodDataset):
    """
    Validation-specific flood dataset
    """

    def _get_transforms(self) -> A.Compose:
        """
        Returns an albumentations transform
        """
        return A.Compose(
            [
                A.LongestMaxSize(max_size=self.resize_height),
                A.PadIfNeeded(
                    min_height=self.resize_height, min_width=self.resize_height
                ),
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ]
        )
