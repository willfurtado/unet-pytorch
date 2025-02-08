"""
Loss functions and calculations used for training
"""

from typing import Optional

import torch
import torch.nn as nn


class Loss:
    """
    Houses loss calculation logic
    """

    POS_WEIGHT: Optional[torch.Tensor] = None

    def __init__(self, reduction: str = "mean"):
        """
        Creates an instance of the `Loss` class

        Parameters:
            reduction (str): Type of reduction to perform to intermediate loss tensor. Defaults
                to "mean" reduction, and will take an average loss value across the entire batch.
        """
        self.reduction = reduction

        # Use BinaryCrossEntropy loss function for single-class pixelwise classification
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.POS_WEIGHT, reduction=self.reduction
        )

    def get_loss(self, pred: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss between model output and ground truth data

        Parameters:
            pred (torch.Tensor): Tensor of model logits, of shape (B, C, H, W)
            ground_truth (torch.Tensor): Tensor of ground truth mask, of shape (B, C, H, W)

        Returns:
            (torch.Tensor): Possibly singleton tensor representing loss calculation
        """
        return self.loss_fn(pred, ground_truth)

    def __repr__(self) -> str:
        """
        String representation of the `Loss` class
        """
        return f"{self.__class__.__name__}()"
