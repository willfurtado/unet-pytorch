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
            weight (float):
        """
        self.reduction = reduction

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.POS_WEIGHT, reduction=self.reduction
        )

    def get_loss(self, pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Calculates the model loss

        Parameters:
            pred (torch.Tensor): _description_
            ground_truth (torch.Tensor): _description_

        Returns:
            (float): _description_
        """
        return self.loss_fn(pred, ground_truth)

    def __repr__(self) -> str:
        """
        String representation of the `Loss` class
        """
        return f"{self.__class__.__name__}()"
