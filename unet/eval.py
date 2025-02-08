"""
Module used for model evaluation and visualization
"""

import torch

import wandb


class Metrics:
    """
    Houses metric calculation logic
    """

    def __init__(self, data_split: str, eps: float = 1e-8):
        """
        Creates an instance of the `Metrics` class

        Parameters:
            data_split (str): Training phase for metric calculation. Either "train" or "val".
            eps (float, optional): Epsilon to avoid division-by-zero error. Defaults to 1e-8.
        """
        self.data_split = data_split
        self.eps = eps

        self.num_correct: int = 0
        self.num_pixels: int = 0
        self.num_updates: int = 0
        self.dice_score_sum: float = 0.0

    def update(self, pred: torch.Tensor, ground_truth: torch.Tensor) -> None:
        """
        Updates the `Metrics` class

        Parameters:
            pred (torch.Tensor): _description_
            ground_truth (torch.Tensor): _description_
        """
        out = torch.sigmoid(pred)

        # Snap all outputs >0.5 to 1, and 0 otherwise
        binary_out = (out > 0.5).float()

        self.num_correct += (binary_out == ground_truth).sum()
        self.num_pixels += binary_out.numel()
        self.num_updates += 1
        self.dice_score_sum += (
            2
            * (binary_out * ground_truth).sum()
            / ((binary_out + ground_truth).sum() + self.eps)
        )

    def calculate_and_reset(self) -> None:
        """
        Calculates per-epoch metrics, logs to WandB, and resets statistics
        """
        wandb.log(
            data={
                f"{self.data_split.title()}/{self.data_split}_acc": self.accuracy,
                f"{self.data_split.title()}/{self.data_split}_dice": self.dice_score,
            }
        )
        self._reset()

    def _reset(self) -> None:
        """
        Resets the running statistics for this `Metrics` instance
        """
        self.num_correct: int = 0
        self.num_pixels: int = 0
        self.num_updates: int = 0
        self.dice_score_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        """
        Returns the naive accuracy over the dataset
        """
        return (self.num_correct / self.num_pixels) if self.num_pixels else 0.0

    @property
    def dice_score(self) -> float:
        """
        Returns the dice score over the dataset
        """
        return (self.dice_score_sum / self.num_updates) if self.num_updates else 0.0

    def __repr__(self) -> str:
        """
        String representation of the `Metrics` class
        """
        return f"{self.__class__.__name__}(split={self.data_split})"
