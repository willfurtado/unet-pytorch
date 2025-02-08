"""
Module used for model evaluation and visualization
"""

import torch


class Metrics:
    """
    Houses metric calculation logic
    """

    def __init__(self, data_split: str):
        """
        Creates an instance of the `Metrics` class
        """
        self.data_split = data_split

    def update(self, pred: torch.Tensor, ground_truth: torch.Tensor) -> None:
        """
        Updates the `Metrics` class

        Parameters:
            pred (torch.Tensor): _description_
            ground_truth (torch.Tensor): _description_
        """
        pass

    def __repr__(self) -> str:
        """
        String representation of the `Metrics` class
        """
        return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}()"
