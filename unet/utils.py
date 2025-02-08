"""
Utility functions used for model training
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


def denormalize_tensor(
    image: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
):
    """
    Denormalizes a tensor image by reversing the normalization process.

    Parameters:
        image (torch.Tensor): The normalized image tensor of shape (C, H, W).
        mean (tuple or list): The mean values used for normalization (per channel).
        std (tuple or list): The standard deviation values used for normalization (per channel).

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    denormalized = image * std + mean

    return torch.clamp(denormalized, 0, 1)


def visualize_example(dataset: Dataset, idx: int, alpha: float = 0.4):
    """
    Shows training or validation example, with segmentation overlaid

    Parameters:
        dataset (_type_): _description_
        idx (int): _description_
        alpha (float, optional): _description_. Defaults to 0.4.

    Returns:
        (_type_): _description_
    """
    image, mask = dataset[idx]

    print(f"Image Shape: \t{image.shape}")
    print(f"Mask Shape: \t{mask.shape}")

    denorm_image = denormalize_tensor(image=image)

    fig, ax = plt.subplots()
    ax.imshow(denorm_image.permute(1, 2, 0).numpy())

    ax.imshow(mask.squeeze(), alpha=alpha)
    ax.axis("off")

    return fig, ax
