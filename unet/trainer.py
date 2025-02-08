"""
Module used to launch model training and evaluation
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from unet.dataset import TrainFloodDataset, ValFloodDataset
from unet.eval import Metrics
from unet.loss import Loss


class Trainer:
    """
    Houses model training, checkpointing, and evaluation logic
    """

    def __init__(
        self,
        model: nn.Module,
        resize_height: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
        train_examples_path: str,
        val_examples_path: str,
        image_dir: str,
        mask_dir: str,
        apply_augmentations: bool = True,
        verbose: bool = False,
    ):
        """
        Creates an instance of the `Trainer` class

        Parameters:
            model (nn.Module): Model object used for training
            resize_height (int): _description_
            num_epochs (int): Number of epochs to train the model for
            batch_size (int):  Number of data examples to use in each batch
            learning_rate (float): Learning rate to use in gradient descent optimization
            device (torch.device): _description_
            train_examples_path (str): _description_
            val_examples_path (str): _description_
            image_dir (str): _description_
            mask_dir (str): _description_
            apply_augmentations (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): Whether to print out per-epoch training losses and accuracies.
        """
        self.model = model.to(device)
        self.resize_height = resize_height
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.train_examples_path = train_examples_path
        self.val_examples_path = val_examples_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.apply_augmentations = apply_augmentations
        self.verbose = verbose

        self.train_dataset = TrainFloodDataset(
            examples_path=self.train_examples_path,
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            resize_height=self.resize_height,
            apply_augmentations=self.apply_augmentations,
        )

        self.val_dataset = ValFloodDataset(
            examples_path=self.val_examples_path,
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            resize_height=self.resize_height,
            apply_augmentations=False,
        )

        self.loss = Loss()
        self.optimizer = self.get_optimizer()

        self.train_metrics = Metrics(data_split="train")
        self.val_metrics = Metrics(data_split="val")

        self.current_epoch = 0

    def train(self) -> None:
        """
        Launches a training run for `num_epochs`
        """
        for epoch in tqdm(range(1, self.num_epochs + 1)):
            self.current_epoch = epoch
            self.run_epoch()

    def run_epoch(self) -> None:
        """
        Runs one epoch of training
        """
        self.run_train_loop()
        self.train_metrics.calculate_and_reset()

        self.run_validation_loop()
        self.val_metrics.calculate_and_reset()

    def run_train_loop(self):
        """
        Runs one epoch of model training
        """
        self.model.train()

        loader = self.get_dataloader(data_split="train")

        for idx, (image_batch, mask_batch) in enumerate(
            tqdm(loader, desc="Training Loop")
        ):
            image_batch = image_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            # Run forward pass of model and calculate loss
            model_out = self.model.forward(x=image_batch)
            curr_loss = self.loss.get_loss(pred=model_out, ground_truth=mask_batch)

            # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            curr_loss.backward()
            self.optimizer.step()

            # Update aggregate training metrics
            self.train_metrics.update(pred=model_out, ground_truth=mask_batch)

            # Log losses to Weights & Biases platform
            wandb.log(data={"Train/train_loss": curr_loss.detach().item()})

    def run_validation_loop(self):
        """
        Runs one epoch of model evaluation on validation set
        """
        self.model.eval()

        loader = self.get_dataloader(data_split="val")

        running_mean_val_loss: float = 0.0
        num_batches: int = len(loader)

        for idx, (image_batch, mask_batch) in enumerate(
            tqdm(loader, desc="Validation Loop")
        ):
            image_batch = image_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            # Run model forward pass with no gradient information
            with torch.no_grad():
                model_out = self.model.forward(x=image_batch)

            # Calculate validation loss
            curr_loss = self.loss.get_loss(pred=model_out, ground_truth=mask_batch)

            # Add to running average of validation loss
            running_mean_val_loss += curr_loss / num_batches

            self.val_metrics.update(pred=model_out, ground_truth=mask_batch)

        # Log per-epoch validation loss
        wandb.log(data={"Val/val_loss": running_mean_val_loss})

    def get_dataloader(self, data_split: str) -> DataLoader:
        """
        Returns dataloader used for specific training phase

        Parameters:
            data_split (str): Training phase, either "train" or "val"

        Returns:
            (DataLoader): Dataloader yielding batches of (image, mask) tensors
        """
        if data_split == "train":
            return DataLoader(
                dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
        elif data_split == "val":
            return DataLoader(
                dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
            )

        raise ValueError(
            f"Data split '{data_split} 'not supported. Expected 'train' or 'val'."
        )

    def get_optimizer(self):
        """
        Returns the optimizer to use for training
        """
        return torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Saves PyTorch model to disk

        Parameters:
            checkpoint_dir (str): Name of directory to save checkpoint to.
        """
        checkpoint_path = os.path.join(
            checkpoint_dir, f"unet-epoch={self.current_epoch}.pth"
        )
        contents = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(contents, checkpoint_path)

    def __repr__(self) -> str:
        """
        String representation of the `Trainer` class
        """
        return f"{self.__class__.__name__}()"
