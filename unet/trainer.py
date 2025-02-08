"""
Module used to launch model training and evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
            model (Model): Model object used for training
            num_epochs (int): Number of epochs to train the model for
            batch_size (int): Number of data examples to use in each batch
            learning_rate (float): Learning rate to use in gradient descent optimization
            verbose (bool): Whether to print out per-epoch training losses and accuracies
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

    def train(self) -> None:
        """
        Launches a training run for `num_epochs`
        """
        for epoch in tqdm(range(1, self.num_epochs + 1)):
            self._run_epoch()

    def _run_epoch(self) -> None:
        """
        Runs one epoch of training
        """
        self._run_train_loop()
        self._run_validation_loop()

    def _run_train_loop(self):
        """
        Runs one epoch of model training
        """
        self.model.train()
        self.optimizer.zero_grad()

        loader = self.get_train_dataloader()

        for idx, (image_batch, mask_batch) in enumerate(
            tqdm(loader, desc="Training Loop")
        ):
            image_batch = image_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            # Run forward pass of model
            model_out = self.model.forward(x=image_batch)

            # Calculate loss and run backpropagation
            curr_loss = self.loss.get_loss(pred=model_out, ground_truth=mask_batch)
            curr_loss.backward()

            # Take optimizer step and clear gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update aggregate training metrics
            self.train_metrics.update(pred=model_out, ground_truth=mask_batch)

    def _run_validation_loop(self):
        """
        Runs one epoch of model evaluation on validation set
        """
        self.model.eval()

        loader = self.get_validation_dataloader()

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

    def get_train_dataloader(self):
        """
        Returns dataloader used for training loop
        """
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def get_validation_dataloader(self):
        """
        Returns dataloader used for validation loop
        """
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def get_optimizer(self):
        """
        Returns the optimizer to use for training
        """
        return torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def __repr__(self) -> str:
        """
        String representation of the `Trainer` class
        """
        return f"{self.__class__.__name__}()"
