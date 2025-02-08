"""
Main launch script for U-Net training
"""

import wandb
from unet.model import UNet
from unet.params import TrainingParams
from unet.trainer import Trainer


def train():
    """
    Run full training job from the command line
    """

    # Instantiate dataclass of training hyperparameters
    params = TrainingParams()

    # Create new run using Weights & Biases
    run = wandb.init(
        project=params.wandb_project_name,
        config={
            "architecture": params.architecture,
            "dataset": params.dataset,
            "learning_rate": params.learning_rate,
            "batch_size": params.batch_size,
            "num_epochs": params.num_epochs,
            "input_shape": (params.resize_height, params.resize_height),
            "layer_dims": params.layer_dims,
        },
    )

    # Create `UNet` model for training
    model = UNet(
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        layer_dims=params.layer_dims,
    )

    # Create `Trainer` object used to launch training job
    trainer = Trainer(
        model=model,
        resize_height=params.resize_height,
        num_epochs=params.num_epochs,
        batch_size=params.batch_size,
        learning_rate=params.learning_rate,
        device=params.device,
        train_examples_path=params.train_examples_path,
        val_examples_path=params.val_examples_path,
        image_dir=params.image_dir,
        mask_dir=params.mask_dir,
        apply_augmentations=params.apply_augmentations,
        verbose=params.verbose,
    )

    # Launch training job
    trainer.train()

    # Save final checkpoint
    trainer.save_checkpoint(checkpoint_dir=run.dir)

    # Upon completion, finish `wandb` run
    wandb.finish()


if __name__ == "__main__":
    train()
