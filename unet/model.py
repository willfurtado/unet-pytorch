"""
U-Net model architecture definition
"""

import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    U-net model architecture definition
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        layer_dims: list[int] = [64, 128, 256, 512],
    ):
        """
        Creates an instance of the `UNet` class

        Parameters:
            in_channels (int): Number of channels in the input image. Defaults to 3, for RGB images.
            out_channels (int): Number of output channels (i.e. number of segmentation classes).
            layer_dims (list[int]): List of number of channels for each intermediate layer.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_dims = layer_dims

        # 1. Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.in_channels,
                    out_channels=self.layer_dims[0],
                    kernel_size=(3, 3),
                    padding="same",
                )
            ]
        )

        for idx in range(1, len(self.layer_dims), 1):
            curr_layer = ConvBlock(
                in_channels=self.layer_dims[idx - 1],
                out_channels=self.layer_dims[idx],
                kernel_size=(3, 3),
                padding="same",
            )
            self.encoder_layers.append(curr_layer)

        # 2. Model bottleneck
        self.bottleneck = ConvBlock(
            in_channels=self.layer_dims[-1],
            out_channels=2 * self.layer_dims[-1],
            kernel_size=(3, 3),
            padding="same",
        )

        # 3. Decoder layers
        self.decoder_layers = nn.ModuleList()

        for idx in range(len(self.layer_dims) - 1, -1, -1):
            curr_up_layer = nn.ConvTranspose2d(
                in_channels=2 * self.layer_dims[idx],
                out_channels=self.layer_dims[idx],
                kernel_size=(2, 2),
                stride=(2, 2),
            )
            curr_conv_layer = ConvBlock(
                in_channels=2 * self.layer_dims[idx],
                out_channels=self.layer_dims[idx],
                kernel_size=(3, 3),
                padding="same",
            )
            self.decoder_layers.extend([curr_up_layer, curr_conv_layer])

        # 4. Final convolutional head (1x1 conv for channel reduction)
        self.final_conv = nn.Conv2d(
            in_channels=self.layer_dims[0],
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            padding="same",
        )

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the `UNet` module

        Parameters:
            x (torch.Tensor): Batch of input images of shape (B, in_channels, H, W)

        Returns:
            (torch.Tensor): Batch of output segmentation masks of shape (B, out_channels, H, W)
        """
        intermediate_activations = []

        for idx in range(0, len(self.encoder_layers), 1):
            enc_layer = self.encoder_layers[idx]
            x = enc_layer(x)
            intermediate_activations.insert(0, x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.decoder_layers), 2):
            up_layer = self.decoder_layers[idx]
            conv_layer = self.decoder_layers[idx + 1]
            x = up_layer(x)
            x = torch.cat([intermediate_activations[idx // 2], x], dim=-3)
            x = conv_layer(x)

        x = self.final_conv(x)

        return x


class ConvBlock(nn.Module):
    """
    Double convolutional block: (Conv -> BatchNorm -> ReLU) x2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        padding: str,
    ):
        """
        Creates an instance of the `ConvBlock` class

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple[int]): Size of kernel used in convolution operation.
            padding (str): Padding strategy for convolution operation.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # 1. First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.bn1 = nn.BatchNorm2d(num_features=self.out_channels)

        # 2. Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.bn2 = nn.BatchNorm2d(num_features=self.out_channels)

        # 3. Activation function
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the `ConvBlock` module

        Parameters:
            x (torch.Tensor): _description_

        Returns:
            (torch.Tensor): _description_
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        return x


def test():
    img = torch.ones(1, 3, 512, 512)
    model = UNet(in_channels=3, out_channels=1, layer_dims=[64, 128, 256, 512])
    out = model(img)

    assert (
        img.shape[2:] == out.shape[2:]
    ), f"Input / Output shape mismatch: {img.shape} vs. {out.shape}"


if __name__ == "__main__":
    test()
