from typing import Literal
from itertools import product

import numpy as np

import torch
import torch.nn as nn
from torch import tensor


class SpecNet(nn.Module):
    """
    A PyTorch neural network model for processing spectrogram data.
    """

    def __init__(self):
        """
        Initializes the SpecNet model.
        """
        super(SpecNet, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0
        )
        self.batch1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()

        # Max pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0
        )
        self.batch2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        # Max pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layer
        self.fc = nn.Linear(46656, 1)

    def forward(self, x):
        """
        Forward pass of the SpecNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the model.
        """
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        x = torch.sigmoid(x)
        return x

"""
Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." 
Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
"""

class SEBlock(nn.Module):
    """
    A PyTorch module for Squeeze-and-Excitation (SE) block.
    """

    def __init__(self, in_channels, reduction_ratio=2):
        """
        Initializes the SEBlock module.

        Args:
            in_channels (int): The number of input channels.
            reduction_ratio (int): The reduction ratio for the channel-wise excitation.
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the SEBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Squeeze-and-Excitation block.
        """
        batch_size, num_channels, _, _ = x.size()

        # Global average pooling
        y = self.avg_pool(x).view(batch_size, num_channels)

        # Channel-wise excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        # Reshape and apply SE scaling
        y = y.view(batch_size, num_channels, 1, 1)
        return x * y

class SpecNetWithSE(nn.Module):
    """
    A PyTorch neural network model with Squeeze-and-Excitation (SE) blocks for processing spectrogram data.
    """

    def __init__(self):
        """
        Initializes the SpecNetWithSE model.

        Args:
            in_channels (int): The number of input channels (default: 3).
        """
        super(SpecNetWithSE, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0
        )
        self.batch1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()

        # Max pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Squeeze-and-Excitation block 1
        self.se_block1 = SEBlock(8)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0
        )
        self.batch2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        # Max pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Squeeze-and-Excitation block 2
        self.se_block2 = SEBlock(16)

        # Fully connected layer
        self.fc = nn.Linear(46656, 1)

    def forward(self, x):
        """
        Forward pass of the SpecNetWithSE model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the model.
        """
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        # Apply Squeeze-and-Excitation block 1
        x = self.se_block1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)

        # Apply Squeeze-and-Excitation block 2
        x = self.se_block2(x)

        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SpecModelSpec:
    model: nn.Module

    @classmethod
    def get_model(cls, device):
        def wrapper(forward):
            def inner(x):
                return torch.sigmoid(forward(x))

            return inner

        cls.model.forward = wrapper(cls.model.forward)
        model = cls.model.to(device)
        return model
    
class WindowModelSpec(SpecModelSpec):
    @classmethod
    def get_model(cls, device, window_size=35, window_stride=10):
        def wrapper(forward):
            def inner(x):
                results = torch.empty(len(x))
                device = x.device
                for index, sample in enumerate(x):
                    sample = sample.cpu()
                    for i in range(1, 500):
                        if torch.sum(sample[:, :, -i]) != 0:
                            sample = sample[:, :, :-i]
                            break
                    windows = tensor(
                        np.array(
                            tuple(
                                sample[:, :, i : i + window_size].numpy()
                                for i in range(
                                    0, len(sample[0, -1]) - window_size, window_stride
                                )
                            )
                        )
                    )
                    windows = forward(windows.to(device))
                    results[index] = torch.sigmoid(torch.mean(windows))
                return results.unsqueeze(1).to(device)

            return inner

        cls.model.forward = wrapper(cls.model.forward)
        model = cls.model.to(device)
        return model

def get_module_name_specnet(
    base_model: Literal["SpecNet", "SpecNetWithSE"],
    model_type: Literal["LSTM", "Linear"],
    window: Literal["Window", "Continuous"],
    single_channel: Literal["SingleChannel"],
):
    return single_channel + window + model_type + base_model + "BasedModel"


spec_models_specnet: dict[str, SpecModelSpec] = {}
for base_model, model_type, window, single_channel in product(
    ("SpecNet", "SpecNetWithSE"),
    ("LSTM", "Linear"),
    ("Window", "Continuous")
):
    model_name = get_module_name_specnet(
        base_model, model_type, window, single_channel
    )

    locals()[model_name]: SpecModelSpec = type(
        model_name, (WindowModelSpec if window == "Window" else SpecModelSpec,), {}
    )
    if base_model == "SpecNet":
        locals()[model_name].model = SpecNet()
    elif base_model == "SpecNetWithSE":
        locals()[model_name].model = SpecNetWithSE()

    if model_type == "Linear":
        locals()[model_name].model.fc = nn.Linear(46656, 1)
    elif model_type == "LSTM":
        locals()[model_name].model.fc = nn.LSTM(46656, 1)

    spec_models_specnet[model_name] = locals()[model_name]
