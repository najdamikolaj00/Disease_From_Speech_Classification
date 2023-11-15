"""
Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." 
Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
"""

import torch.nn as nn

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