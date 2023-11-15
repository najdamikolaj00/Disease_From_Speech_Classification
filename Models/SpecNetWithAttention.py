import torch as tc
import torch.nn as nn

from .AttentionBlock import SelfAttention

class SpecNetWithAttention(nn.Module):
    """
    A PyTorch neural network model with self-attention for processing spectrogram data.
    """

    def __init__(self, in_channels):
        """
        Initializes the SpecNetWithAttention model.
        """
        super(SpecNetWithAttention, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=4, 
                               kernel_size=5, stride=1, padding=2)
        self.batch1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()

        # Max pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=4)
        
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, 
                               kernel_size=5, stride=1, padding=2)
        self.batch2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        
        # Max pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=4)
       
        # Self-attention layer
        self.attention = SelfAttention(14 * 14 * 16)
        
        # Fully connected layer
        self.fc = nn.Linear(14*14*16, 2)

    def forward(self, x):
        """
        Forward pass of the SpecNetWithAttention model.

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

        # Apply self-attention
        x = self.attention(x)

        # Fully connected layer
        x = self.fc(x)

        # Apply sigmoid activation
        x = tc.sigmoid(x)
        
        return x
