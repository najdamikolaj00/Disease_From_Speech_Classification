import torch as tc
import torch.nn as nn

from .SEBlock import SEBlock

class SpecNetWithSE(nn.Module):
    """
    A PyTorch neural network model with Squeeze-and-Excitation (SE) blocks for processing spectrogram data.
    """

    def __init__(self, in_channels):
        """
        Initializes the SpecNetWithSE model.

        Args:
            in_channels (int): The number of input channels (default: 3).
        """
        super(SpecNetWithSE, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, 
                               kernel_size=3, stride=1, padding=0)
        self.batch1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()

        # Max pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Squeeze-and-Excitation block 1
        self.se_block1 = SEBlock(8)
        
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, 
                               kernel_size=3, stride=1, padding=0)
        self.batch2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        
        # Max pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Squeeze-and-Excitation block 2
        self.se_block2 = SEBlock(16)
       
        # Fully connected layer
        self.fc = nn.Linear(46656, 2)

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
        
        x = tc.sigmoid(x)
        return x
