"""
Vaswani, Ashish, et al. 
"Attention is all you need." Advances in neural information processing systems 30 (2017).
"""
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    A PyTorch module for self-attention mechanism.
    """

    def __init__(self, in_dim):
        """
        Initializes the SelfAttention module.

        Args:
            in_dim (int): The input dimension.
        """
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        """
        Forward pass of the SelfAttention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Calculate attention scores
        scores = tc.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = tc.matmul(attention_weights, value)

        return output
