import torch
import torch.nn as nn


class Partitioner(nn.Module):
    """
    A simple MLP-based partitioning network (phi).
    This network takes flattened features from an intermediate layer of the main model
    and outputs a logit distribution over the 'n' possible partitions (triggers).
    """
    def __init__(self, num_partitions: int, feature_dim: int, hidden_dim: int = 128, **kwargs):
        """
        Args:
            num_partitions (int): The number of triggers/partitions (e.g., 4 for Projan4).
            feature_dim (int): The dimension of the flattened feature vector from the main model.
            hidden_dim (int): The size of the hidden layer.
        """
        super().__init__()
        self.num_partitions = num_partitions
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_partitions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Flattened feature tensor from the main model's intermediate layer.
        
        Returns:
            torch.Tensor: Logits over the partitions. Shape: (batch_size, num_partitions).
        """
        return self.net(x)

    def get_partition(self, x: torch.Tensor, get_prob: bool = False) -> torch.Tensor:
        """
        A helper method to get the predicted partition index or probabilities.
        """
        logits = self(x)
        if get_prob:
            return torch.softmax(logits, dim=1)
        return logits.argmax(dim=1)
