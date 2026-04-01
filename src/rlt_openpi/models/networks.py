"""Shared neural network building blocks."""

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Multi-layer perceptron with input LayerNorm.

    Architecture: LayerNorm → [Linear → ReLU] × num_hidden_layers → Linear

    Used by the actor and critic networks (Steps 6+).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        prev_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
