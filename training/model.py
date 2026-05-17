"""Feedforward policy network for behavioral cloning."""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """MLP: state features -> action logits."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STS2ValueNet(nn.Module):
    """Critic: state features -> scalar value estimate (used during PPO training only)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
