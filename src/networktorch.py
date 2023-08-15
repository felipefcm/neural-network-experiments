import torch
from torch import nn


class NeuralNetworkTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(784, 64),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid()
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.stack(x)
