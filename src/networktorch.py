from torch import nn


class NeuralNetworkTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(784, 64),
            nn.Sigmoid(),

            nn.Linear(64, 16),
            nn.Sigmoid(),

            nn.Linear(16, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)
