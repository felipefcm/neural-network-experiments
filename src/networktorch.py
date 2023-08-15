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


# class TestNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.stack = nn.Sequential(
#             nn.Linear(2, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return x * 2
#         # return self.stack(x)


# nn = TestNet()
# input = torch.tensor([[3.0, 2.0], [5.0, 1.5]])
# print('input', input)
# print('result', nn(input))

# err = torch.nn.functional.mse_loss(
#     torch.tensor([[1.0], [2.5]]),
#     torch.tensor([[1.5], [5.5]])
# )
# print('err', err, err.shape)
