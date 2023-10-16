from torch import nn


class NeuralNetworkTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(784, 64),
            nn.Sigmoid(),

            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)


class ConvNeuralNetworkTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 14 * 14, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # we do some reshaping here simply to avoid making changes to the caller
        # so it continues to work with the fully conected network above
        x = x.reshape(-1, 1, 28, 28) / 255

        conv_output = self.conv(x)
        flat = conv_output.reshape(len(x), -1)
        final_output = self.fc(flat)

        return final_output
