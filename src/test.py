import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork

nn = NeuralNetwork([16, 8, 4, 1])

# x = np.array([[0.3], [0.8]])
x = np.array([[0.3] for i in range(16)])
print('Initial output', nn.feed_forward(x))

num_train = 20
per_batch = 100

for t in range(num_train):
    train_inputs = [
        np.array([np.random.rand(1) for a in range(16)]) for i in range(per_batch)
    ]

    train_expected = [
        np.array([1.0]) for i in range(per_batch)
    ]

    mini_batch = list(zip(train_inputs, train_expected))
    nn.adjust_mini_batch(mini_batch, 0.01)

wg, bg = nn.backprop(x, np.array([np.random.rand(1)]))
nn.adjust_single(0.01, wg, bg)

print('Trained output', nn.feed_forward(x))
