import numpy as np
import matplotlib.pyplot as plt

from network import NeuralNetwork
import imagedb

# trainImages, trainLabels = imagedb.load_training()
# print('Training data loaded')


nn = NeuralNetwork([2, 16, 32, 16, 2])

x = np.array([[0.3], [0.8]])
y = np.array([[1.0], [0.0]])
print('Initial output', nn.feed_forward(x))

for train in range(1000):
    weight_gradients, bias_gradients = nn.backprop(x, y)
    nn.adjust_single(0.001, weight_gradients, bias_gradients)

print('Trained output', nn.feed_forward(x))
