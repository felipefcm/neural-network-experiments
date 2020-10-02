
import numpy as np
from network import NeuralNetwork;

nn = NeuralNetwork([ 2, 3, 2 ])

x = np.array([ [0.3], [0.8] ])
y = np.array([ [1.0], [0.0] ])
print('Initial output', nn.feedForward(x))

for train in range(1000):
	weightGradients = nn.backprop(x, y)
	nn.adjustWeights(0.01, weightGradients)

print('Trained output', nn.feedForward(x))
