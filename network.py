
import numpy as np

class NeuralNetwork:

	def __init__(self, neuronsPerLayer):
		
		self.numLayers = len(neuronsPerLayer);
		self.neuronsPerLayer = neuronsPerLayer;
		
		self.weights = [];
		for current, last in zip(neuronsPerLayer[1:], neuronsPerLayer[:-1]):
			self.weights.append(np.random.randn(current, last));

	def feedForward(self, x):
		for layer in range(self.numLayers):
			x = activation(np.dot(self.weights[layer], x));
			

def activation(x):
    return np.maximum(0, x);

def activationDerivative(x):
	d = lambda a : 0 if a <= 0 else 1;
	return np.array([ d(a[0]) for a in x ]);
