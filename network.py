
import numpy as np

class NeuralNetwork:

	def __init__(self, neuronsPerLayer):
		
		self.numLayers = len(neuronsPerLayer)
		self.neuronsPerLayer = neuronsPerLayer
		
		self.weights = []
		for current, last in zip(neuronsPerLayer[1:], neuronsPerLayer[:-1]):
			self.weights.append(np.random.randn(current, last))

	def feedForward(self, x):
		
		for w in self.weights:
			x = relu(np.dot(w, x))
		
		return x

	def backprop(self, x, expected):
		
		weightGradients = [ np.zeros(w.shape) for w in self.weights ]

		zs = []
		activation = np.array(x)
		activations = [ np.array(x) ]

		for w in self.weights:
			z = np.dot(w, activation)
			zs.append(z)
			activation = relu(z)
			activations.append(activation)

		delta = self.costDerivative(activations[-1], expected) * \
			reluDerivative(zs[-1])

		weightGradients[-1] = np.dot(delta, activations[-2].transpose())

		for layer in range(2, self.numLayers):
			z = zs[-layer]
			d = reluDerivative(z)
			delta = np.dot(self.weights[-layer + 1].transpose(), delta) * d
			weightGradients[-layer] = np.dot(
				delta, activations[-layer - 1].transpose()
			)

		return weightGradients

	def costDerivative(self, output, expected):
		return output - expected

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
	d = lambda i : 0 if i <= 0.0 else 1.0
	return np.array([ [d(a)] for a in x ])
