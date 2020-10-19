
import random
import numpy as np

class NeuralNetwork:

	def __init__(self, neuronsPerLayer):
		
		self.numLayers = len(neuronsPerLayer)
		self.neuronsPerLayer = neuronsPerLayer
		
		self.weights = []

		for current, last in zip(neuronsPerLayer[1:], neuronsPerLayer[:-1]):
			self.weights.append(np.random.randn(current, last))

		self.biases = [ np.random.randn(y, 1) for y in neuronsPerLayer[1:] ]
		
	def feedForward(self, x):
		
		for b, w in zip(self.biases, self.weights):
			x = relu(np.dot(w, x) + b)
		
		return x

	def sgd(self, trainingData, epochs, miniBatchSize, lr, test=None):

		n = len(trainingData)
		for e in range(epochs):
			
			random.shuffle(trainingData)
			miniBatches = [
				trainingData[k : k + miniBatchSize]
				for k in range(0, n, miniBatchSize)
			]

			for miniBatch in miniBatches:
				self.adjustWeightsForMiniBatch(miniBatch, lr)


	def backprop(self, x, expected):
		
		weightGradients = [ np.zeros(w.shape) for w in self.weights ]
		biasGradients = [ np.zeros(b.shape) for b in self.biases ]

		zs = []
		activation = np.array(x)
		activations = [ np.array(x) ]

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = relu(z)
			activations.append(activation)

		delta = self.costDerivative(activations[-1], expected) * \
			reluDerivative(zs[-1])

		weightGradients[-1] = np.dot(delta, activations[-2].transpose())
		biasGradients[-1] = delta

		for layer in range(2, self.numLayers):
			
			z = zs[-layer]
			d = reluDerivative(z)
			delta = np.dot(self.weights[-layer + 1].transpose(), delta) * d
			
			weightGradients[-layer] = np.dot(
				delta, activations[-layer - 1].transpose()
			)

			biasGradients[-layer] = delta

		return (biasGradients, weightGradients)

	def adjustWeights(self, lr, weightGradients):
		self.weights = [
			w - lr * nw 
			for w, nw in zip(self.weights, weightGradients)
		]

	def adjustWeightsForMiniBatch(self, miniBatch, lr):

		weightGradients = [ np.zero(w.shape) for w in self.weights ]
		biasGradients = [ np.zero(b.shape) for b in self.biases ]

		for x, expected in miniBatch:
			deltaBias, deltaWeight = self.backprop(x, expected)
			weightGradients = [ nw + dnw for nw, dnw in zip(weightGradients, deltaWeight) ]
			biasGradients = [ nb + dnb for nb, dnb in zip(biasGradients, deltaBias) ]

		# self.weights = [w - (lr / len(miniBatch)) ]

	def costDerivative(self, output, expected):
		return output - expected

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
	d = lambda i : 0 if i <= 0.0 else 1.0
	return np.array([ [d(a)] for a in x ])
