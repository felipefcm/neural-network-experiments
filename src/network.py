import random
import numpy as np


class NeuralNetwork:

    def __init__(self, neurons_per_layer):
        self.num_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer

        self.weights = [
            np.random.randn(current, previous) for current, previous in
            zip(neurons_per_layer[1:], neurons_per_layer[:-1])
        ]

        self.biases = [np.random.randn(y, 1) for y in neurons_per_layer[1:]]

    def feed_forward(self, x):
        self._assert_input_shape(x)

        for w, b in zip(self.weights, self.biases):
            x = activation_fn(np.dot(w, x) + b)

        return x

    def backprop(self, x, expected):
        self._assert_input_shape(x)
        self._assert_output_shape(expected)

        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.biases]

        zs = []
        activation = np.array(x)
        activations = [np.array(x)]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = activation_fn(z)
            activations.append(activation)

        delta = self.cost_derivative(
            activations[-1], expected) * activation_derivative(zs[-1])

        weight_gradients[-1] = np.dot(delta, activations[-2].transpose())
        bias_gradients[-1] = delta

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            d = activation_derivative(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * d

            weight_gradients[-layer] = np.dot(
                delta, activations[-layer - 1].transpose()
            )

            bias_gradients[-layer] = delta

        return (weight_gradients, bias_gradients)

    def adjust_single(self, lr, weight_gradients, bias_gradients):
        self.weights = [
            w - lr * nw for w, nw in
            zip(self.weights, weight_gradients)
        ]

        self.biases = [
            b - lr * nb for b, nb in
            zip(self.biases, bias_gradients)
        ]

    def adjust_mini_batch(self, mini_batch, lr):
        self._assert_input_shape(mini_batch[0][0])
        self._assert_output_shape(mini_batch[0][1])

        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.biases]

        for x, expected in mini_batch:
            input_weight_gradients, input_bias_gradients = self.backprop(
                x, expected)

            weight_gradients = [nw + dnw for nw,
                                dnw in zip(weight_gradients, input_weight_gradients)]
            bias_gradients = [nb + dnb for nb,
                              dnb in zip(bias_gradients, input_bias_gradients)]

        self.weights = [
            w - lr * nw / len(mini_batch)
            for w, nw in zip(self.weights, weight_gradients)
        ]

        self.biases = [
            b - lr * nb / len(mini_batch)
            for b, nb in zip(self.biases, bias_gradients)
        ]

        return (
            weight_gradients,
            bias_gradients
        )

    def cost_derivative(self, output, expected):
        # no need to multiply this for 2
        return output - expected

    def _assert_input_shape(self, input):
        if input.shape != (self.neurons_per_layer[0], 1):
            raise Exception(
                f'incorrect input shape {input.shape} (expected ({self.neurons_per_layer[0]},1))'
            )

    def _assert_output_shape(self, output):
        if output.shape != (self.neurons_per_layer[-1], 1):
            raise Exception(
                f'incorrect ouput shape {output.shape} (expected ({self.neurons_per_layer[-1]},1))'
            )


def activation_fn(x):
    return 1.0 / (1.0 + np.exp(-x))
    # return np.maximum(0, x)


def activation_derivative(x):
    return activation_fn(x) * (1 - activation_fn(x))
    # return np.ceil(np.minimum(1.0, np.maximum(0, x)))
