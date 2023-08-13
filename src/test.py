import numpy as np
from network import NeuralNetwork
import graph


nn = NeuralNetwork([1, 4, 4, 1])

x = np.array([[0.2]])
print('Initial output', nn.feed_forward(x))

wgs = []
bgs = []

num_train = 800
per_batch = 50

for t in range(num_train):
    train_inputs = [
        np.array([np.random.rand(1)]) for i in range(per_batch)
    ]

    train_expected = [
        np.array([train_inputs[i][0] * 2]) for i in range(per_batch)
    ]

    mini_batch = list(zip(train_inputs, train_expected))
    wg, bg = nn.adjust_mini_batch(mini_batch, 0.01)

    wgs.append(graph.sum_delta_gradients(wg))
    bgs.append(graph.sum_delta_gradients(bg))


# num_single = 20000
# for t in range(num_single):
#     train_input = np.array([np.random.rand(1)])
#     train_expected = np.array([train_input[0] * 2])

#     wg, bg = nn.backprop(train_input, train_expected)
#     wgs.append(sum_delta_gradients(wg))
#     bgs.append(sum_delta_gradients(bg))

#     nn.adjust_single(0.01, wg, bg)

print('Trained output', nn.feed_forward(x))
graph.draw_gradients_sum(wgs, bgs)
