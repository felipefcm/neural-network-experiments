import matplotlib.pyplot as plt
import numpy as np


def sum_delta_gradients(gradients):
    acc = 0
    for layer_grads in gradients:
        acc += np.sum(np.abs(layer_grads))

    return acc


def draw_cool_graphs(wgs, bgs, epochs, progress):
    fig = plt.figure()

    # wg_ax = fig.add_subplot(2, 2, 1)
    # wg_ax.set_title('sum of gradients (weights)')
    # wg_ax.plot(list(range(len(wgs))), wgs)

    # bg_ax = fig.add_subplot(2, 2, 2)
    # bg_ax.set_title('sum of gradients (biases)')
    # bg_ax.plot(list(range(len(bgs))), bgs, color='green')

    # p_ax = fig.add_subplot(2, 2, 3)
    p_ax = fig.add_subplot()
    p_ax.set_title('success rate')
    p_ax.plot(list(range(epochs)), progress, color='red')
    p_ax.set_xticks(list(range(epochs)))

    plt.show()
