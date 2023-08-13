import matplotlib.pyplot as plt
import numpy as np


def sum_delta_gradients(gradients):
    acc = 0
    for layer_grads in gradients:
        acc += np.sum(np.abs(layer_grads))

    return acc


def draw_gradients_sum(wgs, bgs):
    fig, ax = plt.subplots(nrows=2, ncols=1, label='gradients')

    ax[0].plot(list(range(len(wgs))), wgs, label='weights')
    ax[1].plot(list(range(len(bgs))), bgs, label='biases', color='green')

    plt.show()
