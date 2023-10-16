import numpy as np
import math
import graph
import random
import timer

from network import NeuralNetwork
import imagedb

tm = timer.Timer()

tm.start('training data loading')
train_images, train_labels = imagedb.load_training()
tm.stop()

tm.start('training data preparation')
input_images = [
    imagedb.convert_image_to_input(image) for image in train_images
]

expected_labels = [
    imagedb.convert_label_to_output(label) for label in train_labels
]

training_data = list(zip(input_images, expected_labels))
tm.stop()


print(f'{len(train_images)} images, {len(train_labels)} labels')


def evaluate(test_data):
    correct = 0

    for test_image, test_label in test_data:
        result = nn.feed_forward(test_image)

        result_label = imagedb.convert_output_to_label(result)
        expected_label = imagedb.convert_output_to_label(test_label)

        if expected_label == result_label:
            correct += 1

    return correct


wgs = []
bgs = []
progress = []

nn = NeuralNetwork([784, 16, 16, 10])

epochs = 50
batch_size = 1
num_tests = 10000

num_batches = math.ceil(len(training_data) / batch_size)

for epoch in range(epochs):
    for idx in range(num_batches):
        mini_batch = training_data[idx:idx + batch_size]

        wg, bg = nn.adjust_mini_batch(mini_batch, 0.001)
        # wgs.append(graph.sum_delta_gradients(wg))
        # bgs.append(graph.sum_delta_gradients(bg))

    correct = evaluate(training_data[:num_tests])
    progress.append(int(100 * correct / num_tests))

    print(f'Finished epoch #{epoch}: {correct}/{num_tests}')
    random.shuffle(training_data)

# -----------------------

num_tests = 60000
correct = evaluate(training_data[:num_tests])
print(f'Error rate: {100 * ((num_tests - correct) / num_tests)}%')

# graph.draw_cool_graphs(wgs, bgs, epochs, progress)
