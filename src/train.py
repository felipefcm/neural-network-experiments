import numpy as np
import math
import graph

from network import NeuralNetwork
import imagedb

train_images, train_labels = imagedb.load_training()
print('Training data loaded')
print(f'{len(train_images)} images, {len(train_labels)} labels')

nn = NeuralNetwork([784, 32, 32, 16, 10])

wgs = []
bgs = []

batch_size = 100
num_batches = math.ceil(len(train_images) / batch_size)
epochs = 1

for epoch in range(epochs):
    for idx in range(num_batches):
        images = train_images[idx:idx + batch_size]
        labels = train_labels[idx:idx + batch_size]

        input_images = [
            imagedb.convert_image_to_input(image) for image in images
        ]

        output_labels = [
            imagedb.convert_label_to_output(label) for label in labels
        ]

        mini_batch = list(zip(input_images, output_labels))

        wg, bg = nn.adjust_mini_batch(mini_batch, 0.01)
        wgs.append(graph.sum_delta_gradients(wg))
        bgs.append(graph.sum_delta_gradients(bg))

        print(f'Processed batch #{epoch}:{idx}')

# -----------------------

num_tests = int(len(train_images) / 10)
correct = 0

for c in range(num_tests):
    test_image = train_images[c]
    test_label = train_labels[c]

    input_image = imagedb.convert_image_to_input(test_image)
    output_label = imagedb.convert_label_to_output(test_label)

    result = nn.feed_forward(input_image)
    label = imagedb.convert_output_to_label(result)

    if test_label == label:
        correct += 1

print(f'Error rate: {100 * ((num_tests - correct) / num_tests)}%')

graph.draw_gradients_sum(wgs, bgs)
