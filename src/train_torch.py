import numpy as np
import math
import graph
import random
import timer
import torch

from networktorch import NeuralNetworkTorch
import imagedb


def convert_label_to_output(label: int):
    output = np.zeros(10)
    output[label] = 1.0
    return output.tolist()


def convert_output_to_label(output: torch.Tensor):
    idx = 0
    max = -100.0
    for i, val in enumerate(output):
        if val > max:
            idx = i
            max = val

    return idx


tm = timer.Timer()

tm.start('training data loading')
train_images, train_labels = imagedb.load_training()
tm.stop()

tm.start('training data preparation')

input_images = torch.tensor(train_images, dtype=torch.float32, device='cuda')

expected_labels = torch.tensor([
    convert_label_to_output(label) for label in train_labels
], device='cuda')

tm.stop()
print(f'{len(train_images)} images, {len(train_labels)} labels')


nn = NeuralNetworkTorch().to('cuda')

for name, param in nn.named_parameters():
    print(name, '==>', param.size())

optimiser = torch.optim.SGD(nn.parameters(), lr=0.1)


def evaluate(inputs, expecteds):
    correct = 0

    results = nn(inputs)

    for i, result in enumerate(results):
        result_label = convert_output_to_label(result)
        expected_label = convert_output_to_label(expecteds[i])

        if expected_label == result_label:
            correct += 1

    return correct


wgs = []
bgs = []
progress = []

epochs = 20
batch_size = 20
num_tests = 30000

num_batches = math.ceil(len(train_images) / batch_size)

for epoch in range(epochs):
    for batch in range(num_batches):
        images = input_images[batch:batch + batch_size]
        expecteds = expected_labels[batch:batch + batch_size]

        output = nn(images)
        loss = torch.nn.functional.mse_loss(output, expecteds)
        nn.zero_grad()
        loss.backward()
        optimiser.step()

    partial_test = num_tests
    correct = evaluate(input_images[:partial_test],
                       expected_labels[:partial_test])

    progress.append(int(100 * correct / num_tests))
    print(f'Finished epoch #{epoch}: {correct}/{partial_test}')

# -----------------------

nn.eval()
tm.start('evaluation')
correct = evaluate(input_images[:num_tests], expected_labels[:num_tests])
tm.stop()
print(f'Error rate: {100 * ((num_tests - correct) / num_tests)}%')
print(f'Success rate: {100 * (correct / num_tests)}%')

graph.draw_cool_graphs(wgs, bgs, epochs, progress)
