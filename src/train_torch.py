import numpy as np
import math
import graph
import random
import timer
import torch

from torch.utils.data import TensorDataset, DataLoader
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

input_images = torch.tensor(train_images, dtype=torch.float32)

expected_labels = torch.tensor([
    convert_label_to_output(label) for label in train_labels
])

dataset = TensorDataset(input_images, expected_labels)

tm.stop()
print(f'{len(train_images)} images, {len(train_labels)} labels')

nn = NeuralNetworkTorch().to('cuda')
optimiser = torch.optim.SGD(nn.parameters(), lr=0.1)

# for name, param in nn.named_parameters():
#     print(name, '==>', param.size())

wgs = []
bgs = []
progress = []

epochs = 20
batch_size = 20
num_tests = 30000

num_batches = math.ceil(len(train_images) / batch_size)

training_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
)

tm.start('training')
for epoch in range(epochs):
    nn.train()
    for images_batch, expecteds_batch in training_loader:
        images_batch = images_batch.to('cuda')
        expecteds_batch = expecteds_batch.to('cuda')

        output = nn(images_batch)
        loss = torch.nn.functional.mse_loss(output, expecteds_batch)

        loss.backward()
        optimiser.step()
        nn.zero_grad()

    nn.eval()
    with torch.no_grad():
        for images_batch, expecteds_batch in validation_loader:
            images_batch = images_batch.to('cuda')
            expecteds_batch = expecteds_batch.to('cuda')

            output = nn(images_batch)
            max_indices = output.argmax(dim=1)
            expected_output = torch.nn.functional.one_hot(
                max_indices,
                num_classes=10
            ).to(torch.float)

    # progress.append(int(100 * correct / num_tests))
    # print(f'Finished epoch #{epoch}: {correct}/{partial_test}')

tm.stop()
# -----------------------

# nn.eval()
# tm.start('evaluation')
# correct = evaluate(input_images[:num_tests], expected_labels[:num_tests])
# tm.stop()
# print(f'Error rate: {100 * ((num_tests - correct) / num_tests)}%')
# print(f'Success rate: {100 * (correct / num_tests)}%')

# graph.draw_cool_graphs(wgs, bgs, epochs, progress)
