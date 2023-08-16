import graph
import timer
import torch

from torch.utils.data import TensorDataset, DataLoader
from networktorch import NeuralNetworkTorch
import imagedb

tm = timer.Timer()

tm.start('training data loading')
train_images, train_labels = imagedb.load_training()
tm.stop()

tm.start('training data preparation')

input_images = torch.tensor(train_images, dtype=torch.float)

expected_labels = torch.nn.functional.one_hot(
    torch.tensor(train_labels),
    num_classes=10
).float()

tm.stop()
print(f'{len(train_images)} images, {len(train_labels)} labels')

nn = NeuralNetworkTorch().to('cuda')
optimiser = torch.optim.SGD(nn.parameters(), lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.9)

progress = []

epochs = 15
batch_size = 10

dataset = TensorDataset(input_images, expected_labels)

training_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    dataset=dataset,
    batch_size=1000,
)

for epoch in range(epochs):
    tm.start('training')
    nn.train()
    for images_batch, expecteds_batch in training_loader:
        images_batch = images_batch.to('cuda')
        expecteds_batch = expecteds_batch.to('cuda')

        output = nn(images_batch)
        loss = torch.nn.functional.mse_loss(output, expecteds_batch)

        loss.backward()
        optimiser.step()
        nn.zero_grad()

    lr_scheduler.step()

    tm.stop()

    tm.start('validation')
    nn.eval()
    with torch.no_grad():
        num_correct = 0
        for images_batch, expecteds_batch in validation_loader:
            images_batch = images_batch.to('cuda')
            expecteds_batch = expecteds_batch.to('cuda')

            output = nn(images_batch)

            max_indices = output.argmax(dim=1)
            output_onehot = torch.nn.functional.one_hot(
                max_indices,
                num_classes=10
            ).to(torch.float)

            correct = (output_onehot == expecteds_batch).all(dim=1)
            num_correct += correct.count_nonzero().item()

    tm.stop()

    rate = int(100 * num_correct / len(dataset))
    progress.append(rate)
    print(f'Finished epoch #{epoch}: {num_correct}/{len(dataset)} ({rate}%)')


graph.draw_cool_graphs([], [], epochs, progress)
