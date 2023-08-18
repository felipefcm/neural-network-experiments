import graph
import timer
import torch

from torch.utils.data import TensorDataset, DataLoader
from networktorch import NeuralNetworkTorch, ConvNeuralNetworkTorch
import imagedb
from mnistdataset import MNISTDataset

tm = timer.Timer()

tm.start('training data loading')

# dataset = MNISTDataset(
#     './digits-train-images-idx3-ubyte',
#     './digits-train-labels-idx1-ubyte',
#     # num=100
# )
dataset = MNISTDataset(
    './fashion-train-images-idx3-ubyte',
    './fashion-train-labels-idx1-ubyte',
    # num=100
)
# dataset = MNISTDataset(
#     './kmnist-train-images-idx3-ubyte',
#     './kmnist-train-labels-idx1-ubyte',
#     # num=100
# )

tm.stop()

# nn = NeuralNetworkTorch().to('cuda')
nn = ConvNeuralNetworkTorch().to('cuda')

# optimiser = torch.optim.SGD(nn.parameters(), lr=0.2)
optimiser = torch.optim.Adam(nn.parameters(), lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.9)

progress = []

epochs = 15
batch_size = 10

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

    # lr_scheduler.step()

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
