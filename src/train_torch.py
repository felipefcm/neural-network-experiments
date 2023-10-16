import graph
import timer
import torch
import pickle

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from networktorch import NeuralNetworkTorch, ConvNeuralNetworkTorch
import imagedb
from mnistdataset import MNISTDataset

writer = SummaryWriter()
tm = timer.Timer()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tm.start('training data loading')

dataset = MNISTDataset(
    './digits-train-images-idx3-ubyte',
    './digits-train-labels-idx1-ubyte',
    # num=100
)
t10k_dataset = MNISTDataset(
    './digits-t10k-images-idx3-ubyte',
    './digits-t10k-labels-idx1-ubyte',
)

# dataset = MNISTDataset(
#     './fashion-train-images-idx3-ubyte',
#     './fashion-train-labels-idx1-ubyte',
#     # num=100
# )
# dataset = MNISTDataset(
#     './kmnist-train-images-idx3-ubyte',
#     './kmnist-train-labels-idx1-ubyte',
#     # num=100
# )

tm.stop()

# train_dataset, test_dataset = torch.utils.data.random_split(
#     dataset,
#     [50000, 10000],
# )
train_dataset = dataset
test_dataset = t10k_dataset

# nn = NeuralNetworkTorch().to(device)
nn = ConvNeuralNetworkTorch().to(device)

# optimiser = torch.optim.SGD(nn.parameters(), lr=0.2)
optimiser = torch.optim.Adam(nn.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimiser, step_size=1, gamma=0.9)

progress = []

epochs = 10
batch_size = 5

train_dataset = train_dataset + test_dataset

training_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

validation_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1000,
)

for epoch in range(epochs):
    # tm.start('training')
    nn.train()
    for images_batch, expecteds_batch in training_loader:
        images_batch = images_batch.to(device)
        expecteds_batch = expecteds_batch.to(device)

        output = nn(images_batch)
        loss = torch.nn.functional.cross_entropy(output, expecteds_batch)
        writer.add_scalar('Loss/train', loss, epoch)

        loss.backward()
        optimiser.step()
        nn.zero_grad()

    lr_scheduler.step()

    # tm.stop()

    # tm.start('validation')
    nn.eval()
    with torch.no_grad():
        num_correct = 0
        for images_batch, expecteds_batch in validation_loader:
            images_batch = images_batch.to(device)
            expecteds_batch = expecteds_batch.to(device)

            output = nn(images_batch)

            max_indices = output.argmax(dim=1)
            output_onehot = torch.nn.functional.one_hot(
                max_indices,
                num_classes=10
            ).to(torch.float)

            correct = (output_onehot == expecteds_batch).all(dim=1)
            num_correct += correct.count_nonzero().item()

    # tm.stop()

    rate = int(100 * num_correct / len(test_dataset))
    progress.append(rate)
    print(
        f'Finished epoch #{epoch}: {num_correct}/{len(test_dataset)} ({rate}%)')

with open('cnn_model.bin', 'wb') as f:
    # pickle.dump(nn, f)
    torch.save(nn, f)

writer.flush()
# graph.draw_cool_graphs([], [], epochs, progress)
