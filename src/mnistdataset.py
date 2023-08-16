import torch
from torch.utils.data import TensorDataset

import imagedb


class MNISTDataset(TensorDataset):
    def __init__(self, images_filepath: str, labels_filepath: str):
        with open(images_filepath, 'rb') as images_file:
            images = imagedb.load_images(images_file)

        with open(labels_filepath, 'rb') as labels_file:
            labels = imagedb.load_labels(labels_file)

        input_images = torch.tensor(images, dtype=torch.float)

        expected_labels = torch.nn.functional.one_hot(
            torch.tensor(labels),
            num_classes=10
        ).float()

        super.__init__(input_images, expected_labels)
