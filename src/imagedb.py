import numpy as np


def load_training():
    images_file = open('./train-images-idx3-ubyte', 'rb')
    labels_file = open('./train-labels-idx1-ubyte', 'rb')

    images = load_images(images_file)
    labels = load_labels(labels_file)

    return images, labels


def load_images(file):
    magic = bytes_to_int(file.read(4))
    if (magic != 2051):
        raise RuntimeError('Wrong file for images')

    num_images = bytes_to_int(file.read(4))
    num_rows = bytes_to_int(file.read(4))
    num_cols = bytes_to_int(file.read(4))

    images = []

    for i in range(num_images):
        image = []

        for p in range(num_rows * num_cols):
            image.append(bytes_to_int(file.read(1)))

        images.append(image)

    return images


def load_labels(file):
    magic = bytes_to_int(file.read(4))
    if (magic != 2049):
        raise RuntimeError('Wrong file for labels')

    numLabels = bytes_to_int(file.read(4))

    labels = []

    for l in range(numLabels):
        labels.append(bytes_to_int(file.read(1)))

    return labels


def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='big')