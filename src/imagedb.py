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
        images.append([
            bytes_to_int(file.read(1)) for p in range(num_rows * num_cols)
        ])

    return images


def load_labels(file):
    magic = bytes_to_int(file.read(4))
    if (magic != 2049):
        raise RuntimeError('Wrong file for labels')

    numLabels = bytes_to_int(file.read(4))

    labels = [
        bytes_to_int(file.read(1)) for l in range(numLabels)
    ]

    return labels


def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='big')


def convert_label_to_output(label: int):
    output = np.zeros((10, 1))
    output[label][0] = 1.0

    return output


def convert_image_to_input(image: list):
    return np.reshape(image, (784, 1))


def convert_output_to_label(output: np.ndarray[np.float64]):
    values = np.reshape(output, (10,))

    idx = 0
    max = -100.0
    for i, val in enumerate(values):
        if val > max:
            idx = i
            max = val

    return idx
