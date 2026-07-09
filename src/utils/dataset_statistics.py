import tensorflow as tf


def print_dataset_statistics(dataset, name):

    samples = 0

    for images, _ in dataset:
        samples += images.shape[0]

    print(f"{name}: {samples} images")