import tensorflow as tf

from config.config import *

from src.datasets.augmentations import (
    preprocess_train,
    preprocess_test
)

AUTOTUNE = tf.data.AUTOTUNE


def create_dataset(path, training=True):

    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=training
    )

    if training:
        dataset = dataset.map(
            preprocess_train,
            num_parallel_calls=AUTOTUNE
        )
    else:
        dataset = dataset.map(
            preprocess_test,
            num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset