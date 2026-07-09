import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

train_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.05),

    tf.keras.layers.RandomZoom(0.10),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    tf.keras.layers.RandomContrast(0.15),

    tf.keras.layers.RandomBrightness(
        factor=0.15
    ),

    tf.keras.layers.GaussianNoise(
        0.02
    )

])

test_augmentation = tf.keras.Sequential([])


def preprocess_train(image, label):

    image = train_augmentation(image)

    image = preprocess_input(image)

    return image, label


def preprocess_test(image, label):

    image = preprocess_input(image)

    return image, label