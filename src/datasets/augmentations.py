import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Training augmentation
train_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
])

# Validation/Test augmentation
test_augmentation = tf.keras.Sequential([])


def preprocess_train(image, label):
    image = train_augmentation(image)
    image = preprocess_input(image)
    return image, label


def preprocess_test(image, label):
    image = preprocess_input(image)
    return image, label