from src.datasets.dataloader import create_dataset
from config.config import *

train_dataset = create_dataset(TRAIN_PATH, training=True)

validation_dataset = create_dataset(
    VALIDATION_PATH,
    training=False
)

test_dataset = create_dataset(
    TEST_PATH,
    training=False
)

for images, labels in train_dataset.take(1):
    print(images.shape)
    print(labels.shape)
    print(images.dtype)