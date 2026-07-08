from src.utils import tensorflow_setup
from src.models.cnn import DeepShieldCNN
from src.training.trainer import Trainer

from src.datasets.dataloader import create_dataset

from config.config import *


train_dataset = create_dataset(
    TRAIN_PATH,
    training=True
)

validation_dataset = create_dataset(
    VALIDATION_PATH,
    training=False
)


import tensorflow as tf
import os

MODEL_PATH = "models/best/best_cnn.keras"

if os.path.exists(MODEL_PATH):

    print("Loading Best Model...")

    model = tf.keras.models.load_model(MODEL_PATH)

else:

    cnn = DeepShieldCNN()

    model = cnn.build()


trainer = Trainer(model)

trainer.compile()

history = trainer.train(
    train_dataset,
    validation_dataset
)

from src.evaluation.evaluate import Evaluator

test_dataset = create_dataset(
    TEST_PATH,
    training=False
)

evaluator = Evaluator(model)

evaluator.evaluate(test_dataset)