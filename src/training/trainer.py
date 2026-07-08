import tensorflow as tf

from config.config import *

from src.training.callbacks import get_callbacks


class Trainer:

    def __init__(self, model):

        self.model = model

    def compile(self):

        self.model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            ),

            loss="binary_crossentropy",

            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()
            ]

        )

    def train(

        self,

        train_dataset,

        validation_dataset

    ):

        history = self.model.fit(

            train_dataset,

            validation_data=validation_dataset,

            epochs=EPOCHS,

            callbacks=get_callbacks()

        )

        return history