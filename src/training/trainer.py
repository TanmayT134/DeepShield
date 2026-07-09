import tensorflow as tf

from config.config import *

from src.training.callbacks import get_callbacks

from src.training.class_weights import get_class_weights


class Trainer:

    def __init__(self, model):

        self.model = model

    def compile(self):

        self.model.compile(

            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=LEARNING_RATE,
                weight_decay=L2_REGULARIZATION
            ),

            loss="binary_crossentropy",

            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]

        )

    def train(

        self,

        train_dataset,

        validation_dataset

    ):

        class_weights = get_class_weights(
            train_dataset
        )
        
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS,
            callbacks=get_callbacks(),
            class_weight=class_weights,
            verbose=1
        )

        return history