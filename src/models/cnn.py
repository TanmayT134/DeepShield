import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout,
    Dense
)

from tensorflow.keras.applications import EfficientNetB0

from config.config import *


class DeepShieldCNN:

    def __init__(self):

        self.base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                CHANNELS
            )
        )

        self.base_model.trainable = not FREEZE_BACKBONE

        if not FREEZE_BACKBONE:

            for layer in self.base_model.layers[:-UNFREEZE_LAST_LAYERS]:
                layer.trainable = False

    def build(self):

        inputs = Input(
            shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                CHANNELS
            )
        )

        x = self.base_model(inputs, training=False)

        x = GlobalAveragePooling2D()(x)

        x = BatchNormalization()(x)

        x = Dropout(0.30)(x)

        x = Dense(
            DENSE_UNITS,
            activation="relu",
            kernel_regularizer=l2(L2_REGULARIZATION)
        )(x)

        x = BatchNormalization()(x)

        x = Dropout(DROPOUT_1)(x)

        x = Dense(
            SECOND_DENSE_UNITS,
            activation="relu",
            kernel_regularizer=l2(L2_REGULARIZATION)
        )(x)

        x = Dropout(DROPOUT_2)(x)

        outputs = Dense(
            1,
            activation="sigmoid"
        )(x)

        model = Model(
            inputs,
            outputs,
            name="DeepShieldCNN"
        )

        return model

    def unfreeze_last_layers(self, num_layers=20):

        self.base_model.trainable = True

        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False