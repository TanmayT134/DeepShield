import tensorflow as tf
import numpy as np

from config.config import *


class Predictor:

    def __init__(self, model):

        self.model = model

    def predict(self, image):

        image = np.expand_dims(image, axis=0)

        probability = self.model.predict(
            image,
            verbose=0
        )[0][0]

        label = (
            "Fake"
            if probability >= PREDICTION_THRESHOLD
            else "Real"
        )

        confidence = (
            probability
            if label == "Fake"
            else 1 - probability
        )

        return label, float(confidence)