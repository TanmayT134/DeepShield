import numpy as np

from config.config import *


class Predictor:

    def __init__(self, model):

        self.model = model

    def predict(self, image):

        image = np.expand_dims(
            image,
            axis=0
        )

        probability = float(

            self.model.predict(
                image,
                verbose=0
            )[0][0]

        )

        if probability >= PREDICTION_THRESHOLD:

            label = "Fake"

            confidence = probability

        else:

            label = "Real"

            confidence = 1.0 - probability

        return {

            "label": label,

            "confidence": round(
                confidence,
                4
            ),

            "probability": round(
                probability,
                4
            )

        }