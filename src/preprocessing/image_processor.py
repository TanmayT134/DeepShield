import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


class ImageProcessor:

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def preprocess(self, image):

        image = cv2.resize(
            image,
            self.image_size
        )

        image = cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        l = self.clahe.apply(l)

        lab = cv2.merge((l, a, b))

        image = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

        image = image.astype(np.float32)

        image = preprocess_input(image)

        return image