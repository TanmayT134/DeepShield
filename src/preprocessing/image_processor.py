import cv2
import numpy as np


class ImageProcessor:

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def preprocess(self, image_path):

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, self.image_size)

        image = image.astype(np.float32) / 255.0

        return image