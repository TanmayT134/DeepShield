import cv2
import os
from mtcnn import MTCNN


class FaceDetector:

    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, input_folder, output_folder):

        os.makedirs(output_folder, exist_ok=True)

        saved = 0

        for image_name in os.listdir(input_folder):

            image_path = os.path.join(input_folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detections = self.detector.detect_faces(rgb)

            if len(detections) == 0:
                continue

            x, y, w, h = detections[0]["box"]

            x = max(0, x)
            y = max(0, y)

            face = image[y:y+h, x:x+w]

            cv2.imwrite(
                os.path.join(output_folder, image_name),
                face
            )

            saved += 1

        return saved