import os
import cv2

from mtcnn import MTCNN

from src.preprocessing.image_processor import ImageProcessor


class FaceDetector:

    def __init__(self):

        self.detector = MTCNN()

        self.processor = ImageProcessor()

    def detect_faces(self, input_folder, output_folder):

        os.makedirs(output_folder, exist_ok=True)

        saved = 0

        for image_name in sorted(os.listdir(input_folder)):

            image_path = os.path.join(
                input_folder,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            try:

                detections = self.detector.detect_faces(rgb)

            except Exception as e:

                print(f"Skipping {image_name}: {e}")

                continue

            if len(detections) == 0:
                continue

            # Select the face with highest confidence
            best_face = max(
                detections,
                key=lambda face: face["confidence"]
            )

            confidence = best_face["confidence"]

            if confidence < 0.95:
                continue

            x, y, w, h = best_face["box"]

            margin = int(
                max(w, h) * 0.20
            )

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)

            x2 = min(
                image.shape[1],
                x + w + margin
            )

            y2 = min(
                image.shape[0],
                y + h + margin
            )

            face = image[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            face = self.processor.preprocess(face)

            face = (
                face * 255
            ).astype("uint8")

            cv2.imwrite(
                os.path.join(
                    output_folder,
                    image_name
                ),
                face
            )

            saved += 1

        return saved