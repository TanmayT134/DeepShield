import os
import shutil
import time
from src.utils import tensorflow_setup
import tensorflow as tf
import cv2

from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.image_processor import ImageProcessor

from src.inference.predictor import Predictor
from src.inference.voting import Voting


class VideoPredictor:

    def __init__(self, model_path):

        self.model = tf.keras.models.load_model(model_path)

        self.predictor = Predictor(self.model)

        self.frame_extractor = FrameExtractor(frame_interval=10)

        self.face_detector = FaceDetector()

        self.image_processor = ImageProcessor()

        self.temp_frames = "temp/inference_frames"
        self.temp_faces = "temp/inference_faces"

    def clear_temp(self):

        shutil.rmtree(self.temp_frames, ignore_errors=True)
        shutil.rmtree(self.temp_faces, ignore_errors=True)

        os.makedirs(self.temp_frames, exist_ok=True)
        os.makedirs(self.temp_faces, exist_ok=True)

    def predict_video(self, video_path):

        start = time.time()

        self.clear_temp()

        total_frames = self.frame_extractor.extract_frames(
            video_path,
            self.temp_frames
        )

        total_faces = self.face_detector.detect_faces(
            self.temp_frames,
            self.temp_faces
        )

        predictions = []

        real_count = 0
        fake_count = 0

        if not os.listdir(self.temp_faces):
            raise ValueError("No faces detected in video.")

        for image_name in sorted(os.listdir(self.temp_faces)):

            image_path = os.path.join(
                self.temp_faces,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = self.image_processor.preprocess(image)

            result = self.predictor.predict(image)

            predictions.append(
                (
                    result["label"],
                    result["confidence"]
                )
            )

            if result["label"] == "Real":

                real_count += 1

            else:

                fake_count += 1

        final_prediction = Voting.majority_vote(predictions)
        average_confidence = Voting.average_confidence(predictions)

        end = time.time()

        return {
            "video": os.path.basename(video_path),
            "frames": total_frames,
            "faces": total_faces,
            "real_frames": real_count,
            "fake_frames": fake_count,
            "prediction": final_prediction,
            "confidence": average_confidence,
            "time": end - start
        }