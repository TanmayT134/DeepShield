import os

from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector
from src.utils.file_utils import clear_directory


class DatasetBuilder:

    def __init__(self):

        self.frame_extractor = FrameExtractor(frame_interval=10)

        self.face_detector = FaceDetector()

        self.temp_folder = "temp/frames"

    def process_video(self, video_path, output_folder):

        clear_directory(self.temp_folder)

        frame_count = self.frame_extractor.extract_frames(
            video_path,
            self.temp_folder
        )

        face_count = self.face_detector.detect_faces(
            self.temp_folder,
            output_folder
        )

        return frame_count, face_count