import os
import time

from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector
from src.utils.file_utils import clear_directory
from src.utils.logger import logger


class DatasetBuilder:

    def __init__(self):

        self.frame_extractor = FrameExtractor(
            frame_interval=10
        )

        self.face_detector = FaceDetector()

        self.temp_folder = "temp/frames"

    def process_video(
        self,
        video_path,
        output_folder
    ):

        start = time.time()

        # ---------------------------------------
        # Clear temporary frames
        # ---------------------------------------

        clear_directory(
            self.temp_folder
        )

        # ---------------------------------------
        # Extract Frames
        # ---------------------------------------

        frame_count = self.frame_extractor.extract_frames(
            video_path,
            self.temp_folder
        )

        if frame_count == 0:

            raise RuntimeError(
                "No frames extracted."
            )

        # ---------------------------------------
        # Detect Faces
        # ---------------------------------------

        face_count = self.face_detector.detect_faces(
            self.temp_folder,
            output_folder
        )

        if face_count == 0:

            logger.warning(
                f"No faces detected in "
                f"{os.path.basename(video_path)}"
            )

        # ---------------------------------------
        # Cleanup
        # ---------------------------------------

        clear_directory(
            self.temp_folder
        )

        elapsed = time.time() - start

        logger.info(
            f"{os.path.basename(video_path)} "
            f"completed in "
            f"{elapsed:.2f}s"
        )

        return frame_count, face_count