import cv2
import os


class FrameExtractor:

    def __init__(self, frame_interval=10):
        self.frame_interval = frame_interval

    def extract_frames(self, video_path, output_folder):

        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            if frame_count % self.frame_interval == 0:

                filename = f"frame_{saved_count:04d}.jpg"

                cv2.imwrite(
                    os.path.join(output_folder, filename),
                    frame
                )

                saved_count += 1

            frame_count += 1

        cap.release()

        return saved_count