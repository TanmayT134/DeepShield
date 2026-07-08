import cv2
import os


class VideoLoader:

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

    def get_video_info(self):

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        return {
            "FPS": fps,
            "Width": width,
            "Height": height,
            "Total Frames": total_frames,
            "Duration (sec)": round(duration, 2)
        }

    def release(self):
        self.cap.release()