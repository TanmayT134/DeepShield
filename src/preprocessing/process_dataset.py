import os

from config.config import *
from src.preprocessing.dataset_builder import DatasetBuilder

builder = DatasetBuilder()


def process_folder(input_folder, output_root):

    videos = sorted([
        video for video in os.listdir(input_folder)
        if video.endswith(".mp4")
    ])

    print(f"\nProcessing {len(videos)} videos...\n")

    for index, video in enumerate(videos, start=1):

        video_name = os.path.splitext(video)[0]

        output_folder = os.path.join(
            output_root,
            video_name
        )

        video_path = os.path.join(input_folder, video)

        print(f"[{index}/{len(videos)}] {video_name}")

        frames, faces = builder.process_video(
            video_path,
            output_folder
        )

        print(f"Frames : {frames}")
        print(f"Faces  : {faces}\n")


process_folder(REAL_PATH, REAL_OUTPUT)
process_folder(FAKE_PATH, FAKE_OUTPUT)

print("\nDataset Processing Completed.")