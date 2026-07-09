import os
import time

from tqdm import tqdm

from config.config import *
from src.preprocessing.dataset_builder import DatasetBuilder
from src.utils.logger import logger

builder = DatasetBuilder()


def process_folder(input_folder, output_root):

    videos = sorted([
        video for video in os.listdir(input_folder)
        if video.endswith(".mp4")
    ])

    logger.info(f"Processing {len(videos)} videos from {input_folder}")

    processed_videos = 0
    skipped_videos = 0
    failed_videos = 0

    total_frames = 0
    total_faces = 0

    total_time = 0

    for video in tqdm(videos, desc=os.path.basename(input_folder)):

        video_name = os.path.splitext(video)[0]

        video_path = os.path.join(
            input_folder,
            video
        )

        output_folder = os.path.join(
            output_root,
            video_name
        )

        # Skip already processed videos
        if os.path.exists(output_folder):

            existing = [
                file
                for file in os.listdir(output_folder)
                if file.lower().endswith(".jpg")
            ]

            if len(existing) > 0:

                skipped_videos += 1

                logger.info(f"Skipping {video_name}")

                continue

        start = time.time()

        try:

            frames, faces = builder.process_video(
                video_path,
                output_folder
            )

            elapsed = time.time() - start

            processed_videos += 1

            total_frames += frames
            total_faces += faces
            total_time += elapsed

            logger.info(
                f"{video_name} | "
                f"Frames={frames} | "
                f"Faces={faces} | "
                f"Time={elapsed:.2f}s"
            )

        except Exception as e:

            failed_videos += 1

            logger.error(
                f"{video_name} failed : {e}"
            )

    logger.info("=" * 60)
    logger.info(f"Summary : {os.path.basename(input_folder)}")
    logger.info("=" * 60)

    logger.info(f"Videos Found      : {len(videos)}")
    logger.info(f"Processed Videos  : {processed_videos}")
    logger.info(f"Skipped Videos    : {skipped_videos}")
    logger.info(f"Failed Videos     : {failed_videos}")

    logger.info(f"Frames Extracted  : {total_frames}")
    logger.info(f"Faces Saved       : {total_faces}")

    logger.info(f"Processing Time   : {total_time:.2f} sec")

    logger.info("=" * 60)


if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("Starting Dataset Processing")
    logger.info("=" * 60)

    process_folder(
        REAL_PATH,
        REAL_OUTPUT
    )

    process_folder(
        FAKE_PATH,
        FAKE_OUTPUT
    )

    logger.info("=" * 60)
    logger.info("Dataset Processing Completed Successfully")
    logger.info("=" * 60)