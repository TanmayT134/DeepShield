import os
import shutil
import random

from config.config import *

random.seed(SEED)


# ==================================================
# Clean Previous Dataset
# ==================================================

if os.path.exists(CNN_DATASET):
    shutil.rmtree(CNN_DATASET)

os.makedirs(CNN_DATASET, exist_ok=True)


# ==================================================
# Get Videos
# ==================================================

def collect_videos(folder):

    videos = []

    for video in sorted(os.listdir(folder)):

        video_path = os.path.join(folder, video)

        if os.path.isdir(video_path):
            videos.append(video_path)

    return videos


# ==================================================
# Split Videos
# ==================================================

def split_videos(videos):

    random.shuffle(videos)

    n = len(videos)

    train = videos[:int(TRAIN_RATIO * n)]

    validation = videos[
        int(TRAIN_RATIO * n):
        int((TRAIN_RATIO + VALID_RATIO) * n)
    ]

    test = videos[
        int((TRAIN_RATIO + VALID_RATIO) * n):
    ]

    return train, validation, test


# ==================================================
# Copy Frames
# ==================================================

def copy_frames(video_list, destination, label):

    destination = os.path.join(
        destination,
        label
    )

    os.makedirs(destination, exist_ok=True)

    total_images = 0

    for video in video_list:

        video_name = os.path.basename(video)

        images = [

            image

            for image in sorted(os.listdir(video))

            if image.lower().endswith(".jpg")

        ]

        random.shuffle(images)

        images = images[:MAX_IMAGES_PER_VIDEO]

        for image in images:

            source = os.path.join(
                video,
                image
            )

            destination_name = f"{video_name}_{image}"

            shutil.copy(
                source,
                os.path.join(
                    destination,
                    destination_name
                )
            )

            total_images += 1

    return total_images


# ==================================================
# Create Dataset
# ==================================================

def create_dataset(label):

    source = os.path.join(
        PROCESSED_PATH,
        label
    )

    videos = collect_videos(source)

    print(f"\n{label.upper()}")

    print(f"Videos : {len(videos)}")

    train, validation, test = split_videos(videos)

    train_images = copy_frames(
        train,
        os.path.join(CNN_DATASET, "train"),
        label
    )

    validation_images = copy_frames(
        validation,
        os.path.join(CNN_DATASET, "validation"),
        label
    )

    test_images = copy_frames(
        test,
        os.path.join(CNN_DATASET, "test"),
        label
    )

    print(f"Train Videos      : {len(train)}")
    print(f"Validation Videos : {len(validation)}")
    print(f"Test Videos       : {len(test)}")

    print()

    print(f"Train Images      : {train_images}")
    print(f"Validation Images : {validation_images}")
    print(f"Test Images       : {test_images}")


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    create_dataset("real")

    create_dataset("fake")

    print("\nCNN Dataset Created Successfully.")