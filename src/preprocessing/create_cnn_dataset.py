import os
import shutil
import random

random.seed(42)

from config.config import *

SOURCE = PROCESSED_PATH
DESTINATION = CNN_DATASET

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


def collect_images(folder):

    images = []

    for video in os.listdir(folder):

        video_path = os.path.join(folder, video)

        if not os.path.isdir(video_path):
            continue

        for image in os.listdir(video_path):

            if image.lower().endswith(".jpg"):

                images.append(
                    os.path.join(video_path, image)
                )

    return images


def split_dataset(images):

    random.shuffle(images)

    n = len(images)

    train = images[:int(0.70*n)]

    valid = images[int(0.70*n):int(0.85*n)]

    test = images[int(0.85*n):]

    return train, valid, test


def copy_images(image_list, destination):

    os.makedirs(destination, exist_ok=True)

    for image in image_list:

        # Parent folder name (video name)
        video_name = os.path.basename(
            os.path.dirname(image)
        )

        # Original image name
        image_name = os.path.basename(image)

        # Unique filename
        new_name = f"{video_name}_{image_name}"

        destination_file = os.path.join(
            destination,
            new_name
        )

        shutil.copy(
            image,
            destination_file
        )


for label in ["real", "fake"]:

    source = os.path.join(SOURCE, label)

    images = collect_images(source)

    train, valid, test = split_dataset(images)

    copy_images(
        train,
        os.path.join(
            DESTINATION,
            "train",
            label
        )
    )

    copy_images(
        valid,
        os.path.join(
            DESTINATION,
            "validation",
            label
        )
    )

    copy_images(
        test,
        os.path.join(
            DESTINATION,
            "test",
            label
        )
    )

    print(f"{label}")

    print(f"Train : {len(train)}")

    print(f"Validation : {len(valid)}")

    print(f"Test : {len(test)}")