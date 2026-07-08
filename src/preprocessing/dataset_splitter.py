import os
import shutil
import random


class DatasetSplitter:

    def __init__(
        self,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42
    ):

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        random.seed(seed)

    def split(self, input_folder, output_folder):

        images = [
            image for image in os.listdir(input_folder)
            if image.endswith(".jpg")
        ]

        random.shuffle(images)

        total = len(images)

        train_end = int(total * self.train_ratio)

        validation_end = train_end + int(total * self.validation_ratio)

        train = images[:train_end]
        validation = images[train_end:validation_end]
        test = images[validation_end:]

        folders = {
            "train": train,
            "validation": validation,
            "test": test
        }

        class_name = os.path.basename(input_folder)

        for split_name, image_list in folders.items():

            destination = os.path.join(
                output_folder,
                split_name,
                class_name
            )

            os.makedirs(destination, exist_ok=True)

            for image in image_list:

                shutil.copy2(
                    os.path.join(input_folder, image),
                    os.path.join(destination, image)
                )

        print(f"{class_name}")
        print(f"Train : {len(train)}")
        print(f"Validation : {len(validation)}")
        print(f"Test : {len(test)}")