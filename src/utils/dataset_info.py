import os

# Dataset Path
DATASET_PATH = "dataset/raw/Celeb-DF"

folders = [
    "Celeb-real",
    "Celeb-synthesis",
    "YouTube-real"
]

print("=" * 50)
print("Celeb-DF Dataset Information")
print("=" * 50)

total = 0

for folder in folders:

    folder_path = os.path.join(DATASET_PATH, folder)

    videos = [
        file for file in os.listdir(folder_path)
        if file.endswith(".mp4")
    ]

    count = len(videos)

    total += count

    print(f"{folder:20} : {count} videos")

print("-" * 50)
print(f"Total Videos       : {total}")
print("=" * 50)