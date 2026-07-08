from src.preprocessing.dataset_builder import DatasetBuilder

video = r"dataset/raw/Celeb-DF/Celeb-real/id0_0000.mp4"

output = r"dataset/processed/faces"

builder = DatasetBuilder()

builder.process_video(video, output)