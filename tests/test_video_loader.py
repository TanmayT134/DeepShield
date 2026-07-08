from src.preprocessing.video_loader import VideoLoader

video_path = r"dataset/raw/Celeb-DF/Celeb-real/id0_0000.mp4"

video = VideoLoader(video_path)

info = video.get_video_info()

print("\nVideo Information\n")

for key, value in info.items():
    print(f"{key}: {value}")

video.release()