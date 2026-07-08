from src.preprocessing.frame_extractor import FrameExtractor

video_path = r"dataset/raw/Celeb-DF/Celeb-real/id0_0000.mp4"

output_folder = r"dataset/processed/sample_frames"

extractor = FrameExtractor(output_folder)

extractor.extract_frames(
    video_path=video_path,
    frame_interval=10
)