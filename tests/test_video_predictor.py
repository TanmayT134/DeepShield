from src.inference.video_predictor import VideoPredictor
from config.config import MODEL_PATH

MODEL_PATH = "models/best/best_cnn.keras"
VIDEO_PATH = "dataset/raw/subset/fake/id0_id16_0000.mp4"

predictor = VideoPredictor(MODEL_PATH)

result = predictor.predict_video(VIDEO_PATH)

print("\n==============================")
print(" DeepShield Analysis")
print("==============================")

print("Video            :", result["video"])
print("Frames Extracted :", result["frames"])
print("Faces Detected   :", result["faces"])
print("Real Frames      :", result["real_frames"])
print("Fake Frames      :", result["fake_frames"])
print("Prediction       :", result["prediction"])
print("Confidence       : {:.2f}%".format(result["confidence"] * 100))
print("Inference Time   : {:.2f} sec".format(result["time"]))

print("==============================")