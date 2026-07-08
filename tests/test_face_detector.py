import os
from src.preprocessing.face_detector import FaceDetector

input_folder = r"dataset/processed/sample_frames"

output_folder = r"dataset/processed/sample_faces"

detector = FaceDetector()

count = 0

for image in os.listdir(input_folder):

    image_path = os.path.join(input_folder, image)

    if detector.detect_face(image_path, output_folder):
        count += 1

print(f"Faces Saved : {count}")