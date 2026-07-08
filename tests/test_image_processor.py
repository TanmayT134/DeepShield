from src.preprocessing.image_processor import ImageProcessor

image_path = r"dataset/processed/sample_faces/frame_0000.jpg"

processor = ImageProcessor()

image = processor.preprocess(image_path)

print("Image Shape :", image.shape)
print("Data Type   :", image.dtype)
print("Minimum     :", image.min())
print("Maximum     :", image.max())