import os

# ==================================================
# Raw Dataset
# ==================================================

RAW_DATASET = "dataset/raw/Celeb-DF"

REAL_PATH = os.path.join(RAW_DATASET, "Celeb-real")
FAKE_PATH = os.path.join(RAW_DATASET, "Celeb-synthesis")

# ==================================================
# Processed Dataset (Video-wise)
# ==================================================

PROCESSED_PATH = "dataset/processed"

REAL_OUTPUT = os.path.join(PROCESSED_PATH, "real")
FAKE_OUTPUT = os.path.join(PROCESSED_PATH, "fake")

# ==================================================
# CNN Dataset
# ==================================================

CNN_DATASET = "dataset/cnn_dataset"

TRAIN_PATH = os.path.join(CNN_DATASET, "train")
VALIDATION_PATH = os.path.join(CNN_DATASET, "validation")
TEST_PATH = os.path.join(CNN_DATASET, "test")

# ==================================================
# Image
# ==================================================

IMAGE_SIZE = (224, 224)
CHANNELS = 3

# ==================================================
# Training
# ==================================================

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 1e-5

NUM_CLASSES = 2

# ==================================================
# Model
# ==================================================

MODEL_NAME = "EfficientNetB0"

MODEL_PATH = "models/best/best_cnn.keras"

PREDICTION_THRESHOLD = 0.5

# ==================================================
# Fine Tuning
# ==================================================

FREEZE_BACKBONE = False

UNFREEZE_LAST_LAYERS = 40

# ==================================================
# Random Seed
# ==================================================

SEED = 42

# ==================================================
# Model Architecture
# ==================================================

DENSE_UNITS = 256
SECOND_DENSE_UNITS = 128

DROPOUT_1 = 0.40
DROPOUT_2 = 0.30

L2_REGULARIZATION = 1e-4

# ==================================================
# Classes
# ==================================================

CLASS_NAMES = [
    "Real",
    "Fake"
]

# ==================================================
# Dataset Creation
# ==================================================

MAX_IMAGES_PER_VIDEO = 30

# ==================================================
# Training Configuration
# ==================================================

FACE_CONFIDENCE_THRESHOLD = 0.90

FRAME_INTERVAL = 10

USE_CLASS_WEIGHTS = True

# ==================================================
# Dataset Split
# ==================================================

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15