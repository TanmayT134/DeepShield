import os

# ==================================================
# Raw Dataset
# ==================================================

RAW_DATASET = "dataset/raw/subset"

REAL_PATH = os.path.join(RAW_DATASET, "real")
FAKE_PATH = os.path.join(RAW_DATASET, "fake")

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
EPOCHS = 20

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

UNFREEZE_LAST_LAYERS = 20

# ==================================================
# Random Seed
# ==================================================

SEED = 42