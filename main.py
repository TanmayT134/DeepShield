import os
import tensorflow as tf

# ==================================================
# TensorFlow Setup
# ==================================================

from src.utils.tensorflow_setup import setup_tensorflow

setup_tensorflow()

# ==================================================
# Configuration
# ==================================================

from config.config import *

# ==================================================
# Dataset
# ==================================================

from src.datasets.dataloader import create_dataset
from src.utils.dataset_statistics import print_dataset_statistics

print("\nLoading datasets...\n")

train_dataset = create_dataset(
    TRAIN_PATH,
    training=True
)

validation_dataset = create_dataset(
    VALIDATION_PATH,
    training=False
)

test_dataset = create_dataset(
    TEST_PATH,
    training=False
)

print("\nDataset Statistics\n")

print_dataset_statistics(
    train_dataset,
    "Train"
)

print_dataset_statistics(
    validation_dataset,
    "Validation"
)

print_dataset_statistics(
    test_dataset,
    "Test"
)

# ==================================================
# Model
# ==================================================

from src.models.cnn import DeepShieldCNN
from src.utils.save_model_summary import save_model_summary

CHECKPOINT_PATH = "models/checkpoints/checkpoint.keras"

if os.path.exists(CHECKPOINT_PATH):

    print("\nResuming from latest checkpoint...\n")

    model = tf.keras.models.load_model(
        CHECKPOINT_PATH
    )

elif os.path.exists(MODEL_PATH):

    print("\nLoading best model...\n")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

else:

    print("\nBuilding new model...\n")

    cnn = DeepShieldCNN()

    model = cnn.build()

save_model_summary(model)

# ==================================================
# Training
# ==================================================

from src.training.trainer import Trainer

trainer = Trainer(model)

trainer.compile()

history = trainer.train(

    train_dataset,

    validation_dataset

)

# ==================================================
# Save Training History
# ==================================================

from src.utils.save_history import save_history

save_history(history)

# ==================================================
# Plot Training Curves
# ==================================================

from src.utils.history_plotter import HistoryPlotter

plotter = HistoryPlotter(history)

plotter.plot()

# ==================================================
# Evaluation
# ==================================================

from src.evaluation.evaluate import Evaluator

evaluator = Evaluator(model)

evaluator.evaluate(test_dataset)

# ==================================================
# Save Final Model
# ==================================================

os.makedirs(
    "models/final",
    exist_ok=True
)

FINAL_MODEL_PATH = os.path.join(
    "models",
    "final",
    "final_cnn.keras"
)

model.save(
    FINAL_MODEL_PATH
)

print("\nFinal model saved successfully.")

print(f"Location : {FINAL_MODEL_PATH}")

print("\nDeepShield Pipeline Completed Successfully.")