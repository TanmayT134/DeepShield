import tensorflow as tf
import os

os.makedirs("models/best", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/tensorboard", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/final", exist_ok=True)

def get_callbacks():

    return [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=1e-4,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best/best_cnn.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/checkpoints/checkpoint.keras",
            save_best_only=False,
            save_freq="epoch",
            verbose=0
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        tf.keras.callbacks.TensorBoard(
            log_dir="outputs/tensorboard",
            histogram_freq=1
        ),

        tf.keras.callbacks.CSVLogger(
            "outputs/logs/training_log.csv"
        ),
        
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"Learning Rate: {self.model.optimizer.learning_rate.numpy():.8f}"
            )
        )

    ]