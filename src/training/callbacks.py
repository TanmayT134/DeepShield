import tensorflow as tf


def get_callbacks():

    return [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best/best_cnn.keras",
            monitor="val_accuracy",
            save_best_only=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3
        ),

        tf.keras.callbacks.CSVLogger(
            "outputs/logs/training_log.csv"
        )

    ]