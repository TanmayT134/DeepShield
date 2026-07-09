import numpy as np

from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(train_dataset):

    labels = []

    for _, batch_labels in train_dataset:

        labels.extend(
            batch_labels.numpy().astype(int).flatten()
        )

    labels = np.array(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    class_weights = {
        i: float(weight)
        for i, weight in enumerate(weights)
    }

    print("\n==============================")
    print("Class Weights")
    print("==============================")

    for cls, weight in class_weights.items():
        print(f"Class {cls}: {weight:.4f}")

    print("==============================\n")

    return class_weights