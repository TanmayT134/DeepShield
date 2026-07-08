import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


class Evaluator:

    def __init__(self, model):
        self.model = model

    def evaluate(self, test_dataset):

        print("\nEvaluating Model...\n")

        loss, accuracy, precision, recall, auc_score = \
            self.model.evaluate(test_dataset)

        print(f"\nLoss      : {loss:.4f}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"AUC       : {auc_score:.4f}")

        y_true = []
        y_pred = []
        y_prob = []

        for images, labels in test_dataset:

            probs = self.model.predict(images, verbose=0)

            preds = (probs > 0.5).astype(int)

            y_true.extend(labels.numpy().flatten())
            y_pred.extend(preds.flatten())
            y_prob.extend(probs.flatten())

        print("\nClassification Report\n")

        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["Real", "Fake"]
            )
        )

        self.plot_confusion_matrix(y_true, y_pred)

        self.plot_roc(y_true, y_prob)

    def plot_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Real", "Fake"]
        )

        disp.plot(cmap="Blues")

        os.makedirs("outputs/plots", exist_ok=True)

        plt.savefig(
            "outputs/plots/confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def plot_roc(self, y_true, y_prob):

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.3f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            "--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "outputs/plots/roc_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()