import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)


class Evaluator:

    def __init__(self, model):
        self.model = model

    def evaluate(self, test_dataset):

        print("\nEvaluating Model...\n")

        loss, accuracy, precision, recall, auc_score = \
            self.model.evaluate(test_dataset, verbose=1)

        print(f"\nLoss      : {loss:.4f}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"AUC       : {auc_score:.4f}")

        # ---------------------------------------------------
        # Save metrics
        # ---------------------------------------------------

        os.makedirs("outputs/reports", exist_ok=True)

        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc_score)
        }

        with open(
            "outputs/reports/metrics.json",
            "w"
        ) as file:

            json.dump(metrics, file, indent=4)

        # ---------------------------------------------------
        # Predictions
        # ---------------------------------------------------

        y_true = []
        y_pred = []
        y_prob = []

        for images, labels in test_dataset:

            probabilities = self.model.predict(
                images,
                verbose=0
            )

            predictions = (
                probabilities > 0.5
            ).astype(int)

            y_true.extend(
                labels.numpy().flatten()
            )

            y_pred.extend(
                predictions.flatten()
            )

            y_prob.extend(
                probabilities.flatten()
            )

        # ---------------------------------------------------
        # Classification Report
        # ---------------------------------------------------

        report = classification_report(
            y_true,
            y_pred,
            target_names=[
                "Real",
                "Fake"
            ]
        )

        print("\nClassification Report\n")

        print(report)

        with open(
            "outputs/reports/classification_report.txt",
            "w"
        ) as file:

            file.write(report)

        # ---------------------------------------------------
        # Plots
        # ---------------------------------------------------

        os.makedirs(
            "outputs/plots",
            exist_ok=True
        )

        self.plot_confusion_matrix(
            y_true,
            y_pred
        )

        self.plot_roc(
            y_true,
            y_prob
        )

        self.plot_precision_recall(
            y_true,
            y_prob
        )

        print("\nEvaluation files saved successfully.")

    # =====================================================

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Real",
                "Fake"
            ]
        )

        disp.plot(
            cmap="Blues",
            values_format="d"
        )

        plt.title("Confusion Matrix")

        plt.savefig(
            "outputs/plots/confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =====================================================

    def plot_roc(
        self,
        y_true,
        y_prob
    ):

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.figure(
            figsize=(6, 6)
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"AUC = {roc_auc:.3f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            "--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve"
        )

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "outputs/plots/roc_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =====================================================

    def plot_precision_recall(
        self,
        y_true,
        y_prob
    ):

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_prob
        )

        plt.figure(
            figsize=(6, 6)
        )

        plt.plot(
            recall,
            precision,
            linewidth=2
        )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title(
            "Precision-Recall Curve"
        )

        plt.grid(True)

        plt.savefig(
            "outputs/plots/precision_recall_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()