import os
import matplotlib.pyplot as plt


class HistoryPlotter:

    def __init__(self, history):

        self.history = history.history

        os.makedirs(
            "outputs/plots",
            exist_ok=True
        )

    def plot(self):

        self.plot_metric(
            "accuracy",
            "Accuracy"
        )

        self.plot_metric(
            "loss",
            "Loss"
        )

        self.plot_metric(
            "auc",
            "AUC"
        )

        self.plot_metric(
            "precision",
            "Precision"
        )

        self.plot_metric(
            "recall",
            "Recall"
        )

        self.plot_learning_rate()

    def plot_metric(self, metric, title):

        if metric not in self.history:
            return

        plt.figure(figsize=(7,5))

        plt.plot(
            self.history[metric],
            label=f"Train {title}"
        )

        val_metric = "val_" + metric

        if val_metric in self.history:

            plt.plot(
                self.history[val_metric],
                label=f"Validation {title}"
            )

        plt.title(title)

        plt.xlabel("Epoch")

        plt.ylabel(title)

        plt.grid(True)

        plt.legend()

        plt.savefig(
            f"outputs/plots/{metric}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def plot_learning_rate(self):

        if "learning_rate" not in self.history:
            return

        plt.figure(figsize=(7,5))

        plt.plot(
            self.history["learning_rate"]
        )

        plt.title("Learning Rate")

        plt.xlabel("Epoch")

        plt.ylabel("Learning Rate")

        plt.grid(True)

        plt.savefig(
            "outputs/plots/learning_rate.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()