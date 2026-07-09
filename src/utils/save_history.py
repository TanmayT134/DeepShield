import json
import os


def save_history(history):

    os.makedirs("outputs/reports", exist_ok=True)

    with open(
        "outputs/reports/history.json",
        "w"
    ) as file:

        json.dump(
            history.history,
            file,
            indent=4
        )