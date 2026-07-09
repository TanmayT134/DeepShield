import os


def save_model_summary(model):

    os.makedirs("outputs/reports", exist_ok=True)

    with open(
        "outputs/reports/model_summary.txt",
        "w"
    ) as file:

        model.summary(
            print_fn=lambda x: file.write(x + "\n")
        )