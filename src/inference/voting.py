from collections import Counter


class Voting:

    @staticmethod
    def majority_vote(predictions):

        labels = [
            label
            for label, _ in predictions
        ]

        return Counter(labels).most_common(1)[0][0]

    @staticmethod
    def average_confidence(predictions):

        confidence = sum(
            score
            for _, score in predictions
        )

        return confidence / len(predictions)