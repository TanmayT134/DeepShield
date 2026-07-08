from src.preprocessing.dataset_splitter import DatasetSplitter

splitter = DatasetSplitter()

splitter.split(
    "dataset/processed/real",
    "dataset/processed"
)

splitter.split(
    "dataset/processed/fake",
    "dataset/processed"
)

print("\nDataset Ready For Training.")