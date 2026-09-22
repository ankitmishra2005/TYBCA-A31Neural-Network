from datasets import load_dataset

print("Loading HC3 dataset...")

dataset = load_dataset(
    "Hello-SimpleAI/HC3",
    revision="refs/convert/parquet",
    split="train"
)

print("\nDataset loaded successfully!")
print(dataset)

print("\nNumber of rows:", len(dataset))
print("\nColumns:", dataset.column_names)

print("\nFirst example:")
print(dataset[0])