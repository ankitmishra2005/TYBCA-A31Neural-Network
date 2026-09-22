import pandas as pd
from sklearn.model_selection import train_test_split
import os


INPUT_PATH = "data/processed/ai_text_dataset.csv"
OUTPUT_DIR = "data/processed"


print("Loading dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Total samples: {len(df)}")


# First split:
# 70% training
# 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)


# Second split:
# 15% validation
# 15% test
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


train_path = os.path.join(OUTPUT_DIR, "train.csv")
validation_path = os.path.join(OUTPUT_DIR, "validation.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")


train_df.to_csv(train_path, index=False, encoding="utf-8")
validation_df.to_csv(validation_path, index=False, encoding="utf-8")
test_df.to_csv(test_path, index=False, encoding="utf-8")


print("\nDataset split completed!")

print(f"\nTraining samples:   {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print(f"Test samples:       {len(test_df)}")


print("\nTraining distribution:")
print(train_df["label"].value_counts(normalize=True))

print("\nValidation distribution:")
print(validation_df["label"].value_counts(normalize=True))

print("\nTest distribution:")
print(test_df["label"].value_counts(normalize=True))


print("\nFiles created:")
print(train_path)
print(validation_path)
print(test_path)