from datasets import load_dataset
import pandas as pd
import os


print("Loading HC3 dataset...")

dataset = load_dataset(
    "Hello-SimpleAI/HC3",
    revision="refs/convert/parquet",
    split="train"
)

print(f"Loaded {len(dataset)} source records.")


records = []

for row in dataset:
    source = row["source"]

    # Human-written answers
    for answer in row["human_answers"]:
        if answer and answer.strip():
            records.append({
                "text": answer.strip(),
                "label": 0,
                "source": source
            })

    # ChatGPT-generated answers
    for answer in row["chatgpt_answers"]:
        if answer and answer.strip():
            records.append({
                "text": answer.strip(),
                "label": 1,
                "source": source
            })


df = pd.DataFrame(records)

print(f"\nTotal text samples: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nRemoving duplicate texts...")

before = len(df)

df = df.drop_duplicates(subset=["text"])
df = df.reset_index(drop=True)

print(f"Removed {before - len(df)} duplicates.")
print(f"Remaining samples: {len(df)}")


# Create output directory
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "ai_text_dataset.csv")

df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\nDataset saved to: {output_path}")

print("\nFinal class distribution:")
print(df["label"].value_counts())

print("\nSample records:")
print(df.head())