import pandas as pd
import os


INPUT_PATH = "data/processed/train.csv"
OUTPUT_PATH = "data/processed/transformer_train.csv"

HUMAN_SAMPLES = 10000
AI_SAMPLES = 10000


print("Loading training dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Original training samples: {len(df)}")


# Separate classes
human_df = df[df["label"] == 0]
ai_df = df[df["label"] == 1]


print(f"Available human samples: {len(human_df)}")
print(f"Available AI samples: {len(ai_df)}")


# Sample equal number from both classes
human_sample = human_df.sample(
    n=HUMAN_SAMPLES,
    random_state=42
)

ai_sample = ai_df.sample(
    n=AI_SAMPLES,
    random_state=42
)


# Combine samples
transformer_df = pd.concat(
    [human_sample, ai_sample],
    ignore_index=True
)


# Shuffle dataset
transformer_df = transformer_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Save
os.makedirs("data/processed", exist_ok=True)

transformer_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


print("\nTransformer training dataset created!")

print(f"Total samples: {len(transformer_df)}")

print("\nClass distribution:")
print(transformer_df["label"].value_counts())

print(f"\nSaved to: {OUTPUT_PATH}")