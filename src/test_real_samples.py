# import pandas as pd

# TEST_PATH = "data/processed/test.csv"

# df = pd.read_csv(TEST_PATH)

# print("=" * 60)
# print("DATASET LABEL DIAGNOSTIC")
# print("=" * 60)

# print("\nLabel distribution:")
# print(df["label"].value_counts())

# print("\nLabel examples:")
# print("\n--- HUMAN SAMPLE ---")
# human_sample = df[df["label"] == 0].iloc[0]
# print(human_sample["text"])

# print("\n--- AI SAMPLE ---")
# ai_sample = df[df["label"] == 1].iloc[0]
# print(ai_sample["text"])

# print("\nHuman sample length:", len(str(human_sample["text"])))
# print("AI sample length:", len(str(ai_sample["text"])))


import pandas as pd
import subprocess

TEST_PATH = "data/processed/test.csv"

df = pd.read_csv(TEST_PATH)

# Get one known AI-generated sample
ai_sample = df[df["label"] == 1].iloc[0]["text"]

print("=" * 60)
print("KNOWN AI SAMPLE")
print("=" * 60)

print(ai_sample)

print("\n" + "=" * 60)
print("TRUE LABEL")
print("=" * 60)

print("Label: 1 = AI-generated")
print("Length:", len(str(ai_sample)))

print("\n" + "=" * 60)
print("Now copy the sample above and test it using predict.py")
print("=" * 60)