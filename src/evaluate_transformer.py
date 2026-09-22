import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==============================
# Configuration
# ==============================

MODEL_PATH = "models/distilbert_ai_detector"
TEST_PATH = "data/processed/test.csv"

MAX_LENGTH = 128
BATCH_SIZE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DISTILBERT AI TEXT DETECTOR - TEST EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")

# ==============================
# Load Test Dataset
# ==============================

df = pd.read_csv(TEST_PATH)

print(f"Test samples: {len(df)}")

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()


# ==============================
# Dataset Class
# ==============================

class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

        return item


# ==============================
# Load Tokenizer
# ==============================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# ==============================
# Create Dataset & DataLoader
# ==============================

test_dataset = TextDataset(
    texts,
    labels,
    tokenizer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==============================
# Load Trained Model
# ==============================

print("Loading trained DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()


# ==============================
# Prediction
# ==============================

all_predictions = []
all_labels = []

print("\nRunning inference on test set...")

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        batch_labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            batch_labels.cpu().numpy()
        )


# ==============================
# Calculate Metrics
# ==============================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


# ==============================
# Display Results
# ==============================

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Human-written",
            "AI-generated"
        ],
        digits=4
    )
)


# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)

print("\n" + "=" * 60)
print("Evaluation completed successfully!")
print("=" * 60)