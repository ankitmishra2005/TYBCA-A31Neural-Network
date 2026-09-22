import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

TRAIN_PATH = "data/processed/transformer_train.csv"
VALIDATION_PATH = "data/processed/validation.csv"

MAX_LENGTH = 128
BATCH_SIZE = 4

SMOKE_TEST = False

if SMOKE_TEST:
    TRAIN_SAMPLES = 200
    VALIDATION_SAMPLES = 40
else:
    TRAIN_SAMPLES = 10000
    VALIDATION_SAMPLES = 11900

EPOCHS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# --------------------------------------------------
# Dataset class
# --------------------------------------------------

class TextDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length):

        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = str(self.texts[index])
        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

        return item


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)

validation_df = pd.read_csv(VALIDATION_PATH)


if SMOKE_TEST:

    train_df = train_df.sample(
        n=TRAIN_SAMPLES,
        random_state=42
    ).reset_index(drop=True)

    validation_df = validation_df.sample(
        n=VALIDATION_SAMPLES,
        random_state=42
    ).reset_index(drop=True)


print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

train_dataset = TextDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

validation_dataset = TextDataset(
    validation_df,
    tokenizer,
    MAX_LENGTH
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)


# --------------------------------------------------
# Training
# --------------------------------------------------

print("\nStarting training...")

model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    for step, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 10 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Step {step + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"\nEpoch {epoch + 1} completed. "
        f"Average Loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

print("\nRunning validation...")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in validation_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


accuracy = correct / total

print(f"\nValidation Accuracy: {accuracy:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

output_directory = "models/distilbert_ai_detector"

model.save_pretrained(output_directory)

tokenizer.save_pretrained(output_directory)

print(
    f"\nModel saved to: {output_directory}"
)