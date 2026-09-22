import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/distilbert_ai_detector"
TEST_PATH = "data/processed/test.csv"
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()


def predict(text):

    encoding = tokenizer(
        str(text),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()

    human_prob = probabilities[0][0].item() * 100
    ai_prob = probabilities[0][1].item() * 100

    return prediction, human_prob, ai_prob


# Load test data
df = pd.read_csv(TEST_PATH)

# Take 5 known Human and 5 known AI samples
human_samples = df[df["label"] == 0].head(5)
ai_samples = df[df["label"] == 1].head(5)

samples = pd.concat([human_samples, ai_samples])

print("\n" + "=" * 75)
print("DIAGNOSTIC TEST")
print("=" * 75)

correct = 0

for number, (_, row) in enumerate(samples.iterrows(), start=1):

    true_label = int(row["label"])

    prediction, human_prob, ai_prob = predict(row["text"])

    if prediction == true_label:
        result = "CORRECT"
        correct += 1
    else:
        result = "WRONG"

    true_name = "Human" if true_label == 0 else "AI"
    predicted_name = "Human" if prediction == 0 else "AI"

    print(f"\nSample {number}")
    print("-" * 40)
    print(f"True label       : {true_name}")
    print(f"Prediction       : {predicted_name}")
    print(f"Human probability: {human_prob:.2f}%")
    print(f"AI probability   : {ai_prob:.2f}%")
    print(f"Text length      : {len(str(row['text']))}")
    print(f"Result           : {result}")


print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

print(f"Correct predictions : {correct}/10")
print(f"Accuracy            : {(correct / 10) * 100:.2f}%")