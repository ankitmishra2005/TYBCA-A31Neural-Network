import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/distilbert_ai_detector"
MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# Load Model
# ==========================================

print("Loading AI Text Detection Model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print(f"Device: {DEVICE}")


# ==========================================
# Prediction Function
# ==========================================

def predict_text(text):

    encoding = tokenizer(
        text,
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

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    human_probability = probabilities[0][0].item() * 100
    ai_probability = probabilities[0][1].item() * 100

    if prediction == 0:
        label = "Human-written"
    else:
        label = "AI-generated"

    return label, human_probability, ai_probability


# ==========================================
# User Input
# ==========================================

print("\n" + "=" * 60)
print("AI-GENERATED TEXT DETECTOR")
print("=" * 60)

print("\nEnter a piece of text to analyze.")
print("Type 'exit' to close the program.\n")


while True:

    text = input("Enter text: ")

    if text.lower() == "exit":
        print("\nProgram closed.")
        break

    if not text.strip():
        print("Please enter some text.\n")
        continue

    # Warn users about very short text
    if len(text.split()) < 10:
        print("\nWarning: The entered text is very short.")
        print("Predictions may be unreliable for short text.")

    label, human_probability, ai_probability = predict_text(text)

    print("\nPrediction       :", label)
    print(f"Human probability: {human_probability:.2f}%")
    print(f"AI probability   : {ai_probability:.2f}%")
    print()