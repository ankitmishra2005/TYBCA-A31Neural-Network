import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv("data/processed/train.csv")
validation_df = pd.read_csv("data/processed/validation.csv")
test_df = pd.read_csv("data/processed/test.csv")

print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print(f"Test samples:       {len(test_df)}")


X_train = train_df["text"]
y_train = train_df["label"]

X_validation = validation_df["text"]
y_validation = validation_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


# --------------------------------------------------
# 2. TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_validation_tfidf = vectorizer.transform(X_validation)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature creation completed.")
print(f"Number of features: {X_train_tfidf.shape[1]}")


# --------------------------------------------------
# 3. Train Logistic Regression
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Training completed.")


# --------------------------------------------------
# 4. Validation evaluation
# --------------------------------------------------

print("\nEvaluating on validation set...")

validation_predictions = model.predict(X_validation_tfidf)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_precision = precision_score(
    y_validation,
    validation_predictions
)

validation_recall = recall_score(
    y_validation,
    validation_predictions
)

validation_f1 = f1_score(
    y_validation,
    validation_predictions
)

print("\nValidation Results")
print("------------------")
print(f"Accuracy : {validation_accuracy:.4f}")
print(f"Precision: {validation_precision:.4f}")
print(f"Recall   : {validation_recall:.4f}")
print(f"F1 Score : {validation_f1:.4f}")


# --------------------------------------------------
# 5. Final test evaluation
# --------------------------------------------------

print("\nEvaluating on test set...")

test_predictions = model.predict(X_test_tfidf)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions
)

test_recall = recall_score(
    y_test,
    test_predictions
)

test_f1 = f1_score(
    y_test,
    test_predictions
)


print("\nTest Results")
print("------------")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")


# --------------------------------------------------
# 6. Classification report
# --------------------------------------------------

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Human",
            "AI-generated"
        ]
    )
)


# --------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------

print("\nConfusion Matrix")
print("----------------")

cm = confusion_matrix(
    y_test,
    test_predictions
)

print(cm)


# --------------------------------------------------
# 8. Save model and vectorizer
# --------------------------------------------------

print("\nSaving baseline model...")

joblib.dump(
    model,
    "models/tfidf_logistic_regression.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("Baseline model saved successfully.")