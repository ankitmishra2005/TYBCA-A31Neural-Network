# Detection of AI-Generated Text Using Deep Learning and Transformer Models

## Project Overview

This project focuses on detecting whether a given piece of text is **human-written** or **AI-generated** using Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques.

Two approaches are implemented and compared:

1. **TF-IDF + Logistic Regression** — traditional machine-learning baseline
2. **DistilBERT** — transformer-based deep-learning model

The project also includes an interactive prediction program that allows users to enter text and receive a classification with the model's predicted probabilities.

---

## Problem Statement

The increasing use of generative AI systems has made it difficult to distinguish between human-written and AI-generated content. This project investigates whether NLP and deep-learning techniques can be used to classify text according to its likely source.

---

## Objectives

* Collect a dataset containing human-written and AI-generated text.
* Clean and preprocess the text data.
* Create training, validation, and test datasets.
* Develop a traditional TF-IDF + Logistic Regression baseline.
* Train a DistilBERT transformer model for binary text classification.
* Evaluate the models using accuracy, precision, recall, and F1-score.
* Develop an interactive prediction system.
* Compare the performance of the traditional and transformer-based approaches.

---

## Dataset

The project uses the **HC3 (Human ChatGPT Comparison Corpus)** dataset.

The original dataset contains questions together with human-written and ChatGPT-generated answers.

After extracting the answers, removing empty entries and removing duplicate text, the processed dataset contained:

* **79,331 unique text samples**
* **53,086 human-written samples**
* **26,245 AI-generated samples**

The data was divided using a stratified split into:

* Training: **55,531 samples**
* Validation: **11,900 samples**
* Test: **11,900 samples**

A balanced subset of **20,000 samples** was created for DistilBERT training:

* 10,000 human-written
* 10,000 AI-generated

---

## Project Architecture

```text
HC3 Dataset
     |
     v
Data Cleaning
     |
     v
Duplicate Removal
     |
     v
Train / Validation / Test Split
     |
     +---------------------------+
     |                           |
     v                           v
TF-IDF + Logistic          DistilBERT Transformer
Regression                       |
     |                           |
     v                           v
Baseline Evaluation        Model Training
     |                           |
     +-------------+-------------+
                   |
                   v
             Test Evaluation
                   |
                   v
          Interactive Prediction
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* DistilBERT
* TF-IDF
* Logistic Regression
* Git and GitHub

---

## Machine Learning Baseline

The baseline model uses:

```text
TF-IDF
   |
   v
Logistic Regression
```

TF-IDF converts text into numerical feature vectors using unigram and bigram features.

The Logistic Regression classifier then performs binary classification:

```text
0 → Human-written
1 → AI-generated
```

### Baseline Test Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 97.96% |
| Precision | 96.78% |
| Recall    | 97.05% |
| F1 Score  | 96.92% |

---

## Deep Learning Model

The deep-learning approach uses:

**DistilBERT (`distilbert-base-uncased`)**

DistilBERT is a smaller and faster transformer architecture derived from BERT.

The model was fine-tuned for binary sequence classification:

```text
Input Text
    |
    v
DistilBERT Tokenizer
    |
    v
DistilBERT
    |
    v
Classification Layer
    |
    +------ Human-written
    |
    +------ AI-generated
```

### Training Configuration

* Model: DistilBERT
* Maximum sequence length: 128 tokens
* Batch size: 4
* Training samples: 20,000
* Validation samples: 11,900
* Epochs: 1
* Device: CPU

---

## DistilBERT Test Results

The trained model was evaluated on the untouched test set containing **11,900 samples**.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **98.31%** |
| Precision | **95.23%** |
| Recall    | **99.90%** |
| F1 Score  | **97.51%** |

### Confusion Matrix

```text
                  Predicted
                Human     AI

Actual Human     7766     197
Actual AI           4    3933
```

The model correctly classified 7,766 human-written samples and 3,933 AI-generated samples in the test set.

---

## Model Comparison

| Metric    | TF-IDF + Logistic Regression | DistilBERT |
| --------- | ---------------------------: | ---------: |
| Accuracy  |                       97.96% | **98.31%** |
| Precision |                   **96.78%** |     95.23% |
| Recall    |                       97.05% | **99.90%** |
| F1 Score  |                       96.92% | **97.51%** |

The DistilBERT model achieved higher accuracy, recall, and F1-score on the held-out HC3 test set, while the TF-IDF baseline achieved higher precision.

---

## Interactive Prediction

The project includes:

```text
src/predict.py
```

Run it using:

```bash
python src/predict.py
```

The program accepts user text and returns:

```text
Prediction       : Human-written
Human probability: XX.XX%
AI probability   : XX.XX%
```

Very short inputs generate a warning because predictions from extremely short text can be unreliable.

---

## Project Structure

```text
TYBCA-A31Neural-Network/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── split_dataset.py
│   ├── create_transformer_dataset.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── evaluate_transformer.py
│   ├── predict.py
│   ├── test_real_samples.py
│   └── diagnose_predictions.py
│
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ankitmishra2005/TYBCA-A31Neural-Network.git
```

### 2. Open the project

```bash
cd TYBCA-A31Neural-Network
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the prediction system

```bash
python src/predict.py
```

---

## Limitations

This project should not be interpreted as a universal or definitive AI-content detector.

The reported performance is based on the HC3 dataset and therefore reflects the characteristics and distribution of that dataset. Text from other domains, writing styles, AI systems, or generation methods may produce different results.

In particular:

* Very short text provides limited linguistic information.
* AI-generated text can be edited or paraphrased.
* Human-written text may sometimes resemble AI-generated writing.
* The model may learn dataset-specific patterns.
* A high test-set score does not guarantee the same performance on unseen real-world sources.

Therefore, the predictions should be treated as **model-based classifications rather than proof of authorship**.

---

## Conclusion

The project demonstrates an end-to-end NLP and deep-learning pipeline for binary classification of human-written and AI-generated text.

A traditional TF-IDF + Logistic Regression model was first developed as a baseline. A DistilBERT transformer model was then trained and evaluated on the same held-out test distribution.

The DistilBERT model achieved **98.31% test accuracy** and an **F1-score of 97.51%**, with **99.90% recall for AI-generated text** on the HC3 test set.

The project also provides an interactive prediction interface for testing new text.

---

## Author

**Ankit Mishra**

TYBCA
KES' B. K. Shroff College of Arts and Commerce
University of Mumbai
