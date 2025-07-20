import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

from ann import ANNs
from naive_bayes import NaiveBayesTextClassifier
import torch
import torch.nn as nn

# -------------------------------------
# Load dataset
# -------------------------------------
# TODO 1:
# 1. Read the suicide intention dataset CSV.
# 2. Convert the "class" column into binary: 
#    - "suicide" → 1
#    - "non-suicide" → 0
# 3. Make sure the labels (`y`) are reshaped into a column vector (shape: [N, 1])
raise NotImplementedError()
df = 


# -------------------------------------
# Choose embedding method
# -------------------------------------
embedding_method = "manual"
if embedding_method == "bow":
    vectorizer = CountVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts).toarray()
elif embedding_method == "tfidf":
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts).toarray()
elif embedding_method == "tfidf-bigram":
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
    X = vectorizer.fit_transform(texts).toarray()
elif embedding_method == "manual":
    import re
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from collections import Counter

    # TODO 2:
    # 1. Lowercase and tokenize each text (using regex).
    # 2. Remove stopwords using `ENGLISH_STOP_WORDS`.
    # 3. Build vocabulary from most frequent words.
    # 4. For each sentence:
    #    - Calculate frequency weight for each word.
    #    - Pad or truncate to `max_len`.
    # 5. Normalize and convert to NumPy array.
    
    raise NotImplementedError("Write your code here")
    
else:
    raise ValueError("Invalid embedding method")

# -------------------------------------
# Define shared manual loss function
# -------------------------------------
def manual_bce_loss(preds, targets):
    epsilon = 1e-7
    preds = torch.clamp(preds, epsilon, 1 - epsilon)
    # TODO 3:
    # Implement the binary cross-entropy loss manually.
    # - Use epsilon to avoid log(0).
    # - Formula: −mean(y * log(p) + (1 − y) * log(1 − p))
    raise NotImplementedError("Write your code here")

# -------------------------------------
# Split data: Train, Val, Test (60/20/20)
# -------------------------------------
# TODO 4:
# Implement the binary cross-entropy loss manually.
# - Use epsilon to avoid log(0).
# - Formula: −mean(y * log(p) + (1 − y) * log(1 − p))
# 1. First split 20% of the data for final test set.
# 2. Then split the remaining 80% into:
#    - 60% train
#    - 20% validation
# 3. Choose an appropriate number of epochs.
raise NotImplementedError("Write your code here")
X_temp, X_test, y_temp, y_test = 
X_train, X_val, y_train, y_val = 
epoches=

# -------------------------------------
# ANN
# -------------------------------------
# TODO 5:
# 1. Customize the ANN structure: `layer_dims` can have more or fewer layers.
#    For example: [input_dim, 16, 16, 1] or deeper.
# 2. Tune the learning rate (e.g. 0.001, 0.01).
# 3. Monitor training and validation losses to avoid overfitting.
# 4. You may add dropout in ANN.py if needed later.
# ann = ANN(layer_dims=[X_train.shape[1], your configuration ,1], learning_rate=0.01)
raise NotImplementedError("Write your code here")
ann = ANN(layer_dims=[], learning_rate=)
ann_train_losses = []
ann_val_losses = []
for epoch in range(epoches):
    train_loss = ann.train_one_epoch(X_train, y_train)
    ann_train_losses.append(train_loss)

    y_val_pred = ann.predict_proba(X_val)
    val_output = torch.tensor(y_val_pred, dtype=torch.float32)
    val_target = torch.tensor(y_val, dtype=torch.float32)
    val_loss = manual_bce_loss(val_output, val_target).item()
    ann_val_losses.append(val_loss)

    if epoch < 5 or epoch % 50 == 0:
        print(f"[ANN] Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

ann_pred = ann.predict(X_test)

# -------------------------------------
# Naive Bayes
# -------------------------------------
nb = NaiveBayesTextClassifier()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# -------------------------------------
# Loss Plot
# -------------------------------------
def plot_loss(train_losses, val_losses, title, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_loss(ann_train_losses, ann_val_losses, "ANN Train vs Val Loss", f"ann_loss_{embedding_method}_plot.png")

# Helper function to compute evaluation metrics
y_true = y_test.flatten()
def compute_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return [
        report["accuracy"],
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"]
    ]

# Compute metrics for each model
ann_metrics = compute_metrics(y_true, ann_pred)
nb_metrics = compute_metrics(y_true, nb_pred)

# Define metric labels and plotting positions
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metric_names))
bar_width = 0.25

# Plotting the comparison chart
x = np.arange(len(metric_names))
bar_width = 0.30

plt.figure(figsize=(8, 6))
plt.bar(x - bar_width/2, ann_metrics, width=bar_width, label="ANN")
plt.bar(x + bar_width/2, nb_metrics,  width=bar_width, label="Naive Bayes")

plt.xticks(x, metric_names)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title(f"Model Performance Comparison Using {embedding_method} Embedding")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(f"model_performance_comparison_{embedding_method}.png")

# -------------------------------------
# Evaluation
# -------------------------------------``
print("\n[ANN] Classification Report:")
print(classification_report(y_true, ann_pred))
print("\n[Naive Bayes] Classification Report:")
print(classification_report(y_true, nb_pred))