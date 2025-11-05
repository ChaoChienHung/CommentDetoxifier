import os
import torch
import numpy as np
import pandas as pd
from config import *
from dataset import CommentDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------
# Ensure Directories Exist
# ------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_CACHE, exist_ok=True)

# ------------------------
# LOAD MODEL & TOKENIZER
# ------------------------
print("-" * 38)
print("🔹 Loading tokenizer and model...")
print("-" * 38)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Load trained BERT model for multi-label classification
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Move model to specified device (GPU/CPU) and set to evaluation mode
model.to(DEVICE)
model.eval()

# -----------------
# LOAD TEST DATA
# -----------------
print("-" * 30)
print("🔹 Loading test data...")
print("-" * 30)

# Load test data CSV file
test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Create a CommentDataset object (custom PyTorch Dataset)
test_dataset = CommentDataset(
    data=test_data,
    tokenizer_name=TOKENIZER,
    tokenizer_cache=BERT_TOKENIZER_CACHE,
    cache_data=os.path.join(DATA_CACHE, "test_dataset.pt")  # optional caching for faster reload
)

# Create a DataLoader for batching during inference
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE_EVAL, 
    shuffle=False
)

# -----------------
# INFERENCE LOOP
# -----------------
print("-" * 30)
print("🔹 Running inference...")
print("-" * 30)

# Lists to collect predictions and true labels
all_preds, all_labels = [], []

# Disable gradient computation for inference to save memory
with torch.no_grad():
    for batch in test_loader:
        # Move input tensors to the device
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.cpu().numpy()
        else:
            labels = None

        # Forward pass through the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Apply sigmoid to logits to get probabilities for each label
        probs = torch.sigmoid(outputs.logits).cpu().numpy()

        # Convert probabilities to binary predictions using threshold
        preds = (probs > THRESHOLD).astype(int)

        # Collect predictions and labels
        all_preds.append(preds)
        all_labels.append(labels)

# -----------------
# METRICS PER LABEL
# -----------------
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
metrics_path = os.path.join(RESULTS_DIR, "inference_metrics_per_label.csv")

if all_labels and all_labels[0] is not None:
    # Concatenate predictions and labels from all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Initialize lists to store per-label metrics
    per_label_results = []

    # Loop over each label
    for i, label_name in enumerate(LABELS):
        label_true = all_labels[:, i]
        label_pred = all_preds[:, i]

        # Compute metrics for this label
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_true, label_pred, average="binary"
        )
        acc = accuracy_score(label_true, label_pred)

        per_label_results.append({
            "label": label_name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        # Print metrics
        print(f"📊 Metrics for '{label_name}':")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1       : {f1:.4f}\n")

    # Save per-label metrics to CSV
    metrics_df = pd.DataFrame(per_label_results)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Saved per-label metrics to {metrics_path}")

else:
    print("⚠️ No labels found — skipping metrics computation.")
    all_preds = np.concatenate(all_preds, axis=0)
