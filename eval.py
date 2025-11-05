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
# METRICS
# -----------------
print("-" * 30)
print("🔹 Computing metrics...")
print("-" * 30)

metrics_path = os.path.join(RESULTS_DIR, "inference_metrics.csv")

if all_labels and all_labels[0] is not None:
    # Concatenate predictions and labels from all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate precision, recall, F1-score (sample-wise) and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="samples"
    )
    acc = accuracy_score(all_labels, all_preds)

    # Store results in a dictionary
    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # Print evaluation metrics
    print("📊 Evaluation Results:")
    for k, v in results.items():
        print(f"{k:>10}: {v:.4f}")

    # -----------------
    # SAVE METRICS TO FILE
    # -----------------
    metrics_df = pd.DataFrame([results])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Saved metrics to {metrics_path}")

else:
    print("⚠️ No labels found — skipping metrics computation.")
    all_preds = np.concatenate(all_preds, axis=0)
