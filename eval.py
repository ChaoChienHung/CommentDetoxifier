import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from dataset import CommentDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc

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

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

# -----------------
# LOAD TEST DATA
# -----------------
print("-" * 30)
print("🔹 Loading test data...")
print("-" * 30)

test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

test_dataset = CommentDataset(
    data=test_data,
    tokenizer_name=TOKENIZER,
    tokenizer_cache=BERT_TOKENIZER_CACHE,
    cache_data=os.path.join(DATA_CACHE, "test_dataset.pt")
)

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

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.cpu().numpy()
        else:
            labels = None

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        preds = (probs > THRESHOLD).astype(int)

        all_preds.append(preds)
        all_probs.append(probs)
        all_labels.append(labels)

# -----------------
# METRICS PER LABEL
# -----------------
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
metrics_path = os.path.join(RESULTS_DIR, "inference_metrics_per_label.csv")

if all_labels and all_labels[0] is not None:
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    per_label_results = []

    for i, label_name in enumerate(LABELS):
        label_true = all_labels[:, i]
        label_pred = all_preds[:, i]

        precision, recall, f1, _ = precision_recall_fscore_support(
            label_true, label_pred, average="binary"
        )
        acc = accuracy_score(label_true, label_pred)

        # Compute ROC and AUC
        fpr, tpr, _ = roc_curve(label_true, all_probs[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot and save individual ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {label_name}')
        plt.legend(loc='lower right')

        roc_path = os.path.join(RESULTS_DIR, f"roc_{label_name}.png")
        plt.savefig(roc_path)
        plt.close()

        print(f"📊 Metrics for '{label_name}':")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  AUC      : {roc_auc:.4f}")
        print(f"  🧩 Saved ROC curve to {roc_path}\n")

        per_label_results.append({
            "label": label_name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": roc_auc
        })

    metrics_df = pd.DataFrame(per_label_results)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Saved per-label metrics to {metrics_path}")

    # Combined ROC plot for all labels
    plt.figure(figsize=(8, 6))
    for i, label_name in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Labels')
    plt.legend(loc='lower right')

    combined_roc_path = os.path.join(RESULTS_DIR, "roc_all_labels.png")
    plt.savefig(combined_roc_path)
    plt.close()
    print(f"✅ Saved combined ROC curve to {combined_roc_path}")

else:
    print("⚠️ No labels found — skipping metrics and ROC computation.")
    all_preds = np.concatenate(all_preds, axis=0)
