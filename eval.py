import torch
import numpy as np
import pandas as pd
from dataset import CommentDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import DATA_CACHE, TOKENIZER, DEVICE, THRESHOLD, BATCH_SIZE_EVAL, CACHE_DIR, MODEL_PATH

# ------------------------
# LOAD MODEL & TOKENIZER
# ------------------------
print("-" * 30)
print("🔹 Loading tokenizer and model...")
print("-" * 30)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER, cache_dir=f"{CACHE_DIR}/tokenizers")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, cache_dir=f"{CACHE_DIR}/models")
model.to(DEVICE)
model.eval()

# -----------------
# LOAD TEST DATA
# -----------------
print("-" * 20)
print("🔹 Loading test data...")
print("-" * 20)
test_data = pd.read_csv("data/test.csv")

test_dataset = CommentDataset(
    data=test_data,
    tokenizer_name=TOKENIZER,
    cache_path=f"{DATA_CACHE}/test_dataset.pt"
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)

# -----------------
# INFERENCE LOOP
# -----------------
print("-" * 20)
print("🔹 Running inference...")
print("-" * 20)
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        preds = (probs > THRESHOLD).astype(int)

        all_preds.append(preds)
        all_labels.append(labels)

# -----------------
# METRICS
# -----------------
print("-" * 30)
print("🔹 Computing metrics...")
print("-" * 30)

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="samples")
acc = accuracy_score(all_labels, all_preds)

results = {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1": f1
}

print("📊 Evaluation Results:")
for k, v in results.items():
    print(f"{k:>10}: {v:.4f}")

# ---------------------------
# OPTIONAL: SAVE PREDICTIONS
# ---------------------------
output_df = test_data.copy()
for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
    output_df[f"pred_{col}"] = all_preds[:, i]

output_path = "./results/inference_predictions.csv"
output_df.to_csv(output_path, index=False)
print(f"✅ Saved predictions to {output_path}")
