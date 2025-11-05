import os
import wandb
import torch
import datetime
import pandas as pd
from config import *
from tqdm import tqdm
from torch.optim import AdamW
from dataset import CommentDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,
    EarlyStoppingCallback
)

# Debugging
# raise SystemExit("Debugging complete.")

# Load full dataset
data = pd.read_csv("data/train.csv")

# Split into train and validation
train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)



# Train Dataset
train_dataset = CommentDataset(data=train_df, cache_path=os.path.join(DATA_CACHE, "train_dataset.pt"))

# Validation Dataset
val_dataset = CommentDataset(data=val_df, cache_path=os.path.join(DATA_CACHE, "val_dataset.pt"))

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

os.makedirs(DATA_CACHE, exist_ok=True)
os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs(TOKENIZER_CACHE, exist_ok=True)

os.makedirs(BERT_CACHE, exist_ok=True)


# -----------------
# W&B Setup
# -----------------
wandb.init(project="CommentDetoxifier", name=CURRENT_TIME)
wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size_train": BATCH_SIZE_TRAIN,
    "batch_size_eval": BATCH_SIZE_EVAL
}

# -----------------
# Device Setup 
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

# -----------------
# Tokenizer
# -----------------
tokenizer = BertTokenizer.from_pretrained(MODEL, cache_dir=TOKENIZER_CACHE)

# -----------------
# Model Setup
# -----------------
model = BertForSequenceClassification.from_pretrained(
    MODEL, num_labels=6, problem_type="multi_label_classification", cache_dir=BERT_CACHE
).to(device)

training_args = TrainingArguments(
    output_dir=os.path.join(BERT_CACHE, CURRENT_TIME),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,  # optional but recommended with early stopping
    metric_for_best_model="f1",   
    greater_is_better=True        # True if higher metric is better
)

def compute_metrics(pred):
    logits = torch.tensor(pred.predictions)
    labels = torch.tensor(pred.label_ids)
    preds = torch.sigmoid(logits) > 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='samples')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
