import wandb
import torch
import datetime
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from dataset import CommentDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)

# Debugging
# raise SystemExit("Debugging complete.")

# Load full dataset
data = pd.read_csv("data/train.csv")

# Split into train and validation
train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)

# Train Dataset
train_dataset = CommentDataset(data=train_df)

# Validation Dataset
val_dataset = CommentDataset(data=val_df)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# -----------------
# Hyperparameters
# -----------------
learning_rate = 2e-6
epochs = 10
batch_size_train = 16
batch_size_eval = 32

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# -----------------
# W&B Setup
# -----------------
wandb.init(project="CommentDetoxifier", name=current_time)
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size_train": batch_size_train,
    "batch_size_eval": batch_size_eval
}

# -----------------
# Device Setup 
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

# -----------------
# Tokenizer
# -----------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -----------------
# Model Setup
# -----------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=6, problem_type="multi_label_classification"
).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size_train,
    per_device_eval_batch_size=batch_size_eval,
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
