import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import MODEL, TOKENIZER_CACHE

class CommentDataset(Dataset):
    """
    Dataset for comment toxicity classification with caching support.

    Args:
        data (pd.DataFrame): The data containing comment_text and label columns.
        tokenizer_name (str): Hugging Face tokenizer name or path.
        cache_path (str, optional): Path to save/load preprocessed dataset.
        max_length (int): Max token length for tokenizer.
    """
    def __init__(self, data, tokenizer_name=MODEL, cache_path=None, max_length=128):
        self.data = data.reset_index(drop=True)
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.cache_path = cache_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=TOKENIZER_CACHE)

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            cached = torch.load(cache_path)
            self.encodings = cached["encodings"]
            self.labels = cached["labels"]

        else:
            print("⚙️ Tokenizing dataset...")
            texts = list(self.data["comment_text"].astype(str))
            self.encodings = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            self.labels = torch.tensor(
                self.data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values,
                dtype=torch.float
            )

            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save({"encodings": self.encodings, "labels": self.labels}, cache_path)
                print(f"✅ Saved tokenized dataset to {cache_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
