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
        cache_data (str, optional): Path to save/load preprocessed dataset.
        max_length (int): Max token length for tokenizer.
    """
    def __init__(self, data, tokenizer_name=MODEL, tokenizer_cache=os.path.join(TOKENIZER_CACHE, "toxic", "Miscellaneous"), cache_data=None, max_length=128):
        self.data = data.reset_index(drop=True)
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.cache_data = cache_data
        self.tokenizer_cache = tokenizer_cache

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.tokenizer_cache)

        if cache_data and os.path.exists(cache_data):
            print(f"Loading cached dataset from {cache_data}")
            cached = torch.load(cache_data)
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

            if cache_data:
                os.makedirs(os.path.dirname(cache_data), exist_ok=True)
                torch.save({"encodings": self.encodings, "labels": self.labels}, cache_data)
                print(f"✅ Saved tokenized dataset to {cache_data}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
