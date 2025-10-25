import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CommentDataset(Dataset):
    """
    data: pd.DataFrame
    """
    def __init__(self, data, tokenizer_name="bert-base-uncased", max_length=128):
        self.data: pd.DataFrame = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        comment = str(self.data.iloc[idx]["comment_text"])
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # squeeze to remove extra dimension from return_tensors
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        labels = self.data.iloc[idx][["toxic", "severe_toxic", "obscene",
                              "threat", "insult", "identity_hate"]].values.astype(float)
        item["labels"] = torch.tensor(labels, dtype=torch.float)

        return item
