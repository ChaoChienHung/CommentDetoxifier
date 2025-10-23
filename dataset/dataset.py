import torch
from torch.utils.data import Dataset


class ToxicDataaset(Dataset):
    def __init__(self, filepath, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)
        return item

    def __len__(self):
        return len(self.labels)
    
    def get_labels(self):
        return self.labels