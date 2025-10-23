import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# Create the directory if it doesn't exist
save_path = "../models/tokenizer/vocab_expanded_tokenizer"
os.makedirs(save_path, exist_ok=True)

def get_training_corpus(file_paths: list[str]):
    dataset = pd.Series([])
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if "text" in df.keys():
            dataset = pd.concat([dataset, df["text"]])

        else:
            dataset = pd.concat([dataset, df['messageContent']])
    
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx: start_idx + 1000]
        yield samples

# Initialize the tokenizer
tokenizer_path = "../models/huggingface/tokenizer"
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert", cache_dir=tokenizer_path)

# Train on additional corpus
training_corpus = get_training_corpus(file_paths)
extra_corpus_tokenizer = tokenizer.train_new_from_iterator(training_corpus, 45000)

emoji = np.genfromtxt('../dataset/emoji.txt', dtype=str, delimiter='\n', encoding='utf-8')

# Add new tokens to the existing tokenizer vocabulary
new_vocabulary = list(extra_corpus_tokenizer.vocab.keys())
new_vocabulary = set(new_vocabulary) | set(emoji) - set(tokenizer.vocab.keys())

tokenizer.add_tokens(list(new_vocabulary))

# Save the tokenizer with the expanded vocabulary
tokenizer.save_pretrained(save_path)

print(f"Tokenizer saved at {save_path}")