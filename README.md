# CommentDetoxifier

**CommentDetoxifier** is a system designed to detect and neutralize harmful online comments. It identifies hateful, abusive, or threatening speech and transforms it into socially acceptable language — preserving the original context and intent.

## Project Structure
```bash
CommentDetoxifier/
├── config.py               # Project constants, hyperparameters, paths, device setup
├── dataset.py              # Custom PyTorch Dataset for multi-label toxicity detection
├── train.py                # Script to train the BERT classifier with multi-label output
├── eval.py                 # Evaluate the model on a test dataset
├── main.py                 # Main inference & detoxification pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and usage instructions
```

- **`config.py`**  
  Centralized configuration for the project. Defines constants, hyperparameters, directory paths, device setup, and data schemas, ensuring consistency and easy maintenance across scripts.

- **`dataset.py`**  
  Defines a custom PyTorch `CommentDataset` class for multi-label comment toxicity classification. Integrates Hugging Face tokenizers and provides optional caching for faster repeated usage.

- **`eval.py`**  
  Performs inference and evaluation on a pre-trained BERT multi-label classifier. Computes performance metrics, optionally saves predictions, and seamlessly integrates with the `CommentDataset` pipeline.

- **`main.py`**  
  Orchestrates the complete detoxification pipeline. Combines a BERT-based toxicity detector with an OpenAI LLM to automatically transform toxic comments into acceptable language. Includes model loading, tokenization, toxicity inference, LLM-based detoxification, logging, and robust retry mechanisms.

- **`model.py`**  
  Defines the architecture of the toxicity detection model. Currently implements a BERT-based classifier for multi-label toxicity detection.

- **`train.py`**  
  Trains a BERT-based multi-label classifier across six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. Uses Hugging Face’s `Trainer` API and integrates with W&B for experiment tracking and logging.

## Key Improvements
1. Modularity
   - Extracted the OpenAI agent into agent.py.
   - Tokenizer/model loading, toxicity detection, and LLM detoxification separated clearly.
2. Retry & Fallback Logic
   - `MAX_ATTEMPTS` ensures we don’t loop indefinitely.
   - LLM fallback handled gracefully.
3. Logging
   - Logs all key events, including errors, toxic detection, and revised comments.
5. User-Friendly
   - API key prompt hidden with getpass.
   - Clear feedback for socially acceptable or detoxified comments.
