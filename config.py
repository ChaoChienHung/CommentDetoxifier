import os
import torch
import datetime
from pydantic import BaseModel, Field

# -----------------
# Directories
# -----------------
CACHE_DIR = "./cache"
RESULTS_DIR = "./results"
MODEL_CACHE = f"{CACHE_DIR}/models"
DATA_CACHE = f"{CACHE_DIR}/datasets"
TOKENIZER_CACHE = f"{CACHE_DIR}/tokenizers"

# -----------------
# Model & Tokenizer
# -----------------
MODEL = "bert-base-uncased"
TOKENIZER = "bert-base-uncased"
MODEL_PATH = f"{RESULTS_DIR}/20251028-021857/checkpoint-35904"  # Best checkpoint folder

# -----------------
# Hyperparameters
# -----------------
EPOCHS = 10
THRESHOLD = 0.5
BATCH_SIZE_EVAL = 32
LEARNING_RATE = 2e-6
BATCH_SIZE_TRAIN = 16

# -----------------
# Device
# -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------
# API Key
# ----------
API_KEY: str = os.environ.get("OPENAI_API_KEY")

# -----------------
# Timestamp
# -----------------
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ---------
# Schema
# ---------
class LLMReply(BaseModel):
    success: bool = Field(description="Indicates whether the LLM successfully processed the input.")
    has_meaning: bool = Field(description="Indicates if the original comment contains meaningful content.")
    error_message: str = Field(description="Detailed error message if processing failed; empty if successful.")
    revised_comment: str = Field(description="The revised version of the comment, modified to be socially acceptable.")
