import torch
import datetime

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
MODEL_PATH = f"{RESULTS_DIR}/20251027-123456"  # Best checkpoint folder

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

# -----------------
# Timestamp
# -----------------
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
