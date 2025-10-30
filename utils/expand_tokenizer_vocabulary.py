import os
from config import TOKENIZER, TOKENIZER_CACHE, MODEL, MODEL_CACHE
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, cache_dir=os.path.join(TOKENIZER_CACHE, "toxic"))
model = AutoModel.from_pretrained(MODEL, cache_dir=os.path.join(MODEL_CACHE, "toxic"))

# new tokens
new_tokens = ["[ar]", "[bg]", "[de]", "[el]", "[en]", "[es]", 
              "[fr]", "[hi]", "[it]", "[ja]", "[nl]", "[pl]", "[pt]",
              "[ru]", "[sw]", "[th]", "[tr]", "[ur]", "[vi]", "[zh]"]

existing_tokens = set(tokenizer.get_vocab().keys())
new_tokens = set(new_tokens) - existing_tokens

# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained(os.path.join(MODEL_CACHE, "toxic"))
tokenizer.save_pretrained(os.path.join(TOKENIZER_CACHE, "toxic"))