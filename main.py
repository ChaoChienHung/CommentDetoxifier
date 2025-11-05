import os
import time
import torch
import logging
import getpass
from typing import Literal

from config import *
from openai import OpenAI
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer

# -----------------
# OpenAI Agent
# -----------------
class Agent:
    """
    Agent for converting toxic comments to socially acceptable comments
    using OpenAI's LLM. If OpenAI API is unavailable, falls back gracefully.
    """

    def __init__(self, model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o-mini", max_retries: int = 3):
        """
        Initialize the Agent.

        Args:
            model: Name of the OpenAI model to use.
            max_retries: Number of retries if request to OpenAI fails.
        """
        self.client: OpenAI | None = None
        self.model: str = model
        self.max_retries: int = max_retries
        self._create_secure_openai_client()

    def _create_secure_openai_client(self):
        """
        Create a secure OpenAI client using the API key from environment variables.
        Logs warning if no key is found.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "No OPENAI_API_KEY found. "
                "Set it using 'export OPENAI_API_KEY=your_key' (Linux/Mac) "
                "or 'setx OPENAI_API_KEY your_key' (Windows)."
            )
            return
        try:
            self.client = OpenAI(api_key=api_key)
            self.client.models.list()  # Test connection
            logger.info("OpenAI client created and tested successfully.")
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")

    def detoxify(self, comment: str) -> LLMReply:
        """
        Convert a toxic comment to a socially acceptable version using OpenAI.
        Falls back to naive method if client unavailable.

        Args:
            comment: The input comment string.

        Returns:
            LLMReply object with revised comment and status.
        """
        if not self.client:
            logger.warning("No OpenAI client detected. Using naive fallback.")
            return LLMReply(
                success=False,
                has_meaning=False,
                error_message="No OpenAI client detected. Using naive fallback.",
                revised_comment=comment
            )

        if not isinstance(comment, str):
            raise TypeError(f"Expected comment as string, got {type(comment)}")

        # Define JSON schema for response validation
        schema = LLMReply.model_json_schema()
        schema["additionalProperties"] = False
        response_format_schema = {"name": "llm_reply", "schema": schema, "strict": True}

        # Retry loop in case OpenAI API call fails
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Convert hateful/toxic comments into socially acceptable comments "
                                "while preserving meaning."
                            )
                        },
                        {"role": "user", "content": comment}
                    ],
                    response_format={"type": "json_schema", "json_schema": response_format_schema}
                )

                try:
                    # Validate and return the response as LLMReply
                    return LLMReply.model_validate_json(response.choices[0].message.content)
                except Exception as e:
                    logger.error(f"Validation failed: {e}")
                    return LLMReply(success=False, has_meaning=False, error_message=str(e), revised_comment=comment)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                if attempt == self.max_retries - 1:
                    logger.error("All retries failed. Using fallback.")
                    return LLMReply(success=False, has_meaning=False, error_message=str(e), revised_comment=comment)


# -----------------
# Main
# -----------------
if __name__ == "__main__":

    # -----------------
    # Logger setup
    # -----------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename='app.log',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # -----------------
    # Load tokenizer and model
    # -----------------
    print("-" * 40)
    print("🔹 Loading tokenizer and model...")
    print("-" * 40)


    # Load trained BERT model if exists; otherwise, fallback to base model
    if os.path.exists(MODEL_PATH):
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        
    else:
        # Load BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER, cache_dir=BERT_TOKENIZER_CACHE)
        model = BertForSequenceClassification.from_pretrained(
            MODEL,
            num_labels=6, 
            problem_type="multi_label_classification", 
            cache_dir=BERT_CACHE
        )

    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode

    # -----------------
    # Load OpenAI agent
    # -----------------
    print("-" * 30)
    print("🔹 Loading OpenAI agent...")
    print("-" * 30)
    agent = Agent()

    # -----------------
    # Prompt for API key if missing
    # -----------------
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key (will not be shown as you type): ")
        os.environ["OPENAI_API_KEY"] = api_key

    # Prompt user for input comment
    comment: str = input("Please input the comment: ")

    # -------------------
    # Language Detection
    # -------------------
    detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR, cache_dir=ROBERTA_TOKENIZER_CACHE)
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR, cache_dir=ROBERTA_CACHE)

    inputs = detector_tokenizer(comment, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = detector(**inputs).logits

    preds = torch.softmax(logits, dim=-1)
    
    # Map raw predictions to languages
    id2lang = detector.config.id2label
    vals, idxs = torch.max(preds, dim=1)
    language_score = {id2lang[k.item()]: v.item() for k, v in zip(idxs, vals)}
    comment = f'[{list(language_score.keys())[0]}] ' + comment
    
    # -----------------
    # Inference Loop
    # -----------------
    counter: int = 0  # Track number of failed attempts
    shouldDetect: bool = True
    llm_reply: LLMReply = LLMReply(success=True, has_meaning=True, error_message="", revised_comment=comment)

    while shouldDetect:
        # -----------------
        # Detect toxicity using BERT
        # -----------------
        with torch.no_grad():
            encodings = tokenizer(comment, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = encodings["input_ids"].to(DEVICE)
            attention_mask = encodings["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)

        # If no toxic labels detected, stop the loop
        if not preds.any():
            logger.info("The comment is socially acceptable.")
            break

        logger.info(f"The comment: \"{comment}\" is detected to be toxic.")

        # -----------------
        # Detoxify with OpenAI LLM
        # -----------------
        llm_reply = agent.detoxify(comment)

        if llm_reply.success:
            logger.info(f"Original Comment: \"{comment}\"")
            logger.info(f"OpenAI LLM Revised Comment: \"{llm_reply.revised_comment}\"")
            comment = llm_reply.revised_comment

        else:
            logger.error(llm_reply.error_message)
            counter += 1

        # Stop after 3 failed attempts
        if counter == 3:
            break

    # -----------------
    # Final output
    # -----------------
    if llm_reply.success:
        if llm_reply.has_meaning:
            logger.info(f"Comment: {comment}")
            print(f"Comment: {comment}")

        else:
            logger.info("This comment doesn't contain any meaningful content.")
            print("This comment doesn't contain any meaningful content.")
            
    else:
        logger.error(f"Model's error: {llm_reply.error_message}")
        print(f"Model's error: {llm_reply.error_message}")
