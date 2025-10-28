import os
import time
import torch
import logging
import getpass
from typing import Literal

from openai import OpenAI
from transformers import BertTokenizer, BertForSequenceClassification

from config import API_KEY, TOKENIZER, DEVICE, CACHE_DIR, MODEL, MODEL_PATH, THRESHOLD, LLMReply

# -----------------
# OpenAI Agent
# -----------------
class Agent:
    """
    Agent for converting toxic comments to socially acceptable comments.
    """

    def __init__(self, model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o-mini", max_retries: int = 3):
        self.client: OpenAI | None = None
        self.model: str = model
        self.max_retries: int = max_retries
        self._create_secure_openai_client()

    def _create_secure_openai_client(self):
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

        schema = LLMReply.model_json_schema()
        schema["additionalProperties"] = False
        response_format_schema = {"name": "llm_reply", "schema": schema, "strict": True}

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
                    return LLMReply.model_validate_json(response.choices[0].message.content)
                except Exception as e:
                    logger.error(f"Validation failed: {e}")
                    return LLMReply(success=False, has_meaning=False, error_message=str(e), revised_comment=comment)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
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

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER, cache_dir=f"{CACHE_DIR}/tokenizers")
    # If we have trained our model
    if os.path.exists(MODEL_PATH):
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH, cache_dir=f"{CACHE_DIR}/models")

    # Else fallback to pretrained base model
    else:
        model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=6, problem_type="multi_label_classification", cache_dir=f"{CACHE_DIR}/models")
    model.to(DEVICE)
    model.eval()

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

    comment: str = input("Please input the comment: ")

    # -----------------
    # Inference Loop
    # -----------------

    counter: int = 0
    shouldDetect: bool = True
    llm_reply: LLMReply = LLMReply(success=True, has_meaning=True, error_message="", revised_comment=comment)

    while shouldDetect:
        # -----------------
        # Detect toxicity
        # -----------------
        with torch.no_grad():
            encodings = tokenizer(comment, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = encodings["input_ids"].to(DEVICE)
            attention_mask = encodings["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)

        if not preds.any():
            logger.info("The comment is socially acceptable.")
            break

        logger.info(f"The comment: \"{comment}\" is detected to be toxic.")
        # -----------------
        # Detoxify with LLM
        # -----------------
        llm_reply = agent.detoxify(comment)

        if llm_reply.success:
            logger.info(f"Original Comment: \"{comment}\"")
            logger.info(f"OpenAI LLM Revised Comment: \"{llm_reply.revised_comment}\"")
            comment = llm_reply.revised_comment

        else:
            logger.error(llm_reply.error_message)
            counter += 1

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
