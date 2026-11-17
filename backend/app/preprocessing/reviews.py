"""The original Colab cleaning function and input validation."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MAX_REVIEW_CHARACTERS = 10_000


def clean_text(text: str) -> str:
    """Apply the verified training cleaning, without fitting any transformer."""
    if not isinstance(text, str):
        raise ValueError("Review must be a string.")
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def preprocess_review(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review must be a non-empty English text.")
    if len(text) > MAX_REVIEW_CHARACTERS:
        raise ValueError("Review exceeds the 10000 character limit.")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Review has no words left after training preprocessing.")
    return cleaned
