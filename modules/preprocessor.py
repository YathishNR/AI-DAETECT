import re
import string
import unicodedata
from typing import Optional

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _STOP = set(ENGLISH_STOP_WORDS)
except Exception:
    _STOP = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "as", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "this", "that", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "what", "which",
    }


def unicode_normalize(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text))


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Normalize and lightly clean text for ML and consistent matching."""
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return ""
    s = unicode_normalize(str(text)).strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(rf"[{re.escape(string.punctuation)}]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    if remove_stopwords and s:
        tokens = [t for t in s.split() if t not in _STOP]
        s = " ".join(tokens)
    return s


def normalize_for_display(text: str, max_len: int = 500) -> str:
    if text is None:
        return ""
    s = unicode_normalize(str(text)).strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def row_to_combined_text(row_dict: dict, text_columns: Optional[list] = None) -> str:
    """Combine selected columns into one string for analysis."""
    if text_columns:
        parts = []
        for c in text_columns:
            if c in row_dict and row_dict[c] is not None:
                v = row_dict[c]
                if isinstance(v, (list, dict)):
                    parts.append(str(v))
                else:
                    parts.append(str(v))
        return " ".join(parts)
    parts = []
    for k, v in row_dict.items():
        if v is None or (isinstance(v, float) and str(v) == "nan"):
            continue
        parts.append(str(v))
    return " ".join(parts)
