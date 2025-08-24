import re
import unicodedata
from typing import Callable, Iterable, Optional, Set


def default_english_normalizer(case: str = "auto") -> Callable[[str], str]:
    """
    Normalize transcripts for CTC:
    - NFKD unicode normalize, strip diacritics
    - Case handling: "auto" (no change), "lower", or "upper"
    - Remove punctuation except internal apostrophes
    - Collapse spaces
    """
    punct_to_space = r"[,;:()\[\]{}!?\"“”‘’—–\-…/\\`~@#$%^&*_+=<>|]"
    apostrophe_rule = r"(?<!\w)'|'(?!\w)"  # keep apostrophes between letters

    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        if case == "lower":
            text = text.lower()
        elif case == "upper":
            text = text.upper()
        # else: "auto" -> leave case as-is
        text = re.sub(punct_to_space, " ", text)
        text = re.sub(apostrophe_rule, " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    return normalize


def whitelist_normalizer(allowed: Iterable[str],
                         base: Optional[Callable[[str], str]] = None) -> Callable[[str], str]:
    """
    Further restrict characters to those in tokenizer vocab.
    Always allow a literal space ' ' so the tokenizer can map it to its delimiter (e.g., '|').
    """
    allowed_set: Set[str] = set(allowed)
    base = base or (lambda x: x)

    def normalize(text: str) -> str:
        t = base(text)
        return "".join(ch for ch in t if ch in allowed_set or ch == " ")
    return normalize
