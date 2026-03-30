"""Shared utilities for privacy scoring and text masking."""

import numpy as np

MAX_LENGTH = 2048

LABEL_MAP = {f"LABEL_{i}": i + 1 for i in range(5)}
for i in range(1, 6):
    LABEL_MAP[str(i)] = i


def _parse_label(label: str) -> int:
    """Convert a pipeline output label to a privacy rating (1-5)."""
    if label.isdigit():
        rating = int(label)
        if 1 <= rating <= 5:
            return rating
    return LABEL_MAP.get(label, 3)


def predict_rating(classifier, text: str) -> int:
    """Get a privacy rating (1-5) from a single text."""
    result = classifier(text, truncation=True, max_length=MAX_LENGTH)[0]
    return _parse_label(result["label"])


def score_texts(classifier, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Score a list of texts and return an array of ratings (1-5)."""
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = classifier(batch, truncation=True, max_length=MAX_LENGTH, batch_size=batch_size)
        scores.extend(_parse_label(r["label"]) for r in results)
    return np.array(scores)


def mask_entities(text: str, entities: list[dict], mask_types: set[str]) -> str:
    """Replace entity spans whose identifier_type is in *mask_types* with [REDACTED]."""
    to_mask = [e for e in entities if e.get("identifier_type") in mask_types]
    if not to_mask:
        return text
    to_mask.sort(key=lambda e: e["start_offset"], reverse=True)
    for e in to_mask:
        start, end = e["start_offset"], e["end_offset"]
        if 0 <= start < end <= len(text):
            text = text[:start] + "[REDACTED]" + text[end:]
    return text


def randomly_mask_text(text: str, fraction: float = 0.3, seed: int = 42) -> str:
    """Replace approximately *fraction* of words with [REDACTED]."""
    rng = np.random.default_rng(seed)
    words = text.split()
    if not words:
        return text
    n = max(1, int(len(words) * fraction))
    for i in rng.choice(len(words), size=min(n, len(words)), replace=False):
        words[i] = "[REDACTED]"
    return " ".join(words)
