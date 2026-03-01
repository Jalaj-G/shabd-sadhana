"""
Metrics module for ASR evaluation.

Provides functions to compute Word Error Rate (WER) and related metrics
for evaluating automatic speech recognition model quality.
"""

import re
from typing import List, Dict
import jiwer


def normalize_text(text: str) -> str:
    """
    Normalize text for fair WER comparison.

    Whisper models often ignore punctuation and casing, so we normalize
    both reference and hypothesis texts to ensure WER measures actual
    transcription accuracy rather than formatting differences.

    Normalization steps:
    - Convert to lowercase
    - Remove punctuation (except apostrophes in contractions)
    - Normalize whitespace (collapse multiple spaces, strip edges)

    Args:
        text: Input text string to normalize

    Returns:
        Normalized text string

    Examples:
        >>> normalize_text("Hello, World!")
        "hello world"
        >>> normalize_text("It's a test.  Multiple   spaces.")
        "it's a test multiple spaces"
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation except apostrophes (to preserve contractions like "it's", "don't")
    # This pattern keeps alphanumeric characters, spaces, and apostrophes
    text = re.sub(r"[^a-z0-9\s']", "", text)

    # Normalize whitespace: collapse multiple spaces and strip edges
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_wer(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Compute Word Error Rate (WER) using the jiwer library.

    WER is the industry-standard metric for ASR evaluation, calculated as:
    WER = (Substitutions + Deletions + Insertions) / Total Words in Reference

    The function normalizes both reference and prediction texts before
    computing WER to ensure fair comparison (lowercase, remove punctuation).

    Args:
        references: List of ground truth transcripts
        predictions: List of model-generated transcripts

    Returns:
        Dictionary containing:
        - "wer": Word Error Rate as a percentage (0-100+)
                Note: WER can exceed 100% if there are many insertions

    Raises:
        ValueError: If references and predictions have different lengths

    Examples:
        >>> references = ["hello world", "this is a test"]
        >>> predictions = ["hello world", "this is test"]
        >>> result = compute_wer(references, predictions)
        >>> print(result)
        {'wer': 12.5}  # 1 deletion out of 8 total words
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"References and predictions must have same length. "
            f"Got {len(references)} references and {len(predictions)} predictions."
        )

    # Handle edge case: empty lists
    if len(references) == 0:
        return {"wer": 0.0}

    # Normalize all texts
    normalized_refs = [normalize_text(ref) for ref in references]
    normalized_preds = [normalize_text(pred) for pred in predictions]

    # Filter out empty strings (can occur after normalization)
    # Keep track of indices to align references and predictions
    valid_pairs = [
        (ref, pred)
        for ref, pred in zip(normalized_refs, normalized_preds)
        if ref.strip()  # Only keep if reference is non-empty
    ]

    # If all references are empty, return 0 WER
    if not valid_pairs:
        return {"wer": 0.0}

    filtered_refs, filtered_preds = zip(*valid_pairs)

    # Compute WER using jiwer
    # jiwer.wer returns a float between 0 and infinity (can exceed 1 with many insertions)
    wer_score = jiwer.wer(
        reference=list(filtered_refs),
        hypothesis=list(filtered_preds)
    )

    # Convert to percentage and return
    return {"wer": wer_score * 100}


def compute_wer_batch(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Alias for compute_wer for batch processing.

    This function is identical to compute_wer but has a more explicit name
    for use in training loops where batch evaluation is performed.

    Args:
        references: List of ground truth transcripts
        predictions: List of model-generated transcripts

    Returns:
        Dictionary containing:
        - "wer": Word Error Rate as a percentage (0-100+)
    """
    return compute_wer(references, predictions)
