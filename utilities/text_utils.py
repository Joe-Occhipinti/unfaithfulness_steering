"""
text_utils.py

Text processing utilities for analyzing and filtering generated texts.
Reusable across filtering and analysis scripts.
"""

import re
from typing import List


def count_words(text: str) -> int:
    """
    Count words in a text string.

    Uses regex to find word boundaries, matching the pattern used in
    measure_hinted_lengths.py for consistency.

    Args:
        text: Input text string

    Returns:
        Number of words in the text

    Example:
        >>> count_words("Hello world!")
        2
        >>> count_words("This is a test.")
        4
    """
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def get_word_lengths(texts: List[str]) -> List[int]:
    """
    Get word counts for a list of texts.

    Args:
        texts: List of text strings

    Returns:
        List of word counts corresponding to each text

    Example:
        >>> get_word_lengths(["Hello world", "Test"])
        [2, 1]
    """
    return [count_words(text) for text in texts]
