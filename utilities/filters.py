"""
filters.py

Filtering utilities for dataset processing.
Supports filtering records based on text length and other criteria.
"""

import statistics
from typing import List, Dict, Any, Optional
from src.utilities.text_utils import count_words


def calculate_median_word_length(
    data: List[Dict[str, Any]],
    text_field: str
) -> float:
    """
    Calculate the median word length from a specific field across all records.

    Args:
        data: List of dictionaries (records from JSONL)
        text_field: Name of the field containing text to measure

    Returns:
        Median word count as a float

    Raises:
        ValueError: If no valid texts are found in the specified field

    Example:
        >>> data = [
        ...     {"text": "Hello world"},
        ...     {"text": "This is a test"},
        ...     {"text": "Short"}
        ... ]
        >>> calculate_median_word_length(data, "text")
        2.0
    """
    word_lengths = []

    for record in data:
        text = record.get(text_field, '')
        if text:
            word_lengths.append(count_words(text))

    if not word_lengths:
        raise ValueError(f"No valid texts found in field '{text_field}'")

    return statistics.median(word_lengths)


def filter_by_text_length(
    data: List[Dict[str, Any]],
    text_field: str,
    max_words: int
) -> List[Dict[str, Any]]:
    """
    Filter records where the specified text field has word count <= max_words.

    All other fields in each record are preserved in the output.

    Args:
        data: List of dictionaries (records from JSONL)
        text_field: Name of the field containing text to filter by
        max_words: Maximum word count threshold (inclusive)

    Returns:
        Filtered list of records meeting the length criteria

    Example:
        >>> data = [
        ...     {"id": 1, "text": "Hello world", "other": "data"},
        ...     {"id": 2, "text": "This is a much longer text", "other": "more"}
        ... ]
        >>> filtered = filter_by_text_length(data, "text", 3)
        >>> len(filtered)
        1
        >>> filtered[0]["id"]
        1
    """
    filtered_data = []

    for record in data:
        text = record.get(text_field, '')
        if text and count_words(text) <= max_words:
            filtered_data.append(record)

    return filtered_data


def get_length_statistics(
    data: List[Dict[str, Any]],
    text_field: str
) -> Dict[str, float]:
    """
    Calculate length statistics for a text field across all records.

    Args:
        data: List of dictionaries (records from JSONL)
        text_field: Name of the field containing text to analyze

    Returns:
        Dictionary with 'mean', 'median', 'min', 'max' word counts

    Example:
        >>> data = [
        ...     {"text": "Hello world"},
        ...     {"text": "This is a test"},
        ...     {"text": "Short"}
        ... ]
        >>> stats = get_length_statistics(data, "text")
        >>> stats['median']
        2.0
    """
    word_lengths = []

    for record in data:
        text = record.get(text_field, '')
        if text:
            word_lengths.append(count_words(text))

    if not word_lengths:
        return {'mean': 0, 'median': 0, 'min': 0, 'max': 0}

    return {
        'mean': statistics.mean(word_lengths),
        'median': statistics.median(word_lengths),
        'min': min(word_lengths),
        'max': max(word_lengths)
    }
