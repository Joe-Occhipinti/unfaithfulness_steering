"""
data.py

Data loading and processing utilities for faithfulness steering workflow.
Reusable across baseline, hinted, and steering evaluation scripts.
"""

import json
import os
from typing import Dict, Any, List
from datasets import load_dataset

def load_mmlu_simple(subjects: List[str]) -> List[Dict[str, Any]]:
    """
    Dead simple MMLU loader - just load subjects, all splits.
    Reusable across baseline and hinted evaluation.

    Args:
        subjects: List of MMLU subject names (e.g., ['high_school_psychology'])

    Returns:
        List of dictionaries with question, choices, answer, subject, split
    """
    print(f"\n--- Loading MMLU data (simple) ---")
    all_data = []

    for subject in subjects:
        print(f"Loading {subject}...")
        try:
            dataset = load_dataset("cais/mmlu", subject)

            for split_name, split_data in dataset.items():
                print(f"  {split_name}: {len(split_data)} questions")
                for item in split_data:
                    all_data.append({
                        'question': item['question'],
                        'choices': item['choices'],
                        'answer': item['answer'],  # This is 0,1,2,3
                        'subject': subject,
                        'split': split_name
                    })
        except Exception as e:
            print(f"Error loading {subject}: {e}")
            continue

    print(f"Total loaded: {len(all_data)} questions from {len(subjects)} subjects")
    return all_data

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    Reusable across all evaluation scripts.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries loaded from file
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to JSONL file.
    Reusable across all evaluation scripts.

    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def convert_answer_to_letter(answer_idx: int) -> str:
    """
    Convert MMLU answer index to letter.
    Reusable across all evaluation scripts.

    Args:
        answer_idx: Answer index (0, 1, 2, 3)

    Returns:
        Answer letter (A, B, C, D)
    """
    return ['A', 'B', 'C', 'D'][answer_idx]

def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.3,
    test_ratio: float = 0.0,
    seed: int = 42,
    filter_field: str = None,
    filter_value: Any = None
) -> tuple[List[Dict[str, Any]], ...]:
    """
    Split data into train, validation, and optionally test sets with optional filtering.

    General-purpose splitting function that works with any list of dictionaries.
    Applies consistent randomization (seed 42 by default) and splitting strategy
    used across the faithfulness steering pipeline.

    Args:
        data: List of dictionaries (typically loaded from JSONL)
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.3)
        test_ratio: Proportion of data for test (default: 0.0)
        seed: Random seed for reproducibility (default: 42)
        filter_field: Optional field name to filter by (e.g., 'faithfulness_classification')
        filter_value: Value to filter for (e.g., 'unfaithful')

    Returns:
        Tuple of (train_data, val_data) if test_ratio is 0
        Tuple of (train_data, val_data, test_data) if test_ratio > 0

    Example:
        >>> # Basic train/val split (70/30)
        >>> train, val = split_data(annotated_data)
        >>>
        >>> # Train/val/test split (70/15/15)
        >>> train, val, test = split_data(annotated_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        >>>
        >>> # Split and filter for unfaithful examples
        >>> train, val = split_data(
        ...     annotated_data,
        ...     train_ratio=0.7,
        ...     val_ratio=0.3,
        ...     filter_field='faithfulness_classification',
        ...     filter_value='unfaithful'
        ... )
    """
    import random

    # Validate ratios sum to 1.0 (with tolerance for floating point)
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Randomize with seed
    random.seed(seed)
    shuffled_indices = list(range(len(data)))
    random.shuffle(shuffled_indices)

    # Calculate split sizes
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    # test_size is whatever remains to avoid rounding issues

    # Split indices
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    # Extract splits
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices] if test_ratio > 0 else []

    # Apply optional filtering
    if filter_field is not None and filter_value is not None:
        train_data = [item for item in train_data if item.get(filter_field) == filter_value]
        val_data = [item for item in val_data if item.get(filter_field) == filter_value]
        if test_ratio > 0:
            test_data = [item for item in test_data if item.get(filter_field) == filter_value]

    # Return appropriate tuple based on whether test split exists
    if test_ratio > 0:
        return train_data, val_data, test_data
    else:
        return train_data, val_data