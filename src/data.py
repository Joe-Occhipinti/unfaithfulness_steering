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