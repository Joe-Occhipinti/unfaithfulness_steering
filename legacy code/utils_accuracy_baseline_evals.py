#!/usr/bin/env python3
"""
Baseline Accuracy Calculator

This script calculates accuracy by comparing extracted letters from val_baseline_correct_results
with ground truth answers from the source dataset.

Usage:
    python calculate_baseline_accuracy.py

Configuration:
    - RESULTS_FILE: Path to file with extracted letters (field: letter)
    - GROUND_TRUTH_FILE: Path to source dataset with correct answers (field: answer)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def load_ground_truth(ground_truth_file: str) -> Dict[int, str]:
    """
    Load ground truth answers indexed by line number.
    
    Args:
        ground_truth_file: Path to JSONL file with 'answer' field
        
    Returns:
        Dictionary mapping line numbers (0-indexed) to answer letters
    """
    ground_truth = {}
    ground_truth_path = Path(ground_truth_file)
    
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
    
    print(f"Loading ground truth from: {ground_truth_file}")
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                answer = data.get('answer', '')
                
                # Convert numeric answer to letter if needed
                if isinstance(answer, int) and answer in [0, 1, 2, 3]:
                    answer = ['A', 'B', 'C', 'D'][answer]
                elif isinstance(answer, str):
                    answer = answer.strip().upper()
                
                if answer in ['A', 'B', 'C', 'D']:
                    ground_truth[line_num] = answer
                else:
                    print(f"Warning: Invalid answer '{answer}' at line {line_num + 1}")
            except json.JSONDecodeError as e:
                print(f"Error parsing ground truth line {line_num + 1}: {e}")
    
    print(f"Loaded {len(ground_truth)} ground truth answers")
    return ground_truth

def load_predictions(results_file: str) -> Dict[int, str]:
    """
    Load predicted letters indexed by line number.
    
    Args:
        results_file: Path to JSONL file with 'letter' field
        
    Returns:
        Dictionary mapping line numbers (0-indexed) to predicted letters
    """
    predictions = {}
    results_path = Path(results_file)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    print(f"Loading predictions from: {results_file}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                letter = data.get('letter', '')
                if letter:
                    predictions[line_num] = letter.strip().upper()
                else:
                    print(f"Warning: No letter extracted at line {line_num + 1}")
            except json.JSONDecodeError as e:
                print(f"Error parsing results line {line_num + 1}: {e}")
    
    print(f"Loaded {len(predictions)} predictions")
    return predictions

def calculate_accuracy(ground_truth: Dict[int, str], predictions: Dict[int, str]) -> Tuple[float, int, int, int]:
    """
    Calculate accuracy between ground truth and predictions.
    
    Args:
        ground_truth: Dictionary of ground truth answers
        predictions: Dictionary of predicted letters
        
    Returns:
        Tuple of (accuracy, correct_count, valid_comparisons, missing_predictions)
    """
    correct_count = 0
    valid_comparisons = 0
    missing_predictions = 0
    
    print("\nCalculating accuracy...")
    print("=" * 60)
    
    # Check all ground truth entries
    for line_num in sorted(ground_truth.keys()):
        ground_truth_answer = ground_truth[line_num]
        
        if line_num in predictions:
            predicted_answer = predictions[line_num]
            valid_comparisons += 1
            is_correct = ground_truth_answer == predicted_answer
            
            if is_correct:
                correct_count += 1
            
            status = "OK" if is_correct else "FAIL"
            if valid_comparisons <= 20 or not is_correct:  # Show first 20 or all errors
                print(f"Line {line_num + 1:3d}: {status} GT: {ground_truth_answer} | Pred: {predicted_answer}")
        else:
            missing_predictions += 1
            print(f"Line {line_num + 1:3d}: FAIL GT: {ground_truth_answer} | Pred: MISSING")
    
    accuracy = correct_count / valid_comparisons if valid_comparisons > 0 else 0.0
    
    return accuracy, correct_count, valid_comparisons, missing_predictions

def print_summary(accuracy: float, correct_count: int, valid_comparisons: int, missing_predictions: int, total_ground_truth: int):
    """Print accuracy summary."""
    print("=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    print(f"Correct predictions: {correct_count}")
    print(f"Valid comparisons: {valid_comparisons}")
    print(f"Total ground truth: {total_ground_truth}")
    print(f"Missing predictions: {missing_predictions}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
    print(f"Coverage: {valid_comparisons/total_ground_truth:.3f} ({valid_comparisons/total_ground_truth * 100:.1f}%)")
    print("=" * 60)

def main():
    """Main function."""
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Results file with extracted letters
    RESULTS_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_baseline_correct_results_2025-08-12.jsonl.jsonl"
    
    # Ground truth file with correct answers
    GROUND_TRUTH_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_validation_high_school_psychology_all_2025-08-12_prompts_baseline.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Baseline Accuracy Calculator")
    print("=" * 60)
    
    try:
        # Load data
        ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
        predictions = load_predictions(RESULTS_FILE)
        
        if not ground_truth:
            print("Error: No ground truth data loaded")
            return
        
        if not predictions:
            print("Error: No predictions loaded")
            return
        
        # Calculate accuracy
        accuracy, correct_count, valid_comparisons, missing_predictions = calculate_accuracy(ground_truth, predictions)
        
        # Print summary
        print_summary(accuracy, correct_count, valid_comparisons, missing_predictions, len(ground_truth))
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()