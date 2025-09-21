#!/usr/bin/env python3
"""
Accuracy Calculator

This script calculates how many times the model's extracted letter choice matches
the ground truth answer. It converts between letter format (A,B,C,D) and 
numeric format (0,1,2,3) for comparison.

Usage:
    python calculate_accuracy.py

Configuration:
    - INPUT_FILE: Path to merged JSONL file with both 'answer' and 'extracted_letter' fields
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import Counter

def letter_to_number(letter: str) -> Optional[int]:
    """Convert letter choice (A,B,C,D) to number (0,1,2,3)."""
    if not letter or not isinstance(letter, str):
        return None
    letter = letter.upper().strip()
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return mapping.get(letter)

def number_to_letter(number: int) -> Optional[str]:
    """Convert number (0,1,2,3) to letter choice (A,B,C,D)."""
    if number not in [0, 1, 2, 3]:
        return None
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    return mapping.get(number)

def calculate_accuracy(input_file: str) -> Dict[str, Any]:
    """
    Calculate accuracy statistics from the merged JSONL file.
    
    Args:
        input_file: Path to JSONL file with 'answer' and 'extracted_letter' fields
        
    Returns:
        Dictionary with accuracy statistics
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Processing: {input_file}")
    print("-" * 60)
    
    total_entries = 0
    valid_comparisons = 0
    correct_matches = 0
    missing_answers = 0
    missing_extracted_letters = 0
    invalid_answers = 0
    invalid_extracted_letters = 0
    
    # Track distribution of answers and predictions
    answer_distribution = Counter()
    prediction_distribution = Counter()
    confusion_matrix = {i: {j: 0 for j in range(4)} for i in range(4)}
    
    # Store detailed results for analysis
    results = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                # Extract ground truth answer
                ground_truth = data.get('answer')
                extracted_letter = data.get('extracted_letter')
                
                # Track missing data
                if ground_truth is None:
                    missing_answers += 1
                    print(f"Line {line_num}: Missing ground truth answer")
                    continue
                    
                if extracted_letter is None:
                    missing_extracted_letters += 1
                    if line_num % 50 == 0:  # Don't spam too much
                        print(f"Line {line_num}: Missing extracted letter")
                    continue
                
                # Validate ground truth (should be 0,1,2,3)
                if not isinstance(ground_truth, int) or ground_truth not in [0, 1, 2, 3]:
                    invalid_answers += 1
                    print(f"Line {line_num}: Invalid ground truth answer: {ground_truth}")
                    continue
                
                # Convert extracted letter to number
                predicted_number = letter_to_number(extracted_letter)
                if predicted_number is None:
                    invalid_extracted_letters += 1
                    if line_num % 50 == 0:  # Don't spam too much
                        print(f"Line {line_num}: Invalid extracted letter: {extracted_letter}")
                    continue
                
                # Valid comparison
                valid_comparisons += 1
                
                # Check if correct
                is_correct = (ground_truth == predicted_number)
                if is_correct:
                    correct_matches += 1
                
                # Update distributions
                answer_distribution[ground_truth] += 1
                prediction_distribution[predicted_number] += 1
                confusion_matrix[ground_truth][predicted_number] += 1
                
                # Store result
                results.append({
                    'line_num': line_num,
                    'ground_truth': ground_truth,
                    'ground_truth_letter': number_to_letter(ground_truth),
                    'predicted_number': predicted_number,
                    'predicted_letter': extracted_letter,
                    'correct': is_correct,
                    'question': data.get('question', '')[:100] + '...' if len(data.get('question', '')) > 100 else data.get('question', '')
                })
                
                # Progress update
                if line_num % 100 == 0:
                    current_accuracy = correct_matches / valid_comparisons * 100 if valid_comparisons > 0 else 0
                    print(f"Processed {line_num} lines... Current accuracy: {current_accuracy:.1f}%")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    # Calculate final statistics
    accuracy = correct_matches / valid_comparisons * 100 if valid_comparisons > 0 else 0
    
    statistics = {
        'total_entries': total_entries,
        'valid_comparisons': valid_comparisons,
        'correct_matches': correct_matches,
        'accuracy_percentage': accuracy,
        'missing_answers': missing_answers,
        'missing_extracted_letters': missing_extracted_letters,
        'invalid_answers': invalid_answers,
        'invalid_extracted_letters': invalid_extracted_letters,
        'answer_distribution': dict(answer_distribution),
        'prediction_distribution': dict(prediction_distribution),
        'confusion_matrix': confusion_matrix,
        'sample_errors': []
    }
    
    # Add sample errors for analysis
    error_samples = [r for r in results if not r['correct']][:10]  # First 10 errors
    statistics['sample_errors'] = error_samples
    
    return statistics

def print_detailed_results(stats: Dict[str, Any]) -> None:
    """Print detailed accuracy results."""
    print("=" * 60)
    print("ACCURACY RESULTS")
    print("=" * 60)
    
    print(f"Total entries processed: {stats['total_entries']}")
    print(f"Valid comparisons: {stats['valid_comparisons']}")
    print(f"Correct matches: {stats['correct_matches']}")
    print(f"ACCURACY: {stats['accuracy_percentage']:.2f}%")
    print()
    
    print("Data Quality:")
    print(f"  Missing ground truth answers: {stats['missing_answers']}")
    print(f"  Missing extracted letters: {stats['missing_extracted_letters']}")
    print(f"  Invalid ground truth answers: {stats['invalid_answers']}")
    print(f"  Invalid extracted letters: {stats['invalid_extracted_letters']}")
    print()
    
    print("Answer Distribution (Ground Truth):")
    for answer_num in [0, 1, 2, 3]:
        count = stats['answer_distribution'].get(answer_num, 0)
        letter = number_to_letter(answer_num)
        percentage = count / stats['valid_comparisons'] * 100 if stats['valid_comparisons'] > 0 else 0
        print(f"  {answer_num} ({letter}): {count} ({percentage:.1f}%)")
    print()
    
    print("Prediction Distribution (Model Choices):")
    for pred_num in [0, 1, 2, 3]:
        count = stats['prediction_distribution'].get(pred_num, 0)
        letter = number_to_letter(pred_num)
        percentage = count / stats['valid_comparisons'] * 100 if stats['valid_comparisons'] > 0 else 0
        print(f"  {pred_num} ({letter}): {count} ({percentage:.1f}%)")
    print()
    
    print("Confusion Matrix (Ground Truth vs Predicted):")
    print("     A    B    C    D")
    for gt in [0, 1, 2, 3]:
        row = [stats['confusion_matrix'][gt][pred] for pred in [0, 1, 2, 3]]
        gt_letter = number_to_letter(gt)
        print(f"{gt_letter} {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}")
    print()
    
    if stats['sample_errors']:
        print("Sample Errors (first 10):")
        for i, error in enumerate(stats['sample_errors'], 1):
            print(f"{i:2d}. Line {error['line_num']}: GT={error['ground_truth_letter']} Pred={error['predicted_letter']} | {error['question']}")
    
    print("=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input file (merged JSONL with both answer and extracted_letter fields)
    INPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_test_only_merged_2025-08-10.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Model Accuracy Calculator")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print("=" * 60)
    
    try:
        # Calculate accuracy statistics
        results = calculate_accuracy(INPUT_FILE)
        
        # Print detailed results
        print_detailed_results(results)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")