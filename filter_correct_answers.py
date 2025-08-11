#!/usr/bin/env python3
"""
Correct Answers Filter

This script creates a new dataset containing only the rows where the model
answered correctly (extracted_letter matches the ground truth answer).

Usage:
    python filter_correct_answers.py

Configuration:
    - INPUT_FILE: Path to merged JSONL file with both 'answer' and 'extracted_letter' fields
    - OUTPUT_FILE: Path to output JSONL file with only correct answers
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

def letter_to_number(letter: str) -> Optional[int]:
    """Convert letter choice (A,B,C,D) to number (0,1,2,3)."""
    if not letter or not isinstance(letter, str):
        return None
    letter = letter.upper().strip()
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return mapping.get(letter)

def filter_correct_answers(input_file: str, output_file: str) -> None:
    """
    Filter dataset to include only entries where the model answered correctly.
    
    Args:
        input_file: Path to merged JSONL file
        output_file: Path to output JSONL file with only correct answers
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("-" * 60)
    
    total_entries = 0
    correct_entries = 0
    skipped_entries = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                # Get ground truth and prediction
                ground_truth = data.get('answer')
                extracted_letter = data.get('extracted_letter')
                
                # Skip if missing data
                if ground_truth is None or extracted_letter is None:
                    skipped_entries += 1
                    if skipped_entries <= 5:  # Only show first 5 warnings
                        print(f"Line {line_num}: Skipping - missing data (GT: {ground_truth}, Pred: {extracted_letter})")
                    continue
                
                # Validate ground truth
                if not isinstance(ground_truth, int) or ground_truth not in [0, 1, 2, 3]:
                    skipped_entries += 1
                    print(f"Line {line_num}: Skipping - invalid ground truth: {ground_truth}")
                    continue
                
                # Convert letter to number
                predicted_number = letter_to_number(extracted_letter)
                if predicted_number is None:
                    skipped_entries += 1
                    if skipped_entries <= 10:  # Show more letter errors since they're important
                        print(f"Line {line_num}: Skipping - invalid extracted letter: {extracted_letter}")
                    continue
                
                # Check if answer is correct
                if ground_truth == predicted_number:
                    # Add metadata about filtering
                    data['filtered_as'] = 'correct_answer'
                    data['filter_timestamp'] = datetime.now().isoformat()
                    
                    # Write to output file
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    outfile.flush()
                    
                    correct_entries += 1
                    
                    # Progress update
                    if correct_entries % 50 == 0:
                        print(f"Found {correct_entries} correct answers so far...")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                skipped_entries += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_entries += 1
    
    # Calculate statistics
    accuracy = correct_entries / (total_entries - skipped_entries) * 100 if (total_entries - skipped_entries) > 0 else 0
    
    print("-" * 60)
    print("Filtering complete!")
    print(f"Total entries processed: {total_entries}")
    print(f"Correct answers found: {correct_entries}")
    print(f"Skipped entries: {skipped_entries}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Filtered dataset saved to: {output_file}")
    print("-" * 60)

def preview_correct_dataset(output_file: str, num_examples: int = 3) -> None:
    """Preview the filtered dataset."""
    output_path = Path(output_file)
    
    if not output_path.exists():
        print("Filtered dataset not found. Please run the filtering first.")
        return
    
    print("\nPreview of filtered dataset (correct answers only):")
    print("=" * 60)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > num_examples:
                break
            
            try:
                data = json.loads(line.strip())
                question = data.get('question', 'N/A')
                choices = data.get('choices', [])
                answer_idx = data.get('answer', -1)
                extracted_letter = data.get('extracted_letter', 'N/A')
                
                correct_choice = choices[answer_idx] if 0 <= answer_idx < len(choices) else 'N/A'
                
                print(f"Example {i}:")
                print(f"  Question: {question}")
                print(f"  Correct Answer: {answer_idx} ({extracted_letter}) - {correct_choice}")
                print()
                
            except json.JSONDecodeError as e:
                print(f"Error reading example {i}: {e}")
    
    # Count total entries
    with open(output_path, 'r', encoding='utf-8') as f:
        total_correct = sum(1 for _ in f)
    
    print(f"Total correct answers in dataset: {total_correct}")
    print("=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input file (merged dataset with ground truth and predictions)
    INPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_test_only_merged_2025-08-10.jsonl"
    
    # Output file (filtered dataset with only correct answers)
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_correct_answers_only.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Correct Answers Filter")
    print("=" * 60)
    
    try:
        # Filter the dataset
        filter_correct_answers(INPUT_FILE, OUTPUT_FILE)
        
        # Preview the results
        preview_correct_dataset(OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Filtering completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")