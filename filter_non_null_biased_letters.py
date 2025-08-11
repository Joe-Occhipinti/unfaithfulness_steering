#!/usr/bin/env python3
"""
Filter Non-Null Biased Letters

This script removes all rows with null values in the 'extracted_biased_letter' field
from the merged dataset.

Usage:
    python filter_non_null_biased_letters.py

Configuration:
    - INPUT_FILE: Path to merged dataset
    - OUTPUT_FILE: Path to filtered dataset (only non-null extracted_biased_letter)
"""

import json
from pathlib import Path
from datetime import datetime

def filter_non_null_biased_letters(input_file: str, output_file: str) -> None:
    """
    Filter dataset to keep only rows with non-null extracted_biased_letter values.
    
    Args:
        input_file: Path to input dataset
        output_file: Path to filtered output dataset
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("-" * 60)
    
    total_entries = 0
    kept_entries = 0
    null_entries = 0
    invalid_entries = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                # Check extracted_biased_letter field
                extracted_biased_letter = data.get('extracted_biased_letter')
                
                # Filter out null, None, empty string, or invalid values
                if extracted_biased_letter is None:
                    null_entries += 1
                    continue
                
                # Convert to string and check if it's a valid letter
                letter_str = str(extracted_biased_letter).upper().strip()
                
                if not letter_str or letter_str in ['NULL', 'NONE', '']:
                    null_entries += 1
                    continue
                
                if letter_str not in ['A', 'B', 'C', 'D']:
                    invalid_entries += 1
                    if invalid_entries <= 5:  # Show first 5 invalid entries
                        print(f"Line {line_num}: Invalid letter '{extracted_biased_letter}', skipping")
                    continue
                
                # Keep this entry - it has a valid extracted_biased_letter
                data['filter_applied'] = 'non_null_biased_letter_only'
                data['filter_timestamp'] = datetime.now().isoformat()
                
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                outfile.flush()
                
                kept_entries += 1
                
                if kept_entries % 100 == 0:
                    print(f"Kept {kept_entries} entries so far...")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                invalid_entries += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                invalid_entries += 1
    
    # Calculate statistics
    filtered_out = total_entries - kept_entries
    retention_rate = kept_entries / total_entries * 100 if total_entries > 0 else 0
    
    print("-" * 60)
    print("Filtering complete!")
    print(f"Total entries processed: {total_entries}")
    print(f"Entries kept: {kept_entries}")
    print(f"Entries filtered out: {filtered_out}")
    print(f"  - Null/empty values: {null_entries}")
    print(f"  - Invalid letters: {invalid_entries}")
    print(f"Retention rate: {retention_rate:.2f}%")
    print(f"Filtered dataset saved to: {output_file}")

def preview_filtered_dataset(output_file: str, num_examples: int = 3) -> None:
    """Preview the filtered dataset."""
    output_path = Path(output_file)
    
    if not output_path.exists():
        print("Filtered dataset not found.")
        return
    
    print("\n" + "=" * 60)
    print("FILTERED DATASET PREVIEW")
    print("=" * 60)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > num_examples:
                break
            
            try:
                data = json.loads(line.strip())
                
                print(f"\nEntry {i}:")
                print(f"  Question: {data.get('question', 'N/A')}")
                print(f"  Original letter: {data.get('extracted_letter', 'N/A')}")
                print(f"  Biased letter: {data.get('extracted_biased_letter', 'N/A')}")
                print(f"  Ground truth: {data.get('answer', 'N/A')}")
                
            except json.JSONDecodeError as e:
                print(f"Error reading example {i}: {e}")
    
    # Count total entries
    with open(output_path, 'r', encoding='utf-8') as f:
        total_filtered = sum(1 for _ in f)
    
    print(f"\nTotal entries with valid extracted_biased_letter: {total_filtered}")
    print("=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # First, I need to check if the merged dataset has extracted_biased_letter field
    # If not, we need to run letter extraction first
    
    INPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_correct_with_biased_extracted.jsonl"
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_correct_with_biased_filtered.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Non-Null Biased Letters Filter")
    print("=" * 60)
    
    try:
        # Check if extracted_biased_letter field exists
        input_path = Path(INPUT_FILE)
        if input_path.exists():
            with open(input_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    sample_data = json.loads(first_line.strip())
                    if 'extracted_biased_letter' not in sample_data:
                        print("ERROR: 'extracted_biased_letter' field not found in dataset!")
                        print("Please run letter extraction on the biased answers first.")
                        print("Available fields:", list(sample_data.keys()))
                        exit(1)
        
        # Filter the dataset
        filter_non_null_biased_letters(INPUT_FILE, OUTPUT_FILE)
        
        # Preview the results
        preview_filtered_dataset(OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Filtering completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")