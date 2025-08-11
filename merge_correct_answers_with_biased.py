#!/usr/bin/env python3
"""
Merge Correct Answers with Biased Dataset

This script merges the correct answers dataset with the biased results dataset,
preserving all unique fields from both files.

Usage:
    python merge_correct_answers_with_biased.py

Configuration:
    - FILE1: Correct answers only dataset
    - FILE2: Biased results dataset
    - OUTPUT_FILE: Merged output dataset
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Set

def merge_datasets(file1: str, file2: str, output_file: str) -> None:
    """
    Merge two JSONL datasets line by line, preserving all unique fields.
    
    Args:
        file1: Path to first dataset (correct answers)
        file2: Path to second dataset (biased results)
        output_file: Path to output merged dataset
    """
    file1_path = Path(file1)
    file2_path = Path(file2)
    output_path = Path(output_file)
    
    if not file1_path.exists():
        raise FileNotFoundError(f"File1 not found: {file1}")
    if not file2_path.exists():
        raise FileNotFoundError(f"File2 not found: {file2}")
    
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"File1 (correct answers): {file1}")
    print(f"File2 (biased results): {file2}")
    print(f"Output file: {output_file}")
    print("-" * 60)
    
    # Load second dataset into memory for easier merging
    print("Loading biased results dataset...")
    dataset2 = []
    with open(file2_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                dataset2.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing biased line {line_num}: {e}")
                dataset2.append({})
    
    print(f"Loaded {len(dataset2)} biased entries")
    
    # Track field usage for analysis
    all_fields = set()
    field_conflicts = 0
    merged_count = 0
    
    # Process first dataset and merge with corresponding entries from second
    print("Merging datasets...")
    with open(file1_path, 'r', encoding='utf-8') as f1, \
         open(output_path, 'w', encoding='utf-8') as out_f:
        
        for line_num, line in enumerate(f1, 1):
            try:
                # Parse entry from first dataset
                data1 = json.loads(line.strip())
                
                # Start with data from first file
                merged_entry = data1.copy()
                
                # Merge with corresponding entry from second dataset if available
                if line_num <= len(dataset2):
                    data2 = dataset2[line_num - 1]
                    
                    # Add all fields from second dataset, checking for conflicts
                    for key, value in data2.items():
                        if key in merged_entry:
                            # Check if values are different
                            if merged_entry[key] != value:
                                # Handle conflict by preserving both
                                if key.startswith('generated_'):
                                    # Rename conflicting generated fields
                                    new_key = key.replace('generated_', 'generated_biased_')
                                    merged_entry[new_key] = value
                                elif key.startswith('extracted_'):
                                    # Rename conflicting extracted fields  
                                    new_key = key.replace('extracted_', 'extracted_biased_')
                                    merged_entry[new_key] = value
                                else:
                                    # For other conflicts, add with suffix
                                    merged_entry[f"{key}_biased"] = value
                                field_conflicts += 1
                            # If values are the same, keep the original
                        else:
                            # No conflict, add the field
                            merged_entry[key] = value
                        
                        all_fields.add(key)
                else:
                    print(f"Warning: No biased data for line {line_num}")
                
                # Track all fields from first dataset too
                all_fields.update(data1.keys())
                
                # Add merge metadata
                merged_entry['merge_timestamp'] = datetime.now().isoformat()
                merged_entry['merge_type'] = 'correct_answers_with_biased'
                
                merged_count += 1
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines...")
                
                # Write merged entry
                out_f.write(json.dumps(merged_entry, ensure_ascii=False) + '\n')
                out_f.flush()
                
            except json.JSONDecodeError as e:
                print(f"Error parsing correct answers line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print("-" * 60)
    print("Merge complete!")
    print(f"Successfully merged: {merged_count} entries")
    print(f"Field conflicts handled: {field_conflicts}")
    print(f"Total unique fields: {len(all_fields)}")
    print(f"Output saved to: {output_file}")
    
    # Show all fields for verification
    print("\nAll fields in merged dataset:")
    for field in sorted(all_fields):
        print(f"  - {field}")

def preview_merged_dataset(output_file: str, num_examples: int = 2) -> None:
    """Preview the merged dataset."""
    output_path = Path(output_file)
    
    if not output_path.exists():
        print("Merged dataset not found.")
        return
    
    print("\n" + "=" * 60)
    print("MERGED DATASET PREVIEW")
    print("=" * 60)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > num_examples:
                break
            
            try:
                data = json.loads(line.strip())
                
                print(f"\nEntry {i} fields:")
                for key in sorted(data.keys()):
                    value = data[key]
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"  {key}: {repr(value) if len(str(value)) < 50 else str(value)[:50] + '...'}")
                
            except json.JSONDecodeError as e:
                print(f"Error reading example {i}: {e}")
    
    # Count total entries
    with open(output_path, 'r', encoding='utf-8') as f:
        total_entries = sum(1 for _ in f)
    
    print(f"\nTotal entries in merged dataset: {total_entries}")
    print("=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input files
    FILE1 = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_correct_answers_only_2025-08-10.jsonl"  # Correct answers
    FILE2 = r"C:\Users\l440\unfaithfulness_steering\datasets\final_results_biased_2025-08-11.jsonl"  # Biased results
    
    # Output file
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_correct_with_biased_merged.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Correct Answers + Biased Results Merger")
    print("=" * 60)
    
    try:
        # Merge the datasets
        merge_datasets(FILE1, FILE2, OUTPUT_FILE)
        
        # Preview the results
        preview_merged_dataset(OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Merge completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")