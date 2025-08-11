#!/usr/bin/env python3
"""
JSON File Merger

This script merges two JSONL files by adding specific fields from the first file to the second file.
It matches entries by line number and adds 'generated_answer' and 'extracted_letter' fields
from the first file to the corresponding entries in the second file.

Usage:
    python merge_json_files.py

Configuration:
    - SOURCE_FILE: File containing the fields to copy (generated_answer, extracted_letter)
    - TARGET_FILE: File to merge into (will receive the new fields)
    - OUTPUT_FILE: Merged output file
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

def merge_jsonl_files(source_file: str, target_file: str, output_file: str, 
                     fields_to_copy: List[str]) -> None:
    """
    Merge two JSONL files by adding specified fields from source to target.
    
    Args:
        source_file: Path to file containing fields to copy
        target_file: Path to file to merge into
        output_file: Path to output merged file
        fields_to_copy: List of field names to copy from source to target
    """
    source_path = Path(source_file)
    target_path = Path(target_file)
    output_path = Path(output_file)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Source file: {source_file}")
    print(f"Target file: {target_file}")
    print(f"Output file: {output_file}")
    print(f"Fields to copy: {fields_to_copy}")
    print("-" * 60)
    
    # Read source file into memory
    print("Loading source file...")
    source_data = []
    with open(source_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                source_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing source line {line_num}: {e}")
                source_data.append({})  # Add empty dict to maintain line alignment
    
    print(f"Loaded {len(source_data)} entries from source file")
    
    # Process target file and merge
    print("Merging files...")
    merged_count = 0
    skipped_count = 0
    field_conflicts = 0
    
    with open(target_path, 'r', encoding='utf-8') as target_f, \
         open(output_path, 'w', encoding='utf-8') as output_f:
        
        for line_num, line in enumerate(target_f, 1):
            try:
                # Parse target line
                target_data = json.loads(line.strip())
                
                # Check if we have corresponding source data
                if line_num <= len(source_data):
                    source_entry = source_data[line_num - 1]
                    
                    # Copy specified fields from source to target
                    fields_copied = []
                    for field in fields_to_copy:
                        if field in source_entry:
                            # Check if field already exists in target
                            if field in target_data:
                                print(f"Warning: Line {line_num} - field '{field}' already exists in target, skipping")
                                field_conflicts += 1
                            else:
                                target_data[field] = source_entry[field]
                                fields_copied.append(field)
                    
                    if fields_copied:
                        merged_count += 1
                        if line_num % 50 == 0:  # Progress update every 50 lines
                            print(f"Processed {line_num} lines...")
                else:
                    print(f"Warning: Line {line_num} - no corresponding source data")
                    skipped_count += 1
                
                # Add merge metadata
                target_data['merge_timestamp'] = datetime.now().isoformat()
                
                # Write merged entry to output
                output_f.write(json.dumps(target_data, ensure_ascii=False) + '\n')
                output_f.flush()
                
            except json.JSONDecodeError as e:
                print(f"Error parsing target line {line_num}: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_count += 1
    
    print("-" * 60)
    print(f"Merge complete!")
    print(f"Successfully merged: {merged_count} entries")
    print(f"Skipped entries: {skipped_count}")
    print(f"Field conflicts: {field_conflicts}")
    print(f"Output saved to: {output_file}")

def preview_files(source_file: str, target_file: str, num_lines: int = 3) -> None:
    """Preview the structure of both files before merging."""
    print("=" * 60)
    print("FILE PREVIEW")
    print("=" * 60)
    
    # Preview source file
    print(f"\nSource file structure ({source_file}):")
    print("-" * 40)
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                data = json.loads(line.strip())
                print(f"Line {i+1} fields: {list(data.keys())}")
                if i == 0:
                    print(f"Sample data: {json.dumps(data, indent=2)[:200]}...")
    except Exception as e:
        print(f"Error reading source file: {e}")
    
    # Preview target file
    print(f"\nTarget file structure ({target_file}):")
    print("-" * 40)
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                data = json.loads(line.strip())
                print(f"Line {i+1} fields: {list(data.keys())}")
                if i == 0:
                    print(f"Sample data: {json.dumps(data, indent=2)[:200]}...")
    except Exception as e:
        print(f"Error reading target file: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Source file (contains generated_answer and extracted_letter)
    SOURCE_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\final_results_with_extracted_letters.jsonl"
    
    # Target file (MMLU test data to merge into)
    TARGET_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_test_validation_high_school_psychology_all_2025-08-10.jsonl"
    
    # Output file (merged result)
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_merged_results.jsonl"
    
    # Fields to copy from source to target (avoid duplicating existing fields)
    FIELDS_TO_COPY = ["generated_answer", "extracted_letter"]
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("JSON File Merger")
    print("=" * 60)
    
    try:
        # Preview file structures
        preview_files(SOURCE_FILE, TARGET_FILE)
        
        # Perform the merge
        merge_jsonl_files(SOURCE_FILE, TARGET_FILE, OUTPUT_FILE, FIELDS_TO_COPY)
        
        print("\n" + "=" * 60)
        print("Merge completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")