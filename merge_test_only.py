#!/usr/bin/env python3
"""
Test Split Merger (Corrected)

The extracted letters dataset contains ONLY test split entries (544 lines).
The MMLU psychology dataset contains BOTH test and validation splits (604 lines).
This script merges the extracted letters with only the first 544 rows of the MMLU dataset.

Usage:
    python merge_test_only.py
"""

import json
from pathlib import Path
from datetime import datetime

def merge_test_only(source_file: str, target_file: str, output_file: str) -> None:
    """
    Merge extracted letters (test split only) with first N rows of MMLU dataset.
    
    Args:
        source_file: File with extracted letters (test split only, 544 lines)
        target_file: MMLU dataset with test+val (604 lines, we only want first 544)
        output_file: Output merged file
    """
    source_path = Path(source_file)
    target_path = Path(target_file)
    output_path = Path(output_file)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")
    
    output_path.parent.mkdir(exist_ok=True)
    
    # Count lines in source file
    with open(source_path, 'r', encoding='utf-8') as f:
        source_line_count = sum(1 for _ in f)
    
    print(f"Source file (extracted letters, test split only): {source_line_count} lines")
    print(f"Target file (MMLU test+val): will use first {source_line_count} lines only")
    print(f"Output file: {output_file}")
    print("-" * 60)
    
    # Load all source data
    print("Loading source data...")
    source_data = []
    with open(source_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                source_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing source line {line_num}: {e}")
                source_data.append({})
    
    print(f"Loaded {len(source_data)} entries from source")
    
    # Process target file - only first N lines
    merged_count = 0
    
    with open(target_path, 'r', encoding='utf-8') as target_f, \
         open(output_path, 'w', encoding='utf-8') as output_f:
        
        for line_num, line in enumerate(target_f, 1):
            # Only process lines up to source file length
            if line_num > len(source_data):
                print(f"Stopping at line {line_num-1} - reached end of test split")
                break
                
            try:
                target_data = json.loads(line.strip())
                source_entry = source_data[line_num - 1]
                
                # Add fields from source to target
                if 'generated_answer' in source_entry:
                    target_data['generated_answer'] = source_entry['generated_answer']
                if 'extracted_letter' in source_entry:
                    target_data['extracted_letter'] = source_entry['extracted_letter']
                
                # Add metadata
                target_data['merge_timestamp'] = datetime.now().isoformat()
                target_data['split_type'] = 'test_only'
                
                merged_count += 1
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines...")
                
                # Write to output
                output_f.write(json.dumps(target_data, ensure_ascii=False) + '\n')
                output_f.flush()
                
            except json.JSONDecodeError as e:
                print(f"Error parsing target line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print("-" * 60)
    print(f"Merge complete!")
    print(f"Successfully merged: {merged_count} entries (test split only)")
    print(f"Skipped validation split from target file")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Configuration
    SOURCE_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\final_results_with_extracted_letters_2025-08-10.jsonl"  # Test split only
    TARGET_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_test_validation_high_school_psychology_all_2025-08-10.jsonl"  # Test + Val
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_test_only_merged.jsonl"
    
    print("=" * 60)
    print("Test Split Only Merger (Corrected)")
    print("=" * 60)
    
    try:
        merge_test_only(SOURCE_FILE, TARGET_FILE, OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Test-only merge completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")