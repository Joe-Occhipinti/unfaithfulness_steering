#!/usr/bin/env python3
"""
Original and Biased Merger

This script merges the original extracted letters dataset with the biased extracted letters dataset
to create a combined dataset for comparison analysis.

Usage:
    python merge_original_and_biased.py
"""

import json
from pathlib import Path
from datetime import datetime

def merge_original_and_biased(original_file: str, biased_file: str, output_file: str) -> None:
    """
    Merge original and biased datasets for comparison analysis.
    
    Args:
        original_file: Dataset with original extracted letters
        biased_file: Dataset with biased extracted letters  
        output_file: Output merged dataset
    """
    original_path = Path(original_file)
    biased_path = Path(biased_file)
    output_path = Path(output_file)
    
    if not original_path.exists():
        raise FileNotFoundError(f"Original file not found: {original_file}")
    if not biased_path.exists():
        raise FileNotFoundError(f"Biased file not found: {biased_file}")
    
    output_path.parent.mkdir(exist_ok=True)
    
    # Load biased data
    print("Loading biased dataset...")
    biased_data = []
    with open(biased_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                biased_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing biased line {line_num}: {e}")
                biased_data.append({})
    
    print(f"Loaded {len(biased_data)} biased entries")
    
    # Merge with original data
    print("Merging with original dataset...")
    merged_count = 0
    
    with open(original_path, 'r', encoding='utf-8') as orig_f, \
         open(output_path, 'w', encoding='utf-8') as out_f:
        
        for line_num, line in enumerate(orig_f, 1):
            try:
                original_entry = json.loads(line.strip())
                
                # Get corresponding biased entry
                if line_num <= len(biased_data):
                    biased_entry = biased_data[line_num - 1]
                    
                    # Add biased fields to original entry
                    if 'extracted_biased_letter' in biased_entry:
                        original_entry['extracted_biased_letter'] = biased_entry['extracted_biased_letter']
                    if 'generated_biased_answer' in biased_entry:
                        original_entry['generated_biased_answer'] = biased_entry['generated_biased_answer']
                    if 'original_prompt' in biased_entry:
                        original_entry['biased_prompt'] = biased_entry['original_prompt']
                    
                    merged_count += 1
                    
                    if line_num % 100 == 0:
                        print(f"Merged {line_num} entries...")
                else:
                    print(f"Warning: No biased data for original line {line_num}")
                
                # Add merge metadata
                original_entry['comparison_merge_timestamp'] = datetime.now().isoformat()
                
                # Write merged entry
                out_f.write(json.dumps(original_entry, ensure_ascii=False) + '\n')
                out_f.flush()
                
            except json.JSONDecodeError as e:
                print(f"Error parsing original line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print(f"Merge complete! Successfully merged {merged_count} entries")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Configuration
    ORIGINAL_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\final_results_with_extracted_letters_2025-08-10.jsonl"
    BIASED_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\final_results_biased_with_extracted_letters.jsonl"
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\final_results_comparison_dataset.jsonl"
    
    print("=" * 60)
    print("Original and Biased Dataset Merger")
    print("=" * 60)
    
    try:
        merge_original_and_biased(ORIGINAL_FILE, BIASED_FILE, OUTPUT_FILE)
        print("\n" + "=" * 60)
        print("Merge completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")