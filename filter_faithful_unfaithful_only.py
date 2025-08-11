#!/usr/bin/env python3
"""
Faithful and Unfaithful Only Filter

This script creates a dataset containing only the entries classified as 
"faithful" or "unfaithful", excluding all "null" cases.

Usage:
    python filter_faithful_unfaithful_only.py

Configuration:
    - INPUT_FILE: Path to faithfulness classified dataset
    - OUTPUT_FILE: Path to output dataset with only faithful/unfaithful entries
"""

import json
from pathlib import Path
from datetime import datetime

def filter_faithful_unfaithful_only(input_file: str, output_file: str) -> None:
    """
    Filter dataset to keep only entries classified as faithful or unfaithful.
    
    Args:
        input_file: Path to faithfulness classified dataset
        output_file: Path to output dataset with only faithful/unfaithful entries
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
    faithful_count = 0
    unfaithful_count = 0
    null_filtered = 0
    error_entries = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                # Check faithfulness classification
                faithfulness_class = data.get('faithfulness_classification', {})
                faithfulness_result = faithfulness_class.get('faithfulness')
                
                if faithfulness_result in ['faithful', 'unfaithful']:
                    # Keep this entry
                    if faithfulness_result == 'faithful':
                        faithful_count += 1
                    elif faithfulness_result == 'unfaithful':
                        unfaithful_count += 1
                    
                    # Add filter metadata
                    data['faithful_unfaithful_filter'] = {
                        'filter_applied': 'faithful_unfaithful_only',
                        'filter_timestamp': datetime.now().isoformat(),
                        'classification_kept': faithfulness_result
                    }
                    
                    # Write to output file
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    outfile.flush()
                    
                    kept_entries += 1
                    
                    if kept_entries % 10 == 0:
                        print(f"Kept {kept_entries} entries so far...")
                        
                elif faithfulness_result == 'null':
                    null_filtered += 1
                elif faithfulness_result is None:
                    error_entries += 1
                    if error_entries <= 5:  # Show first 5 errors
                        print(f"Line {line_num}: Missing faithfulness classification")
                else:
                    error_entries += 1
                    if error_entries <= 5:
                        print(f"Line {line_num}: Unknown faithfulness value: {faithfulness_result}")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                error_entries += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_entries += 1
    
    # Calculate statistics
    retention_rate = kept_entries / total_entries * 100 if total_entries > 0 else 0
    
    print("-" * 60)
    print("Filtering complete!")
    print(f"Total entries processed: {total_entries}")
    print(f"Entries kept: {kept_entries}")
    print(f"  - Faithful: {faithful_count}")
    print(f"  - Unfaithful: {unfaithful_count}")
    print(f"Entries filtered out: {total_entries - kept_entries}")
    print(f"  - Null classifications: {null_filtered}")
    print(f"  - Errors/missing: {error_entries}")
    print(f"Retention rate: {retention_rate:.2f}%")
    print(f"Filtered dataset saved to: {output_file}")

def analyze_faithful_unfaithful_dataset(output_file: str) -> None:
    """Analyze the filtered dataset."""
    output_path = Path(output_file)
    
    if not output_path.exists():
        print("Filtered dataset not found.")
        return
    
    print("\n" + "=" * 60)
    print("FAITHFUL/UNFAITHFUL DATASET ANALYSIS")
    print("=" * 60)
    
    faithful_examples = []
    unfaithful_examples = []
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                faithfulness_class = data.get('faithfulness_classification', {})
                faithfulness_result = faithfulness_class.get('faithfulness')
                reasoning = faithfulness_class.get('reasoning', 'No reasoning provided')
                
                question = data.get('question', 'N/A')
                original_letter = data.get('extracted_letter', 'N/A')
                biased_letter = data.get('extracted_biased_letter', 'N/A')
                hint_letter = data.get('biased_hint_letter', 'N/A')
                
                example_data = {
                    'line': i,
                    'question': question[:80] + '...' if question and len(question) > 80 else (question or 'N/A'),
                    'original': original_letter,
                    'biased': biased_letter,
                    'hint': hint_letter,
                    'reasoning': reasoning[:100] + '...' if reasoning and len(reasoning) > 100 else (reasoning or 'No reasoning')
                }
                
                if faithfulness_result == 'faithful' and len(faithful_examples) < 5:
                    faithful_examples.append(example_data)
                elif faithfulness_result == 'unfaithful' and len(unfaithful_examples) < 5:
                    unfaithful_examples.append(example_data)
                
            except json.JSONDecodeError as e:
                print(f"Error reading line {i}: {e}")
    
    # Count totals
    with open(output_path, 'r', encoding='utf-8') as f:
        total_faithful_unfaithful = sum(1 for _ in f)
    
    print(f"\nTotal faithful/unfaithful entries: {total_faithful_unfaithful}")
    
    if faithful_examples:
        print(f"\nFAITHFUL Examples ({len(faithful_examples)} shown):")
        for i, ex in enumerate(faithful_examples, 1):
            print(f"{i}. {ex['question']}")
            print(f"   Original: {ex['original']} -> Biased: {ex['biased']} (Hint: {ex['hint']})")
            print(f"   Reasoning: {ex['reasoning']}")
            print()
    
    if unfaithful_examples:
        print(f"UNFAITHFUL Examples ({len(unfaithful_examples)} shown):")
        for i, ex in enumerate(unfaithful_examples, 1):
            print(f"{i}. {ex['question']}")
            print(f"   Original: {ex['original']} -> Biased: {ex['biased']} (Hint: {ex['hint']})")
            print(f"   Reasoning: {ex['reasoning']}")
            print()
    
    print("=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Find the most recent faithfulness classified file
    datasets_dir = Path(r"C:\Users\l440\unfaithfulness_steering\datasets")
    
    # Look for faithfulness classified files
    faithfulness_files = list(datasets_dir.glob("*faithfulness_classified*.jsonl"))
    
    if not faithfulness_files:
        print("No faithfulness classified files found!")
        print("Please run the faithfulness classifier first.")
        exit(1)
    
    # Use the most recent file (by timestamp)
    INPUT_FILE = str(max(faithfulness_files, key=lambda x: x.stat().st_mtime))
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_faithful_unfaithful_only.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Faithful/Unfaithful Only Filter")
    print("=" * 60)
    print(f"Auto-detected input file: {INPUT_FILE}")
    
    try:
        # Filter to faithful/unfaithful only
        filter_faithful_unfaithful_only(INPUT_FILE, OUTPUT_FILE)
        
        # Analyze the results
        analyze_faithful_unfaithful_dataset(OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Filtering completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")