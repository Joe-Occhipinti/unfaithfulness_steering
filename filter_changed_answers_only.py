#!/usr/bin/env python3
"""
Changed Answers Filter

This script creates a dataset containing only the rows where the model
changed its answer due to biased prompts (extracted_letter != extracted_biased_letter).

Usage:
    python filter_changed_answers_only.py

Configuration:
    - INPUT_FILE: Path to dataset with both extracted_letter and extracted_biased_letter
    - OUTPUT_FILE: Path to output dataset with only changed answers
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def filter_changed_answers(input_file: str, output_file: str) -> None:
    """
    Filter dataset to keep only entries where the model changed its answer.
    
    Args:
        input_file: Path to input dataset with both letter fields
        output_file: Path to output dataset with only changed answers
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
    changed_answers = 0
    same_answers = 0
    invalid_entries = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                # Get both letter fields
                original_letter = data.get('extracted_letter')
                biased_letter = data.get('extracted_biased_letter')
                
                # Skip if missing data
                if original_letter is None or biased_letter is None:
                    invalid_entries += 1
                    continue
                
                # Normalize letters (ensure uppercase)
                original_letter = str(original_letter).upper().strip()
                biased_letter = str(biased_letter).upper().strip()
                
                # Skip if invalid letters
                if (original_letter not in ['A', 'B', 'C', 'D'] or 
                    biased_letter not in ['A', 'B', 'C', 'D']):
                    invalid_entries += 1
                    continue
                
                # Check if answer changed
                if original_letter != biased_letter:
                    # Answer changed - keep this entry
                    changed_answers += 1
                    
                    # Add metadata about the change
                    data['answer_changed'] = True
                    data['change_pattern'] = f"{original_letter}_to_{biased_letter}"
                    data['change_filter_timestamp'] = datetime.now().isoformat()
                    
                    # Extract biased hint letter if available
                    biased_hint = extract_biased_hint_letter(data)
                    if biased_hint:
                        data['biased_hint_letter'] = biased_hint
                    
                    # Write to output file
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    outfile.flush()
                    
                    if changed_answers % 10 == 0:
                        print(f"Found {changed_answers} changed answers so far...")
                else:
                    # Same answer - skip this entry
                    same_answers += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                invalid_entries += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                invalid_entries += 1
    
    # Calculate statistics
    change_rate = changed_answers / (total_entries - invalid_entries) * 100 if (total_entries - invalid_entries) > 0 else 0
    
    print("-" * 60)
    print("Filtering complete!")
    print(f"Total entries processed: {total_entries}")
    print(f"Answer changes found: {changed_answers}")
    print(f"Same answers (filtered out): {same_answers}")
    print(f"Invalid entries: {invalid_entries}")
    print(f"Change rate: {change_rate:.2f}%")
    print(f"Changed answers dataset saved to: {output_file}")

def extract_biased_hint_letter(data: Dict[str, Any]) -> str:
    """Extract the biased hint letter from the prompt."""
    # Try to find it in original_prompt field
    original_prompt = data.get('original_prompt', '')
    if original_prompt:
        # Look for pattern like "thinks the best answer is: X."
        import re
        pattern = r"thinks the best answer is:\s*([A-D])"
        match = re.search(pattern, original_prompt, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Try biased_prompt field
    biased_prompt = data.get('biased_prompt', '')
    if biased_prompt:
        import re
        pattern = r"thinks the best answer is:\s*([A-D])"
        match = re.search(pattern, biased_prompt, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None

def analyze_changed_answers(output_file: str) -> None:
    """Analyze the changed answers dataset."""
    output_path = Path(output_file)
    
    if not output_path.exists():
        print("Changed answers dataset not found.")
        return
    
    print("\n" + "=" * 60)
    print("CHANGED ANSWERS ANALYSIS")
    print("=" * 60)
    
    change_patterns = {}
    biased_hints = {}
    correct_answers = {}
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Track change patterns
                pattern = data.get('change_pattern', 'Unknown')
                change_patterns[pattern] = change_patterns.get(pattern, 0) + 1
                
                # Track biased hints
                hint = data.get('biased_hint_letter', 'Unknown')
                biased_hints[hint] = biased_hints.get(hint, 0) + 1
                
                # Track what the correct answer was
                correct = data.get('answer', -1)
                correct_answers[correct] = correct_answers.get(correct, 0) + 1
                
                # Show first few examples
                if i <= 5:
                    orig = data.get('extracted_letter', 'N/A')
                    biased = data.get('extracted_biased_letter', 'N/A')
                    hint_letter = data.get('biased_hint_letter', 'N/A')
                    question = data.get('question', 'N/A')
                    
                    print(f"\nExample {i}:")
                    print(f"  Question: {question[:80]}{'...' if len(question) > 80 else ''}")
                    print(f"  Original answer: {orig}")
                    print(f"  Biased answer: {biased}")
                    print(f"  Biased hint: {hint_letter}")
                    print(f"  Correct answer: {correct}")
                
            except json.JSONDecodeError as e:
                print(f"Error reading line {i}: {e}")
    
    # Count total entries
    with open(output_path, 'r', encoding='utf-8') as f:
        total_changed = sum(1 for _ in f)
    
    print(f"\nTotal changed answers: {total_changed}")
    
    print(f"\nChange Patterns:")
    for pattern, count in sorted(change_patterns.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_changed * 100 if total_changed > 0 else 0
        print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    print(f"\nBiased Hints Distribution:")
    for hint, count in sorted(biased_hints.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_changed * 100 if total_changed > 0 else 0
        print(f"  {hint}: {count} ({percentage:.1f}%)")
    
    print(f"\nCorrect Answers Distribution (where changes occurred):")
    letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    for answer_idx, count in sorted(correct_answers.items()):
        letter = letter_map.get(answer_idx, str(answer_idx))
        percentage = count / total_changed * 100 if total_changed > 0 else 0
        print(f"  {answer_idx} ({letter}): {count} ({percentage:.1f}%)")
    
    print("=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input file (dataset with both original and biased extracted letters)
    INPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_correct_with_biased_filtered.jsonl"
    
    # Output file (dataset with only changed answers)
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_changed_answers_only.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Changed Answers Filter")
    print("=" * 60)
    
    try:
        # Filter to changed answers only
        filter_changed_answers(INPUT_FILE, OUTPUT_FILE)
        
        # Analyze the changed answers
        analyze_changed_answers(OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Changed answers filtering completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")