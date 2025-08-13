#!/usr/bin/env python3
"""
Validation Answer Change Comparison

This script compares letter choices between baseline and biased validation results
to see how many answers changed due to the professor bias.

Usage:
    python compare_val_answer_changes.py

Configuration:
    - BASELINE_FILE: Path to baseline results with 'letter' field
    - BIASED_FILE: Path to biased results with 'letter' field
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def load_letters(file_path: str, file_name: str) -> Dict[int, str]:
    """Load letter choices indexed by line number."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_name} file not found: {file_path}")
    
    letters = {}
    
    print(f"Loading {file_name} from: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                letter = data.get('letter', '')
                if letter:
                    letters[line_num] = letter.strip().upper()
            except json.JSONDecodeError as e:
                print(f"Error parsing {file_name} line {line_num + 1}: {e}")
    
    print(f"Loaded {len(letters)} letters from {file_name}")
    return letters

def load_biased_mapping(biased_prompts_file: str) -> Dict[int, int]:
    """
    Load mapping from biased result line numbers to original line numbers.
    
    Args:
        biased_prompts_file: Path to biased prompts file with line_number field
    
    Returns:
        Dictionary mapping biased_line_num -> original_line_num
    """
    path = Path(biased_prompts_file)
    if not path.exists():
        print(f"Warning: Biased prompts file not found: {biased_prompts_file}")
        return {}
    
    mapping = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for biased_line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                original_line_num = data.get('line_number', -1)
                if original_line_num >= 0:
                    mapping[biased_line_num] = original_line_num
            except json.JSONDecodeError as e:
                print(f"Error parsing mapping line {biased_line_num + 1}: {e}")
    
    print(f"Loaded mapping for {len(mapping)} biased prompts")
    return mapping

def compare_answers(baseline_letters: Dict[int, str], biased_letters: Dict[int, str], 
                   biased_mapping: Dict[int, int]) -> Tuple[int, int, int, List[Dict]]:
    """
    Compare baseline and biased answers.
    
    Args:
        baseline_letters: Baseline results {line_num: letter}
        biased_letters: Biased results {line_num: letter}  
        biased_mapping: Mapping {biased_line_num: original_line_num}
    
    Returns:
        Tuple of (total_comparisons, unchanged_count, changed_count, change_details)
    """
    total_comparisons = 0
    unchanged_count = 0
    changed_count = 0
    change_details = []
    
    print("\nComparing answers...")
    print("=" * 80)
    print(f"{'Orig Line':<10} {'Biased Line':<12} {'Baseline':<10} {'Biased':<10} {'Status':<10} {'Change'}")
    print("=" * 80)
    
    # Compare using biased mapping if available
    if biased_mapping:
        for biased_line_num, original_line_num in sorted(biased_mapping.items()):
            if original_line_num in baseline_letters and biased_line_num in biased_letters:
                total_comparisons += 1
                baseline_letter = baseline_letters[original_line_num]
                biased_letter = biased_letters[biased_line_num]
                
                if baseline_letter == biased_letter:
                    unchanged_count += 1
                    status = "SAME"
                    change = ""
                else:
                    changed_count += 1
                    status = "CHANGED"
                    change = f"{baseline_letter} -> {biased_letter}"
                    
                    change_details.append({
                        'original_line': original_line_num + 1,
                        'biased_line': biased_line_num + 1,
                        'baseline_letter': baseline_letter,
                        'biased_letter': biased_letter
                    })
                
                print(f"{original_line_num + 1:<10} {biased_line_num + 1:<12} {baseline_letter:<10} {biased_letter:<10} {status:<10} {change}")
            else:
                missing_baseline = original_line_num not in baseline_letters
                missing_biased = biased_line_num not in biased_letters
                note = ""
                if missing_baseline:
                    note += "No baseline letter "
                if missing_biased:
                    note += "No biased letter"
                print(f"{original_line_num + 1:<10} {biased_line_num + 1:<12} {'N/A':<10} {'N/A':<10} {'MISSING':<10} {note}")
    
    else:
        # Direct line-by-line comparison if no mapping available
        print("No mapping found, doing direct line comparison...")
        max_lines = max(max(baseline_letters.keys(), default=-1), max(biased_letters.keys(), default=-1))
        
        for line_num in range(max_lines + 1):
            if line_num in baseline_letters and line_num in biased_letters:
                total_comparisons += 1
                baseline_letter = baseline_letters[line_num]
                biased_letter = biased_letters[line_num]
                
                if baseline_letter == biased_letter:
                    unchanged_count += 1
                    status = "SAME"
                    change = ""
                else:
                    changed_count += 1
                    status = "CHANGED"
                    change = f"{baseline_letter} -> {biased_letter}"
                    
                    change_details.append({
                        'original_line': line_num + 1,
                        'biased_line': line_num + 1,
                        'baseline_letter': baseline_letter,
                        'biased_letter': biased_letter
                    })
                
                print(f"{line_num + 1:<10} {line_num + 1:<12} {baseline_letter:<10} {biased_letter:<10} {status:<10} {change}")
    
    return total_comparisons, unchanged_count, changed_count, change_details

def print_summary(total_comparisons: int, unchanged_count: int, changed_count: int, 
                 change_details: List[Dict]):
    """Print comparison summary."""
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Total comparisons: {total_comparisons}")
    print(f"Unchanged answers: {unchanged_count}")
    print(f"Changed answers: {changed_count}")
    
    if total_comparisons > 0:
        unchanged_pct = unchanged_count / total_comparisons * 100
        changed_pct = changed_count / total_comparisons * 100
        print(f"Resistance rate: {unchanged_pct:.1f}% (model stuck to original answer)")
        print(f"Change rate: {changed_pct:.1f}% (model was influenced by bias)")
    
    if change_details:
        print(f"\nAll changes:")
        for detail in change_details:
            print(f"  Original line {detail['original_line']}: {detail['baseline_letter']} -> {detail['biased_letter']}")
    
    print("=" * 80)

def main():
    """Main function."""
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Baseline results file
    BASELINE_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_baseline_correct_results_2025-08-12.jsonl.jsonl"
    
    # Biased results file
    BIASED_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_biased_results_letters_2025-08-12.jsonl"
    
    # Biased prompts file (for mapping)
    BIASED_PROMPTS_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_biased_prompts_2025-08-12.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 80)
    print("Validation Answer Change Comparison")
    print("=" * 80)
    
    try:
        # Load letter choices
        baseline_letters = load_letters(BASELINE_FILE, "baseline")
        biased_letters = load_letters(BIASED_FILE, "biased")
        
        if not baseline_letters:
            print("Error: No baseline letters loaded")
            return
        
        if not biased_letters:
            print("Error: No biased letters loaded")
            return
        
        # Load biased mapping
        biased_mapping = load_biased_mapping(BIASED_PROMPTS_FILE)
        
        # Compare answers
        total_comparisons, unchanged_count, changed_count, change_details = compare_answers(
            baseline_letters, biased_letters, biased_mapping
        )
        
        # Print summary
        print_summary(total_comparisons, unchanged_count, changed_count, change_details)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()