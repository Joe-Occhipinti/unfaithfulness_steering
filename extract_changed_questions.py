#!/usr/bin/env python3
"""
Extract Changed Questions Dataset

This script extracts the questions where answers changed between baseline and biased results,
along with their original prompts and biased answers for faithfulness analysis.

Usage:
    python extract_changed_questions.py

Configuration:
    - BASELINE_FILE: Path to baseline results with 'letter' field
    - BIASED_FILE: Path to biased results with 'letter' field  
    - BIASED_PROMPTS_FILE: Path to biased prompts file
    - BIASED_RESULTS_RAW_FILE: Path to raw biased results with full answers
    - OUTPUT_FILE: Path to output changed questions dataset
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

def load_baseline_letters(baseline_file: str) -> Dict[int, str]:
    """Load baseline letter choices."""
    letters = {}
    with open(baseline_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                letter = data.get('letter', '')
                if letter:
                    letters[line_num] = letter.strip().upper()
            except json.JSONDecodeError:
                pass
    return letters

def load_biased_letters(biased_file: str) -> Dict[int, str]:
    """Load biased letter choices."""
    letters = {}
    with open(biased_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                letter = data.get('letter', '')
                if letter:
                    letters[line_num] = letter.strip().upper()
            except json.JSONDecodeError:
                pass
    return letters

def load_biased_mapping(biased_prompts_file: str) -> Dict[int, Dict]:
    """Load mapping from biased line numbers to original line numbers with prompt details."""
    mapping = {}
    with open(biased_prompts_file, 'r', encoding='utf-8') as f:
        for biased_line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                original_line_num = data.get('line_number', -1)
                if original_line_num >= 0:
                    mapping[biased_line_num] = {
                        'original_line_num': original_line_num,
                        'biased_prompt': data.get('biased_prompt', ''),
                        'original_question': data.get('original_question', ''),
                        'correct_answer': data.get('correct_answer', ''),
                        'biased_hint_letter': data.get('biased_hint_letter', ''),
                        'subject': data.get('subject', ''),
                        'split': data.get('split', '')
                    }
            except json.JSONDecodeError:
                pass
    return mapping

def load_biased_raw_answers(biased_results_raw_file: str) -> Dict[int, str]:
    """Load full biased answers from raw results file."""
    answers = {}
    with open(biased_results_raw_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                answer = data.get('answer', '')
                if answer:
                    answers[line_num] = answer
            except json.JSONDecodeError:
                pass
    return answers

def find_changed_questions(baseline_letters: Dict[int, str], biased_letters: Dict[int, str], 
                          biased_mapping: Dict[int, Dict]) -> Set[int]:
    """Find original line numbers where answers changed."""
    changed_original_lines = set()
    
    for biased_line_num, mapping_info in biased_mapping.items():
        original_line_num = mapping_info['original_line_num']
        
        if (original_line_num in baseline_letters and 
            biased_line_num in biased_letters):
            
            baseline_letter = baseline_letters[original_line_num]
            biased_letter = biased_letters[biased_line_num]
            
            if baseline_letter != biased_letter:
                changed_original_lines.add(original_line_num)
    
    return changed_original_lines

def extract_changed_dataset(baseline_file: str, biased_file: str, biased_prompts_file: str,
                           biased_results_raw_file: str, output_file: str) -> None:
    """Extract dataset of questions that changed answers."""
    
    print("Loading baseline letters...")
    baseline_letters = load_baseline_letters(baseline_file)
    print(f"Loaded {len(baseline_letters)} baseline letters")
    
    print("Loading biased letters...")
    biased_letters = load_biased_letters(biased_file)
    print(f"Loaded {len(biased_letters)} biased letters")
    
    print("Loading biased mapping...")
    biased_mapping = load_biased_mapping(biased_prompts_file)
    print(f"Loaded {len(biased_mapping)} biased mappings")
    
    print("Loading biased raw answers...")
    biased_raw_answers = load_biased_raw_answers(biased_results_raw_file)
    print(f"Loaded {len(biased_raw_answers)} biased raw answers")
    
    print("Finding changed questions...")
    changed_original_lines = find_changed_questions(baseline_letters, biased_letters, biased_mapping)
    print(f"Found {len(changed_original_lines)} questions that changed")
    
    if not changed_original_lines:
        print("No changed questions found!")
        return
    
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Creating changed questions dataset: {output_file}")
    
    extracted_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Find the biased line numbers that correspond to changed questions
        for biased_line_num, mapping_info in biased_mapping.items():
            original_line_num = mapping_info['original_line_num']
            
            if original_line_num in changed_original_lines:
                # Get the full biased answer
                biased_full_answer = biased_raw_answers.get(biased_line_num, '')
                
                if not biased_full_answer:
                    print(f"Warning: No raw answer found for biased line {biased_line_num}")
                    continue
                
                # Create the dataset entry
                entry = {
                    'original_line_number': original_line_num,
                    'biased_line_number': biased_line_num,
                    'original_question': mapping_info['original_question'],
                    'correct_answer': mapping_info['correct_answer'],
                    'biased_hint_letter': mapping_info['biased_hint_letter'],
                    'baseline_answer': baseline_letters.get(original_line_num, ''),
                    'biased_answer_letter': biased_letters.get(biased_line_num, ''),
                    'original_prompt': mapping_info['biased_prompt'],  # The biased prompt used
                    'generated_biased_answer': biased_full_answer,
                    'subject': mapping_info.get('subject', ''),
                    'split': mapping_info.get('split', ''),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'change_type': f"{baseline_letters.get(original_line_num, '')} -> {biased_letters.get(biased_line_num, '')}"
                }
                
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                outfile.flush()
                
                extracted_count += 1
                print(f"Extracted line {original_line_num}: {entry['change_type']} - {entry['original_question'][:50]}...")
    
    print("-" * 60)
    print("Extraction complete!")
    print(f"Total changed questions extracted: {extracted_count}")
    print(f"Output saved to: {output_file}")

def preview_changed_dataset(output_file: str, num_examples: int = 3) -> None:
    """Preview the extracted changed questions dataset."""
    output_path = Path(output_file)
    
    if not output_path.exists():
        print("Changed questions file not found. Please run extraction first.")
        return
    
    print("\nPreview of changed questions:")
    print("=" * 80)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > num_examples:
                break
            
            try:
                data = json.loads(line.strip())
                
                print(f"Example {i}:")
                print(f"Original line: {data.get('original_line_number', 'N/A')}")
                print(f"Question: {data.get('original_question', 'N/A')}")
                print(f"Correct answer: {data.get('correct_answer', 'N/A')}")
                print(f"Change: {data.get('change_type', 'N/A')}")
                print(f"Biased hint: {data.get('biased_hint_letter', 'N/A')}")
                print(f"Generated answer (first 200 chars): {data.get('generated_biased_answer', 'N/A')[:200]}...")
                print("-" * 40)
                
            except json.JSONDecodeError as e:
                print(f"Error reading example {i}: {e}")
    
    # Count total entries
    with open(output_path, 'r', encoding='utf-8') as f:
        total_entries = sum(1 for _ in f)
    
    print(f"Total changed questions extracted: {total_entries}")
    print("=" * 80)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input files
    BASELINE_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_baseline_correct_results_2025-08-12.jsonl.jsonl"
    BIASED_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_biased_results_letters_2025-08-12.jsonl"
    BIASED_PROMPTS_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_biased_prompts_2025-08-12.jsonl"
    BIASED_RESULTS_RAW_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_biased_results_2025-08-12.jsonl"
    
    # Output file
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_changed_questions_2025-08-12.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 80)
    print("Extract Changed Questions Dataset")
    print("=" * 80)
    
    try:
        # Extract changed questions dataset
        extract_changed_dataset(BASELINE_FILE, BIASED_FILE, BIASED_PROMPTS_FILE, 
                               BIASED_RESULTS_RAW_FILE, OUTPUT_FILE)
        
        # Preview the results
        preview_changed_dataset(OUTPUT_FILE)
        
        print("\n" + "=" * 80)
        print("Changed questions extraction completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")