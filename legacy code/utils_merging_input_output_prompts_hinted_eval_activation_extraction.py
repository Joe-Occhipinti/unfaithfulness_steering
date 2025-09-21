#!/usr/bin/env python3
"""
Forward Pass Prompt Creator

This script takes the faithful/unfaithful dataset and creates a new field 
"forward_pass_prompt" by concatenating "original_prompt" + "generated_biased_answer".

Usage:
    python create_forward_pass_prompts.py

Configuration:
    - INPUT_FILE: Path to faithful/unfaithful dataset
    - OUTPUT_FILE: Path to output dataset with forward_pass_prompt field
"""

import json
from pathlib import Path
from datetime import datetime

def create_forward_pass_prompts(input_file: str, output_file: str) -> None:
    """
    Create forward_pass_prompt field by concatenating original_prompt + generated_biased_answer.
    
    Args:
        input_file: Path to faithful/unfaithful dataset
        output_file: Path to output dataset with forward_pass_prompt field
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
    successful_concatenations = 0
    missing_fields = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                # Get the required fields
                original_prompt = data.get('original_prompt', '')
                generated_biased_answer = data.get('generated_biased_answer', '')
                
                if not original_prompt or not generated_biased_answer:
                    missing_fields += 1
                    print(f"Warning: Line {line_num} missing required fields")
                    print(f"  original_prompt: {'✓' if original_prompt else '✗'}")
                    print(f"  generated_biased_answer: {'✓' if generated_biased_answer else '✗'}")
                    # Still add the entry but with empty forward_pass_prompt
                    data['forward_pass_prompt'] = ""
                else:
                    # Create the concatenated forward pass prompt
                    forward_pass_prompt = original_prompt + "\n\n" + generated_biased_answer
                    data['forward_pass_prompt'] = forward_pass_prompt
                    successful_concatenations += 1
                
                # Add metadata about the concatenation
                data['forward_pass_metadata'] = {
                    'created_timestamp': datetime.now().isoformat(),
                    'original_prompt_length': len(original_prompt),
                    'generated_biased_answer_length': len(generated_biased_answer),
                    'forward_pass_prompt_length': len(data['forward_pass_prompt']),
                    'concatenation_successful': bool(original_prompt and generated_biased_answer)
                }
                
                # Write to output file
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                outfile.flush()
                
                if successful_concatenations % 10 == 0 and successful_concatenations > 0:
                    print(f"Successfully concatenated {successful_concatenations} entries...")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print("-" * 60)
    print("Concatenation complete!")
    print(f"Total entries processed: {total_entries}")
    print(f"Successful concatenations: {successful_concatenations}")
    print(f"Missing fields: {missing_fields}")
    if total_entries > 0:
        success_rate = successful_concatenations / total_entries * 100
        print(f"Success rate: {success_rate:.1f}%")
    print(f"Output saved to: {output_file}")

def analyze_forward_pass_prompts(output_file: str, num_examples: int = 3) -> None:
    """Analyze the created forward pass prompts."""
    output_path = Path(output_file)
    
    if not output_path.exists():
        print("Output dataset not found.")
        return
    
    print("\n" + "=" * 60)
    print("FORWARD PASS PROMPTS ANALYSIS")
    print("=" * 60)
    
    faithful_examples = []
    unfaithful_examples = []
    prompt_lengths = []
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                faithfulness_class = data.get('faithfulness_classification', {})
                faithfulness_result = faithfulness_class.get('faithfulness')
                
                forward_pass_prompt = data.get('forward_pass_prompt', '')
                metadata = data.get('forward_pass_metadata', {})
                prompt_length = metadata.get('forward_pass_prompt_length', 0)
                
                if prompt_length > 0:
                    prompt_lengths.append(prompt_length)
                
                question = data.get('question', 'N/A')[:60] + '...' if data.get('question') and len(data.get('question', '')) > 60 else data.get('question', 'N/A')
                
                example_data = {
                    'line': i,
                    'question': question,
                    'faithfulness': faithfulness_result,
                    'prompt_length': prompt_length,
                    'forward_pass_preview': forward_pass_prompt[:200] + '...' if len(forward_pass_prompt) > 200 else forward_pass_prompt
                }
                
                if faithfulness_result == 'faithful' and len(faithful_examples) < num_examples:
                    faithful_examples.append(example_data)
                elif faithfulness_result == 'unfaithful' and len(unfaithful_examples) < num_examples:
                    unfaithful_examples.append(example_data)
                
            except json.JSONDecodeError as e:
                print(f"Error reading line {i}: {e}")
    
    # Calculate statistics
    if prompt_lengths:
        avg_length = sum(prompt_lengths) / len(prompt_lengths)
        min_length = min(prompt_lengths)
        max_length = max(prompt_lengths)
        
        print(f"Forward Pass Prompt Statistics:")
        print(f"  Total entries: {len(prompt_lengths)}")
        print(f"  Average length: {avg_length:.0f} characters")
        print(f"  Min length: {min_length} characters")
        print(f"  Max length: {max_length} characters")
    
    if faithful_examples:
        print(f"\nFAITHFUL Examples ({len(faithful_examples)} shown):")
        for i, ex in enumerate(faithful_examples, 1):
            print(f"{i}. {ex['question']} ({ex['prompt_length']} chars)")
            print(f"   Preview: {ex['forward_pass_preview']}")
            print()
    
    if unfaithful_examples:
        print(f"UNFAITHFUL Examples ({len(unfaithful_examples)} shown):")
        for i, ex in enumerate(unfaithful_examples, 1):
            print(f"{i}. {ex['question']} ({ex['prompt_length']} chars)")
            print(f"   Preview: {ex['forward_pass_preview']}")
            print()
    
    print("=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Find the faithful/unfaithful dataset
    datasets_dir = Path(r"C:\Users\l440\unfaithfulness_steering\datasets")
    
    # Look for faithful/unfaithful files
    faithful_unfaithful_files = list(datasets_dir.glob("*faithful_unfaithful*.jsonl"))
    
    if not faithful_unfaithful_files:
        print("No faithful/unfaithful files found!")
        print("Available files:")
        for f in datasets_dir.glob("*.jsonl"):
            print(f"  - {f.name}")
        exit(1)
    
    # Use the most recent file (by timestamp)
    INPUT_FILE = str(max(faithful_unfaithful_files, key=lambda x: x.stat().st_mtime))
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_forward_pass_prompts.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Forward Pass Prompt Creator")
    print("=" * 60)
    print(f"Auto-detected input file: {INPUT_FILE}")
    
    try:
        # Create forward pass prompts
        create_forward_pass_prompts(INPUT_FILE, OUTPUT_FILE)
        
        # Analyze the results
        analyze_forward_pass_prompts(OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Forward pass prompts created successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")