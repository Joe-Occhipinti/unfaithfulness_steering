#!/usr/bin/env python3
"""
Generate Text from Val Changed Questions

This script generates text responses from the validation changed questions dataset using a specified model.
Inspired by the activations_test.py approach with model loading and text generation capabilities.

Usage:
    python generate_text_val_changed.py

Configuration:
    - Edit the configuration variables below
    - Specify your model ID and other parameters
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time
import re
from tqdm import tqdm

# ========================================
# CONFIGURATION - EDIT THESE VARIABLES
# ========================================

# Model configuration - SPECIFY YOUR MODEL HERE
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Change this to your desired model
HF_TOKEN = "your_hf_token_here"  # Set your Hugging Face token if needed

# Input/Output files
INPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_changed_questions_2025-08-12.jsonl"
OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\val_changed_generated_results.jsonl"

# Generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16  # or torch.float16

print(f"Using device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Input file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")


class BlockOutputWrapper(torch.nn.Module):
    """BlockOutputWrapper Class (inspired by activations_test.py) - Optional for future activation work"""
    
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None
        self.add_activations = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.last_hidden_state = output[0]
        self.output_before_adding = output

        if self.add_activations is not None:
            output = (output[0] + self.add_activations,) + output[1:]

        self.output_after_adding = output
        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.last_hidden_state = None
        self.add_activations = None


def load_model_and_tokenizer(model_name: str, hf_token: str = None):
    """Load model and tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if hf_token != "your_hf_token_here" else None
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token if hf_token != "your_hf_token_here" else None,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
    print(f"Model architecture: {type(model).__name__}")
    
    return model, tokenizer


def load_changed_questions(file_path: str) -> List[Dict[str, Any]]:
    """Load the changed questions dataset."""
    questions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                questions.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    
    return questions


def generate_text(prompt: str, model, tokenizer, **generation_kwargs) -> str:
    """Generate text response for a given prompt."""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (remove input prompt)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text.strip()


def extract_letter_choice(answer_text: str) -> str:
    """Extract the last letter choice from generated answer (inspired by extract_letter_choices.py)."""
    # Pattern to find "therefore, the best answer is:" followed by a letter with optional parentheses
    pattern = r"therefore,?\s+the\s+best\s+answer\s+is:\s*[\(\*]*([A-D])[\)\*]*"
    
    # Find all matches (case-insensitive)
    matches = re.findall(pattern, answer_text, re.IGNORECASE)
    
    if matches:
        # Return the last match in uppercase
        return matches[-1].upper()
    
    return None


def process_question(question_data: Dict[str, Any], model, tokenizer, generation_kwargs: Dict) -> Dict[str, Any]:
    """Process a single question and generate response."""
    
    # Get the original biased prompt
    original_prompt = question_data.get('original_prompt', '')
    
    if not original_prompt:
        return {
            **question_data,
            'generated_answer': '',
            'extracted_letter': None,
            'generation_error': 'Missing original prompt',
            'generation_success': False,
            'extraction_success': False,
            'generation_timestamp': datetime.now().isoformat()
        }
    
    try:
        # Generate response
        generated_answer = generate_text(original_prompt, model, tokenizer, **generation_kwargs)
        
        # Extract letter choice
        extracted_letter = extract_letter_choice(generated_answer)
        
        return {
            **question_data,
            'generated_answer': generated_answer,
            'extracted_letter': extracted_letter,
            'generation_success': True,
            'extraction_success': extracted_letter is not None,
            'generation_timestamp': datetime.now().isoformat(),
            'model_name': MODEL_NAME,
            'generation_config': generation_kwargs
        }
        
    except Exception as e:
        return {
            **question_data,
            'generated_answer': '',
            'extracted_letter': None,
            'generation_error': str(e),
            'generation_success': False,
            'extraction_success': False,
            'generation_timestamp': datetime.now().isoformat()
        }


def main():
    """Main execution function."""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, HF_TOKEN)
    
    # Load changed questions dataset
    print("Loading changed questions dataset...")
    changed_questions = load_changed_questions(INPUT_FILE)
    print(f"Loaded {len(changed_questions)} changed questions")
    
    # Preview first question
    if changed_questions:
        print("\nFirst question preview:")
        first_q = changed_questions[0]
        print(f"Original line: {first_q.get('original_line_number', 'N/A')}")
        print(f"Question: {first_q.get('original_question', 'N/A')[:100]}...")
        print(f"Change: {first_q.get('change_type', 'N/A')}")
        print(f"Correct answer: {first_q.get('correct_answer', 'N/A')}")
        print(f"Biased hint: {first_q.get('biased_hint_letter', 'N/A')}")
    
    # Generation configuration
    generation_kwargs = {
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'do_sample': DO_SAMPLE,
        'pad_token_id': tokenizer.eos_token_id
    }
    
    print(f"\nGeneration configuration: {generation_kwargs}")
    print(f"Processing {len(changed_questions)} questions...")
    print("=" * 80)
    
    # Process all questions
    results = []
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(exist_ok=True)
    
    # Open output file for streaming writes
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for i, question_data in enumerate(tqdm(changed_questions, desc="Generating responses")):
            
            print(f"\nProcessing question {i+1}/{len(changed_questions)}:")
            print(f"  Original line: {question_data.get('original_line_number', 'N/A')}")
            print(f"  Question: {question_data.get('original_question', 'N/A')[:60]}...")
            print(f"  Change: {question_data.get('change_type', 'N/A')}")
            
            # Generate response
            start_time = time.time()
            result = process_question(question_data, model, tokenizer, generation_kwargs)
            generation_time = time.time() - start_time
            
            # Add timing info
            result['generation_time_seconds'] = generation_time
            
            # Preview generated answer
            if result.get('generation_success', False):
                preview = result['generated_answer'][:100] + "..." if len(result['generated_answer']) > 100 else result['generated_answer']
                extracted = result.get('extracted_letter', 'None')
                print(f"  Generated ({generation_time:.1f}s): {preview}")
                print(f"  Extracted letter: {extracted}")
            else:
                print(f"  Error: {result.get('generation_error', 'Unknown error')}")
            
            # Write to file immediately
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            outfile.flush()
            
            results.append(result)
    
    print("\n" + "=" * 80)
    print("Generation complete!")
    
    # Generate summary statistics
    total_questions = len(results)
    successful_generations = sum(1 for r in results if r.get('generation_success', False))
    failed_generations = total_questions - successful_generations
    successful_extractions = sum(1 for r in results if r.get('extraction_success', False))
    
    if successful_generations > 0:
        avg_generation_time = sum(r.get('generation_time_seconds', 0) for r in results) / successful_generations
        total_tokens_generated = sum(len(tokenizer.encode(r.get('generated_answer', ''))) for r in results if r.get('generation_success', False))
    else:
        avg_generation_time = 0
        total_tokens_generated = 0
    
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"Model used: {MODEL_NAME}")
    print(f"Total questions processed: {total_questions}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    print(f"Generation success rate: {successful_generations/total_questions*100:.1f}%")
    print(f"Successful letter extractions: {successful_extractions}")
    print(f"Letter extraction rate: {successful_extractions/successful_generations*100:.1f}%" if successful_generations > 0 else "N/A")
    print(f"Average generation time: {avg_generation_time:.2f} seconds")
    print(f"Total tokens generated: {total_tokens_generated:,}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Preview results
    print("\nPreview of generated results:")
    print("=" * 80)
    
    preview_count = 0
    for i, result in enumerate(results):
        if result.get('generation_success', False) and preview_count < 3:
            preview_count += 1
            print(f"\nExample {preview_count}:")
            print(f"Original line: {result.get('original_line_number', 'N/A')}")
            print(f"Question: {result.get('original_question', 'N/A')}")
            print(f"Change: {result.get('change_type', 'N/A')}")
            print(f"Correct: {result.get('correct_answer', 'N/A')}, Hint: {result.get('biased_hint_letter', 'N/A')}")
            print(f"Extracted: {result.get('extracted_letter', 'None')}")
            print(f"Generated answer (first 200 chars):")
            print(f"{result.get('generated_answer', 'N/A')[:200]}...")
            print("-" * 60)
    
    print(f"\nGeneration and analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()