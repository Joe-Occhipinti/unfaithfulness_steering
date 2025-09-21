"""
model.py

Model loading and generation utilities for faithfulness steering workflow.
Reusable across baseline, hinted, and steering evaluation scripts.
"""

import torch
import gc
from typing import List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_model(model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") -> Tuple[Any, Any]:
    """
    Load model and tokenizer with optimized settings.
    Reusable across all evaluation scripts.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n--- Loading model: {model_id} ---")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")

    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Model loaded successfully")
    return model, tokenizer

def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    batch_size: int = 5,
    max_new_tokens: int = 1024,
    max_input_length: int = 1024
) -> List[str]:
    """
    Generate text for list of prompts using batching and memory management.
    Reusable across all evaluation scripts.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompts: List of input prompts
        batch_size: Batch size for generation
        max_new_tokens: Maximum new tokens to generate
        max_input_length: Maximum input sequence length

    Returns:
        List of generated text responses
    """
    print(f"\n--- Starting generation with batch size {batch_size} ---")

    all_answers = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Batches"):

        # Current batch
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length
        ).to(model.device)

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode generated text (skip input)
        input_length = inputs['input_ids'].shape[1]
        batch_answers = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)

        all_answers.extend([answer.strip() for answer in batch_answers])

        # Memory cleanup
        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Generation complete: {len(all_answers)} responses generated")
    return all_answers