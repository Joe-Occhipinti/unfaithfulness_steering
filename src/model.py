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
from .config import ModelConfig


# =============================================================================
# HuggingFace Backend (Original)
# =============================================================================

def load_model(model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", attn_implementation: str = "eager") -> Tuple[Any, Any]:
    """
    Load model and tokenizer with BF16 precision.
    Reusable across all evaluation scripts.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n--- Loading model: {model_id} ---")

    # Resolve model ID if short name provided
    model_id = ModelConfig.get_model_id(model_id)
    print(f"Resolved ID: {model_id}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")

    # Load model with BF16 precision
    # Note: BF16 requires Ampere GPUs (e.g., A100, RTX 30-series) or newer.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Set precision to BF16
        device_map="auto",           # Automatically handles multi-GPU/CPU offloading
        attn_implementation=attn_implementation
    )

    print(f"Model loaded successfully in BF16")
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


# =============================================================================
# vLLM Backend (High-Performance)
# =============================================================================

def load_model_vllm(
    model_id: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 3072
) -> Any:
    """
    Load model using vLLM for high-performance inference.
    Recommended for large models (32B+) that cause OOM with HuggingFace.
    
    Args:
        model_id: HuggingFace model identifier
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
        max_model_len: Maximum sequence length (input + output). Default 3072 = 1024 + 2048.
        
    Returns:
        vLLM LLM instance
    """
    from vllm import LLM
    
    print(f"\n--- Loading model with vLLM: {model_id} ---")
    
    # Resolve model ID if short name provided
    model_id = ModelConfig.get_model_id(model_id)
    print(f"Resolved ID: {model_id}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  Max model len: {max_model_len}")
    
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,  # Recommended for stability
        max_model_len=max_model_len,
    )
    
    print(f"Model loaded successfully with vLLM")
    return llm


def load_model_easysteer(
    model_id: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 3072
) -> Any:
    """
    Load model using vLLM with EasySteer steering vector support.
    
    This enables applying steering vectors during inference via SteerVectorRequest.
    
    Args:
        model_id: HuggingFace model identifier
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
        max_model_len: Maximum sequence length (input + output). Default 3072 = 1024 + 2048.
        
    Returns:
        vLLM LLM instance with steering vector support enabled
    """
    from vllm import LLM
    
    print(f"\n--- Loading model with EasySteer + vLLM: {model_id} ---")
    
    # Resolve model ID if short name provided
    model_id = ModelConfig.get_model_id(model_id)
    print(f"Resolved ID: {model_id}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  Max model len: {max_model_len}")
    print(f"  Steering vectors: ENABLED")
    
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        enable_steer_vector=True,      # EasySteer key flag
        enforce_eager=True,             # Required for reliable steering
        enable_chunked_prefill=False,   # Required for steering compatibility
        max_model_len=max_model_len,
    )
    
    print(f"Model loaded successfully with EasySteer steering support")
    return llm


def batch_generate_vllm(
    llm: Any,
    prompts: List[str],
    max_new_tokens: int = 2048,
) -> List[str]:
    """
    Generate text using vLLM's optimized inference engine.
    vLLM handles batching and memory management automatically via continuous batching.
    
    Args:
        llm: vLLM LLM instance
        prompts: List of input prompts
        max_new_tokens: Maximum new tokens to generate
        
    Returns:
        List of generated text responses (in same order as prompts)
    """
    from vllm import SamplingParams
    
    print(f"\n--- Starting vLLM generation for {len(prompts)} prompts ---")
    
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,  # Deterministic (greedy decoding)
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract generated text from each output
    all_answers = [output.outputs[0].text.strip() for output in outputs]
    
    print(f"Generation complete: {len(all_answers)} responses generated")
    return all_answers


# =============================================================================
# Other Utilities
# =============================================================================

def load_model_for_forward_pass(model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") -> Tuple[Any, Any]:
    """
    Load model and tokenizer for single forward passes (activation extraction).
    Uses the same base loading as generation but optimized for forward passes.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    # Use the same base loading logic with SDPA for memory efficiency
    model, tokenizer = load_model(model_id, attn_implementation="sdpa")

    # Set to eval mode for forward passes (no generation)
    model.eval()
    print("Model set to eval mode for forward passes (using SDPA)")

    return model, tokenizer


def sample_generate_multiple(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_samples: int = 50,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    max_input_length: int = 1024,
    batch_size: int = 10
) -> List[str]:
    """
    Generate multiple sampled responses for a single prompt.
    Generates in batches for memory efficiency.

    Unlike batch_generate() which processes multiple different prompts deterministically,
    this function generates multiple sampled responses for the same prompt using temperature.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Single input prompt to generate from
        num_samples: Total number of samples to generate (default: 50)
        temperature: Sampling temperature (default: 0.7)
        max_new_tokens: Maximum new tokens to generate per sample
        max_input_length: Maximum input sequence length
        batch_size: How many samples to generate concurrently (default: 10)
                   Can be independent from num_samples (e.g., batch_size=10 for 50 samples = 5 batches)

    Returns:
        List of num_samples generated text responses

    Example:
        >>> # Generate 50 samples with batch_size=10 (5 batches of 10)
        >>> samples = sample_generate_multiple(
        ...     model, tokenizer,
        ...     prompt="What is 2+2?",
        ...     num_samples=50,
        ...     temperature=0.7,
        ...     batch_size=10
        ... )
        >>> len(samples)  # 50
    """
    print(f"\n--- Generating {num_samples} samples for prompt (batch_size={batch_size}, temp={temperature}) ---")

    all_samples = []

    # Calculate number of batches needed
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

    for batch_idx in tqdm(range(num_batches), desc="Sampling batches"):
        # How many samples in this batch? (last batch might be smaller)
        current_batch_size = min(batch_size, num_samples - len(all_samples))

        # Tokenize: duplicate the prompt current_batch_size times
        inputs = tokenizer(
            [prompt] * current_batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length
        ).to(model.device)

        # Generate with sampling (key differences from batch_generate)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Enable sampling (vs deterministic in batch_generate)
                temperature=temperature,  # Use temperature for sampling
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode generated text (skip input tokens)
        input_length = inputs['input_ids'].shape[1]
        batch_samples = tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True
        )

        all_samples.extend([sample.strip() for sample in batch_samples])

        # Memory cleanup
        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Sampling complete: {len(all_samples)} responses generated")
    return all_samples