# Modified generation function with batch processing

import torch
import json
from tqdm import tqdm

# === BATCH PROCESSING CONFIGURATION ===
BATCH_SIZE = 4  # Number of prompts to process concurrently - ADJUST THIS

def generate_steered_responses_batch(prompts: list, layer_idx: int, coefficient: float, batch_size: int = BATCH_SIZE) -> list:
    """Generate steered responses for a batch of prompts."""

    # Reset all wrappers
    for wrapper in wrapped_layers.values():
        wrapper.reset()

    # Set up steering for target layer
    target_wrapper = wrapped_layers[layer_idx]
    steering_vector = steering_vectors[layer_idx]
    target_wrapper.set_steering(steering_vector, coefficient)

    all_responses = []

    try:
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]

            # Tokenize batch with padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,  # Pad to same length in batch
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # Batch-specific settings
                    attention_mask=inputs.get('attention_mask'),
                    use_cache=True
                )

            # Decode each response in batch
            batch_responses = []
            for j, output in enumerate(outputs):
                response = tokenizer.decode(output, skip_special_tokens=True)
                batch_responses.append(response)

            all_responses.extend(batch_responses)

            # Cleanup after each batch
            del inputs, outputs
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in batch generation: {e}")
        # Fallback to single processing
        return [generate_steered_response_single(p, layer_idx, coefficient) for p in prompts]

    finally:
        target_wrapper.reset()

    return all_responses

def generate_steered_response_single(prompt: str, layer_idx: int, coefficient: float) -> str:
    """Fallback single prompt generation (your original function)."""
    # Keep your existing single-prompt function as fallback
    pass

# === MODIFIED MAIN PROCESSING LOOP ===
print("\n--- Generating steered responses with batching... ---")

for layer_idx in test_layers:
    for coefficient in STEERING_COEFFICIENTS:
        print(f"\nProcessing Layer {layer_idx}, Coefficient {coefficient}...")

        # Extract all prompts for this layer/coefficient combination
        prompts = []
        item_indices = []

        for i, item in enumerate(input_data):
            input_prompt = item.get(prompt_field, "")
            if input_prompt:
                prompts.append(input_prompt)
                item_indices.append(i)

        if not prompts:
            print("No valid prompts found!")
            continue

        # Generate responses in batches
        print(f"Processing {len(prompts)} prompts in batches of {BATCH_SIZE}...")
        steered_responses = generate_steered_responses_batch(
            prompts, layer_idx, coefficient, batch_size=BATCH_SIZE
        )

        # Create results
        results = []
        for idx, (item_idx, steered_response) in enumerate(zip(item_indices, steered_responses)):
            item = input_data[item_idx]

            result = {
                "item_index": item_idx,
                "steered_prompt_complete": steered_response,
                "biased_prompt_complete": item.get("biased_prompt", "") + item.get("generated_biased_answer", ""),
                "classification": item.get("classification", ""),
                "layer": layer_idx,
                "coefficient": coefficient,
                "model": model_id
            }
            results.append(result)

        # Save results
        coeff_str = f"pos{abs(coefficient)}" if coefficient > 0 else f"neg{abs(coefficient)}"
        output_filename = os.path.join(output_dir, f"steered_layer_{layer_idx}_coeff_{coeff_str}.jsonl")

        with open(output_filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Saved {len(results)} results to {output_filename}")

# === DYNAMIC BATCH SIZE CONFIGURATION ===
def auto_detect_batch_size():
    """Automatically detect optimal batch size based on GPU memory."""
    try:
        # Test with increasing batch sizes
        for test_batch_size in [1, 2, 4, 8, 16, 32]:
            try:
                test_prompts = ["Test prompt"] * test_batch_size
                inputs = tokenizer(test_prompts, return_tensors="pt", padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    # Try a forward pass
                    _ = model(**inputs)

                torch.cuda.empty_cache()
                optimal_batch_size = test_batch_size

            except torch.cuda.OutOfMemoryError:
                print(f"Batch size {test_batch_size} too large, using {optimal_batch_size}")
                return optimal_batch_size
            except Exception as e:
                print(f"Error testing batch size {test_batch_size}: {e}")
                return max(1, test_batch_size // 2)

        return optimal_batch_size

    except Exception:
        print("Auto-detection failed, using batch_size=1")
        return 1

# Usage:
# BATCH_SIZE = auto_detect_batch_size()  # Auto-detect
# BATCH_SIZE = 8  # Manual setting