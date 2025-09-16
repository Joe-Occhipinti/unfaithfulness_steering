# === FIXED: BATCH GENERATION FUNCTION ===
def generate_steered_responses_batch(prompts: list, layer_idx: int, coefficient: float) -> list:
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
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i+BATCH_SIZE]

            # Tokenize batch with padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,  # This already includes attention_mask
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                    # REMOVED: attention_mask=inputs.get('attention_mask') - this was causing the duplicate
                )

            # Decode each response in batch
            batch_responses = []
            for output in outputs:
                response = tokenizer.decode(output, skip_special_tokens=True)
                batch_responses.append(response)

            all_responses.extend(batch_responses)

            # Cleanup after each batch
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM Error with batch size {BATCH_SIZE}. Falling back to single processing...")
        target_wrapper.reset()  # Reset before fallback
        all_responses = []
        for prompt in prompts:
            response = generate_steered_response_single(prompt, layer_idx, coefficient)
            all_responses.append(response)

    except Exception as e:
        print(f"Error in batch generation: {e}")
        target_wrapper.reset()  # Reset before fallback
        all_responses = []
        for prompt in prompts:
            response = generate_steered_response_single(prompt, layer_idx, coefficient)
            all_responses.append(response)

    finally:
        target_wrapper.reset()

    return all_responses

# === ALSO FIX THE GENERATION FLAGS WARNING ===
# Remove temperature and top_p since you're using do_sample=False anyway

# In both batch and single generation functions, use this generate call:
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,  # This makes temperature/top_p irrelevant
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
    # No temperature, top_p, or duplicate attention_mask
)