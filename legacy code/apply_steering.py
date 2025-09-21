"""
COMPLETE FIXED steering notebook with batch processing + DEBUG VERSION
All issues resolved: duplicate attention_mask, generation flags, error handling
ADDED: Extensive debugging to track coefficient application
"""

#!pip install -U torch transformers bitsandbytes

import torch
import json
import os
from tqdm import tqdm
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pickle
import time

# === CONFIGURATION ===
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
input_filename = r"/content/val_biased_answers_2025-08-12.jsonl"
prompt_field = "biased_prompt"
steering_vectors_filename = r"/content/steering_vectors_body_only_faithful_vs_unfaithful_2025-09-15.pkl"
output_dir = "last_try_steered_prompts_all_strongly_faithful_vs_unfaithful_mmlu_psy_val_2025-09-14_DEBUGGED"

# Steering settings
STEERING_COEFFICIENTS = [1.0, -1.0]
max_new_tokens = 2048

# === BATCH PROCESSING CONFIGURATION ===
BATCH_SIZE = 6  # Start with 6, adjust based on your GPU memory

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("=== STEERED RESPONSE GENERATION WITH BATCHING + DEBUG ===")
print(f"Input: {input_filename}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Coefficients: {STEERING_COEFFICIENTS}")
print(f"Output directory: {output_dir}")

# === LOAD STEERING VECTORS ===
print("--- Loading steering vectors... ---")
with open(steering_vectors_filename, 'rb') as f:
    steering_data = pickle.load(f)

steering_vectors = steering_data['steering_vectors']
test_layers = sorted(steering_vectors.keys())[25:26]  # Test just layer 25 for now
print(f"Loaded vectors for layers: {test_layers}")

# === STEERING WRAPPER WITH DEBUG ===
class LayerSteeringWrapper(torch.nn.Module):
    def __init__(self, block, layer_idx):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.steering_vector = None
        self.coefficient = 0.0
        self.active = False
        self.forward_count = 0  # DEBUG: Count how many times forward is called

    def forward(self, *args, **kwargs):
        self.forward_count += 1
        output = self.block(*args, **kwargs)

        # DEBUG: Check what the original layer returned
        if self.forward_count <= 3:
            print(f"    DEBUG Layer {self.layer_idx}: output type = {type(output)}, output = {output is not None}")

        if self.active and self.steering_vector is not None and output is not None:
            try:
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    # Output is a tensor directly
                    hidden_states = output

                if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                        batch_size, seq_len, hidden_dim = hidden_states.shape
                        modified_hidden_states = hidden_states.clone()

                        # Add to last token
                        steering_addition = self.steering_vector * self.coefficient

                        # DEBUG: Print steering info for first few calls
                        if self.forward_count <= 3:
                            print(f"    *** STEERING APPLIED *** Layer {self.layer_idx} call #{self.forward_count}: coeff={self.coefficient:.1f}, "
                                  f"vector_norm={torch.norm(self.steering_vector):.3f}, "
                                  f"addition_norm={torch.norm(steering_addition):.3f}")

                        # DEBUG: Check before/after values for first few calls
                        if self.forward_count <= 3:
                            original_norm = torch.norm(modified_hidden_states[:, -1, :])
                            print(f"    DEBUG Layer {self.layer_idx} call #{self.forward_count}: "
                                  f"original_norm={original_norm:.3f}, addition_norm={torch.norm(steering_addition):.3f}")
                            print(f"    DEBUG: steering_vector device={self.steering_vector.device}, "
                                  f"hidden_states device={modified_hidden_states.device}")

                        modified_hidden_states[:, -1, :] = (
                            modified_hidden_states[:, -1, :] +
                            steering_addition.unsqueeze(0).expand(batch_size, -1)
                        )

                        # DEBUG: Check after modification for first few calls
                        if self.forward_count <= 3:
                            modified_norm = torch.norm(modified_hidden_states[:, -1, :])
                            print(f"    DEBUG Layer {self.layer_idx} call #{self.forward_count}: "
                                  f"modified_norm={modified_norm:.3f}, change={modified_norm - original_norm:.3f}")

                        # Return in the same format as input
                        if isinstance(output, tuple):
                            output = (modified_hidden_states,) + output[1:]
                        else:
                            output = modified_hidden_states

                elif hidden_states.dim() == 2:  # [batch_size, hidden_dim]
                    batch_size = hidden_states.shape[0]
                    steering_addition = self.steering_vector * self.coefficient

                    # DEBUG: Print steering info for first few calls
                    if self.forward_count <= 3:
                        print(f"    *** STEERING APPLIED (2D) *** Layer {self.layer_idx} call #{self.forward_count}: coeff={self.coefficient:.1f}")

                    modified_hidden_states = (
                        hidden_states +
                        steering_addition.unsqueeze(0).expand(batch_size, -1)
                    )
                    # Return in the same format as input
                    if isinstance(output, tuple):
                        output = (modified_hidden_states,) + output[1:]
                    else:
                        output = modified_hidden_states

            except Exception as e:
                print(f"Error in layer {self.layer_idx} steering: {e}")

        return output

    def set_steering(self, vector: torch.Tensor, coefficient: float):
        self.steering_vector = vector.to(device)
        self.coefficient = coefficient
        self.active = True
        self.forward_count = 0  # Reset counter
        print(f"    DEBUG: Set steering for layer {self.layer_idx}: coeff={coefficient:.1f}, "
              f"vector_norm={torch.norm(vector):.3f}, active={self.active}")

    def reset(self):
        import traceback
        print(f"    DEBUG: Reset layer {self.layer_idx} (was coeff={self.coefficient:.1f})")
        print(f"    DEBUG: Reset called from:")
        for line in traceback.format_stack()[-3:-1]:  # Show last 2 stack frames
            print(f"      {line.strip()}")
        self.active = False
        self.steering_vector = None
        self.coefficient = 0.0
        self.forward_count = 0

# Wrap all test layers
wrapped_layers = {}
for layer_idx in test_layers:
    original_layer = model.model.layers[layer_idx]
    wrapped_layer = LayerSteeringWrapper(original_layer, layer_idx)
    model.model.layers[layer_idx] = wrapped_layer
    wrapped_layers[layer_idx] = wrapped_layer

print(f"Wrapped {len(wrapped_layers)} layers")

# DEBUG: Verify wrapping worked
print("--- VERIFYING WRAPPER INSTALLATION ---")
for layer_idx in test_layers:
    actual_layer = model.model.layers[layer_idx]
    is_wrapped = isinstance(actual_layer, LayerSteeringWrapper)
    print(f"Layer {layer_idx}: {type(actual_layer).__name__}, wrapped={is_wrapped}")
    if is_wrapped:
        print(f"  Original layer type: {type(actual_layer.block).__name__}")

# DEBUG: Test if wrapper forward is called on simple forward pass
print("--- TESTING WRAPPER FORWARD CALL ---")
test_layer_idx = test_layers[0]
wrapper = wrapped_layers[test_layer_idx]
wrapper.set_steering(steering_vectors[test_layer_idx], 1.0)

print(f"Testing forward call on layer {test_layer_idx}...")
test_input = torch.randn(1, 10, 4096).to(device).to(torch.bfloat16)
print(f"Forward count before: {wrapper.forward_count}")

with torch.no_grad():
    try:
        # Directly call the wrapper
        output = wrapper(test_input)
        print(f"Forward count after direct call: {wrapper.forward_count}")
    except Exception as e:
        print(f"Error in direct call: {e}")

wrapper.reset()

# === LOAD INPUT DATA ===
def load_input_data(filename: str) -> list:
    data_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line")
    return data_list

print(f"\n--- Loading input data from {input_filename}... ---")
input_data = load_input_data(input_filename)
print(f"Loaded {len(input_data)} items")

if len(input_data) == 0:
    print("ERROR: No data loaded!")
    exit(1)

# === FIXED: BATCH GENERATION FUNCTION WITH DEBUG ===
def generate_steered_responses_batch(prompts: list, layer_idx: int, coefficient: float) -> list:
    """Generate steered responses for a batch of prompts - FIXED VERSION WITH DEBUG."""

    print(f"  DEBUG: Starting batch generation with layer={layer_idx}, coeff={coefficient:.1f}")

    # Reset all wrappers
    for wrapper in wrapped_layers.values():
        wrapper.reset()

    # Set up steering for target layer
    target_wrapper = wrapped_layers[layer_idx]
    steering_vector = steering_vectors[layer_idx]
    target_wrapper.set_steering(steering_vector, coefficient)

    # CRITICAL DEBUG: Verify the coefficient was set correctly
    print(f"  DEBUG: After setting, target_wrapper.coefficient = {target_wrapper.coefficient:.1f}")
    print(f"  DEBUG: target_wrapper.active = {target_wrapper.active}")

    all_responses = []

    try:
        # Process prompts in batches
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i+BATCH_SIZE]

            print(f"  Processing batch {i//BATCH_SIZE + 1}: {len(batch_prompts)} prompts")

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
                # FIXED: Remove duplicate attention_mask and invalid generation flags
                outputs = model.generate(
                    **inputs,  # This already includes input_ids, attention_mask, etc.
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic generation (no temperature/top_p needed)
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                    # REMOVED: attention_mask (was duplicate)
                    # REMOVED: temperature, top_p (not valid with do_sample=False)
                )

            # Decode each response in batch
            batch_responses = []
            for output in outputs:
                response = tokenizer.decode(output, skip_special_tokens=True)
                batch_responses.append(response)

            all_responses.extend(batch_responses)

            # DEBUG: Verify wrapper state after generation
            print(f"    DEBUG: After batch, target_wrapper.coefficient = {target_wrapper.coefficient:.1f}")

            # Cleanup after each batch
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM Error with batch size {BATCH_SIZE}: {e}")
        print("Falling back to single processing...")
        # Clean up and reset before fallback
        target_wrapper.reset()
        gc.collect()
        torch.cuda.empty_cache()

        # Fallback to single processing
        all_responses = []
        for j, prompt in enumerate(prompts):
            print(f"  Single processing prompt {j+1}/{len(prompts)}")
            response = generate_steered_response_single(prompt, layer_idx, coefficient)
            all_responses.append(response)

    except Exception as e:
        print(f"Error in batch generation: {e}")
        print("Falling back to single processing...")
        # Clean up and reset before fallback
        target_wrapper.reset()
        gc.collect()
        torch.cuda.empty_cache()

        # Fallback to single processing
        all_responses = []
        for j, prompt in enumerate(prompts):
            print(f"  Single processing prompt {j+1}/{len(prompts)}")
            response = generate_steered_response_single(prompt, layer_idx, coefficient)
            all_responses.append(response)

    finally:
        # Always reset steering
        target_wrapper.reset()
        print(f"  DEBUG: Finished batch generation, reset layer {layer_idx}")

    return all_responses

# === FIXED: SINGLE GENERATION FUNCTION WITH DEBUG ===
def generate_steered_response_single(prompt: str, layer_idx: int, coefficient: float) -> str:
    """Generate a steered response for single prompt - FIXED VERSION WITH DEBUG."""

    print(f"    DEBUG: Single generation with layer={layer_idx}, coeff={coefficient:.1f}")

    # Reset all wrappers
    for wrapper in wrapped_layers.values():
        wrapper.reset()

    # Set up steering for target layer
    target_wrapper = wrapped_layers[layer_idx]
    steering_vector = steering_vectors[layer_idx]
    target_wrapper.set_steering(steering_vector, coefficient)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # FIXED: Clean generation parameters
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
                # REMOVED: temperature, top_p, duplicate parameters
            )

        # Get complete response
        complete_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cleanup
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        return complete_response

    except Exception as e:
        print(f"Error in single generation: {e}")
        return ""
    finally:
        # Always reset steering
        target_wrapper.reset()

# === MAIN PROCESSING LOOP WITH DEBUG ===
print("\n--- Generating steered responses with batching + DEBUG... ---")

for layer_idx in test_layers:
    for coefficient in STEERING_COEFFICIENTS:
        print(f"\n=== PROCESSING Layer {layer_idx}, Coefficient {coefficient} ===")

        # Extract all prompts and metadata for this layer/coefficient
        prompts = []
        items_data = []

        for i, item in enumerate(input_data):
            input_prompt = item.get(prompt_field, "")
            if input_prompt:
                prompts.append(input_prompt)
                items_data.append((i, item))

        if not prompts:
            print("No valid prompts found!")
            continue

        # Generate responses in batches
        print(f"Processing {len(prompts)} prompts in batches of {BATCH_SIZE}...")
        steered_responses = generate_steered_responses_batch(prompts, layer_idx, coefficient)

        # Verify we got all responses
        if len(steered_responses) != len(prompts):
            print(f"WARNING: Expected {len(prompts)} responses, got {len(steered_responses)}")

        # Create results
        results = []
        for idx, ((item_idx, item), steered_response) in enumerate(zip(items_data, steered_responses)):
            biased_prompt = item.get("biased_prompt", "")
            generated_biased_answer = item.get("generated_biased_answer", "")
            classification = item.get("classification", "")

            result = {
                "item_index": item_idx,
                "steered_prompt_complete": steered_response,
                "biased_prompt_complete": biased_prompt + generated_biased_answer,
                "classification": classification,
                "layer": layer_idx,
                "coefficient": coefficient,
                "model": model_id,
                # DEBUG: Add debug info
                "debug_coefficient_used": coefficient,
                "debug_layer_used": layer_idx
            }
            results.append(result)

        # Save JSONL file for this combination
        coeff_str = f"pos{abs(coefficient)}" if coefficient > 0 else f"neg{abs(coefficient)}"
        output_filename = os.path.join(output_dir, f"steered_layer_{layer_idx}_coeff_{coeff_str}.jsonl")

        with open(output_filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Saved {len(results)} results to {output_filename}")

        # DEBUG: Show first response snippet for verification
        if len(results) > 0:
            first_response = results[0]['steered_prompt_complete']
            print(f"DEBUG: First response snippet: {first_response[:100]}...")

print(f"\n=== GENERATION COMPLETE ===")
print(f"Output files saved to {output_dir}/")
print(f"Generated {len(test_layers) * len(STEERING_COEFFICIENTS)} files total")

# === DOWNLOAD RESULTS ===
import shutil

print("\nPreparing results for download...")

# Check what's actually in /content/
print("Contents of /content/:")
for item in os.listdir('/content/'):
    if os.path.isdir(os.path.join('/content/', item)):
        print(f"  [DIR] {item}")
    else:
        print(f"  [FILE] {item}")

print(f"\nOutput directory exists: {os.path.exists(output_dir)}")

if os.path.exists(output_dir):
    output_contents = os.listdir(output_dir)
    print(f"Output directory contents: {len(output_contents)} files")

    # Show first few files
    for i, filename in enumerate(output_contents[:5]):
        print(f"  {filename}")
    if len(output_contents) > 5:
        print(f"  ... and {len(output_contents) - 5} more files")

    # Create a zip file of the entire output directory
    print("\nCreating zip archive...")
    archive_path = shutil.make_archive('/content/steering_results_batch_debugged', 'zip', output_dir)
    print(f"Archive created: {archive_path}")

    # Download
    from google.colab import files
    print("Starting download...")
    files.download(archive_path)
    print("Download initiated!")

else:
    print("Output directory doesn't exist - looking for similar names:")
    for item in os.listdir('/content/'):
        if 'steered' in item.lower() or 'fullsweep' in item.lower():
            print(f"  Found: {item}")