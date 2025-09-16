"""
Modified version of your steering notebook with batch processing
"""

!pip install -U torch transformers bitsandbytes

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
steering_vectors_filename = r"/content/steering_vectors_faithful_vs_unfaithful_2025-09-04.pkl"
output_dir = "fullsweep_layerwise_steered_prompts_all_strongly_faithful_vs_unfaithful_mmlu_psy_val_2025-09-14"

# Steering settings
STEERING_COEFFICIENTS = [5.0, -5.0]
max_new_tokens = 2048

# === NEW: BATCH PROCESSING CONFIGURATION ===
BATCH_SIZE = 4  # Start with 4, increase based on your GPU memory

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("=== STEERED RESPONSE GENERATION WITH BATCHING ===")
print(f"Input: {input_filename}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Coefficients: {STEERING_COEFFICIENTS}")
print(f"Output directory: {output_dir}")

# === LOAD MODEL AND TOKENIZER ===
print("\n--- Loading model and tokenizer... ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === LOAD STEERING VECTORS ===
print("--- Loading steering vectors... ---")
with open(steering_vectors_filename, 'rb') as f:
    steering_data = pickle.load(f)

steering_vectors = steering_data['steering_vectors']
test_layers = sorted(steering_vectors.keys())
print(f"Loaded vectors for layers: {test_layers}")

# === STEERING WRAPPER (UNCHANGED) ===
class LayerSteeringWrapper(torch.nn.Module):
    def __init__(self, block, layer_idx):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.steering_vector = None
        self.coefficient = 0.0
        self.active = False

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if self.active and self.steering_vector is not None:
            try:
                if isinstance(output, tuple):
                    hidden_states = output[0]

                    if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                        batch_size, seq_len, hidden_dim = hidden_states.shape
                        modified_hidden_states = hidden_states.clone()

                        # Add to last token
                        steering_addition = self.steering_vector * self.coefficient
                        modified_hidden_states[:, -1, :] = (
                            modified_hidden_states[:, -1, :] +
                            steering_addition.unsqueeze(0).expand(batch_size, -1)
                        )

                        output = (modified_hidden_states,) + output[1:]

                    elif hidden_states.dim() == 2:  # [batch_size, hidden_dim]
                        batch_size = hidden_states.shape[0]
                        steering_addition = self.steering_vector * self.coefficient
                        modified_hidden_states = (
                            hidden_states +
                            steering_addition.unsqueeze(0).expand(batch_size, -1)
                        )
                        output = (modified_hidden_states,) + output[1:]

            except Exception as e:
                print(f"Error in layer {self.layer_idx} steering: {e}")

        return output

    def set_steering(self, vector: torch.Tensor, coefficient: float):
        self.steering_vector = vector.to(device)
        self.coefficient = coefficient
        self.active = True

    def reset(self):
        self.active = False

# Wrap all test layers
wrapped_layers = {}
for layer_idx in test_layers:
    original_layer = model.model.layers[layer_idx]
    wrapped_layer = LayerSteeringWrapper(original_layer, layer_idx)
    model.model.layers[layer_idx] = wrapped_layer
    wrapped_layers[layer_idx] = wrapped_layer

print(f"Wrapped {len(wrapped_layers)} layers")

# === LOAD INPUT DATA (UNCHANGED) ===
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

# === NEW: BATCH GENERATION FUNCTION ===
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
                padding=True,  # KEY CHANGE: Enable padding for batch processing
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
                    attention_mask=inputs.get('attention_mask')  # Important for padded sequences
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
        # Fallback to single processing
        all_responses = []
        for prompt in prompts:
            response = generate_steered_response_single(prompt, layer_idx, coefficient)
            all_responses.append(response)

    except Exception as e:
        print(f"Error in batch generation: {e}")
        # Fallback to single processing
        all_responses = []
        for prompt in prompts:
            response = generate_steered_response_single(prompt, layer_idx, coefficient)
            all_responses.append(response)

    finally:
        target_wrapper.reset()

    return all_responses

# === FALLBACK: SINGLE GENERATION FUNCTION ===
def generate_steered_response_single(prompt: str, layer_idx: int, coefficient: float) -> str:
    """Generate a steered response for single prompt (fallback function)."""

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
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Get complete response
        complete_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cleanup
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        return complete_response

    except Exception as e:
        print(f"Error in generation: {e}")
        return ""
    finally:
        target_wrapper.reset()

# === MODIFIED MAIN PROCESSING LOOP ===
print("\n--- Generating steered responses with batching... ---")

for layer_idx in test_layers:
    for coefficient in STEERING_COEFFICIENTS:
        print(f"\nProcessing Layer {layer_idx}, Coefficient {coefficient}...")

        # Extract all prompts and metadata for this layer/coefficient
        prompts = []
        items_data = []

        for i, item in enumerate(input_data):
            input_prompt = item.get(prompt_field, "")
            if input_prompt:
                prompts.append(input_prompt)
                items_data.append((i, item))  # Store index and full item data

        if not prompts:
            print("No valid prompts found!")
            continue

        # Generate responses in batches
        print(f"Processing {len(prompts)} prompts in batches of {BATCH_SIZE}...")
        steered_responses = generate_steered_responses_batch(prompts, layer_idx, coefficient)

        # Create results
        results = []
        for (item_idx, item), steered_response in zip(items_data, steered_responses):
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
                "model": model_id
            }
            results.append(result)

        # Save JSONL file for this combination
        coeff_str = f"pos{abs(coefficient)}" if coefficient > 0 else f"neg{abs(coefficient)}"
        output_filename = os.path.join(output_dir, f"steered_layer_{layer_idx}_coeff_{coeff_str}.jsonl")

        with open(output_filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Saved {len(results)} results to {output_filename}")

print(f"\n=== GENERATION COMPLETE ===")
print(f"Output files saved to {output_dir}/")
print(f"Generated {len(test_layers) * len(STEERING_COEFFICIENTS)} files total")

# === DOWNLOAD RESULTS (UNCHANGED) ===
import shutil
import os

# Check what's actually in /content/
print("Contents of /content/:")
print(os.listdir('/content/'))

print(f"\nDirectory exists: {os.path.exists(output_dir)}")

if os.path.exists(output_dir):
    print(f"Directory contents: {len(os.listdir(output_dir))} items")

    # Create a zip file of the entire output directory
    archive_path = shutil.make_archive('/content/steering_results', 'zip', output_dir)

    # Download
    from google.colab import files
    files.download(archive_path)

else:
    print("Directory doesn't exist - looking for similar names:")
    for item in os.listdir('/content/'):
        if 'steered' in item.lower() or 'fullsweep' in item.lower():
            print(f"  Found: {item}")