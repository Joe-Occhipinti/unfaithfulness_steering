#!/usr/bin/env python3
"""
Fixed Steering Vector Application Code

This script applies steering vectors to model generation with proper error handling
and tensor dimension management.
"""

import torch
import json
import os
from tqdm import tqdm
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import time

# === 2. CONFIGURATION ===
# --- Model and File Settings ---
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# The JSONL file with the prompts you want to run steered generation on.
prompts_filename = 'val_changed_questions_2025-08-12.jsonl'
# The field inside your JSONL that contains the prompt text.
prompt_field = "original_prompt"
# The .pkl file containing your pre-computed steering vectors.
steering_vector_filename = "contrastive_activation_dataset_steering_vector.pkl"  # Fixed filename
# The output file for the steered generation results.
output_filename = "steered_results.jsonl"

# --- Steering Settings ---
# The layer you want to add the steering vector to.
TARGET_LAYER = 15
# The strength of the steering effect. Positive values steer towards 'faithful',
# negative values steer towards 'unfaithful'. Start with values between 1.0 and 3.0.
STEERING_COEFFICIENT = 2.0
batch_size = 1 # A safe batch size to prevent freezes
max_new_tokens = 512 # Reduced for stability

# === 3. LOAD ALL COMPONENTS ===
print("--- Loading model, tokenizer, and steering vector... ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval() # Set to evaluation mode for consistent results

# Set a pad token if one doesn't exist (crucial for batching)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer pad_token set to eos_token.")

# Load the steering vector
print(f"--- Inspecting file: {steering_vector_filename} ---")
with open(steering_vector_filename, 'rb') as f:
    loaded_object = pickle.load(f)

print("Type of loaded object:", type(loaded_object))

if isinstance(loaded_object, dict):
    print("Available keys in your file:", list(loaded_object.keys()))
    
    # Get the vector from the dictionary
    steering_vector_numpy = loaded_object.get('steering_vector')
    if steering_vector_numpy is None:
        raise ValueError("The key 'steering_vector' was not found in your .pkl file.")
        
    # Convert the NumPy array to a PyTorch tensor
    steering_vector = torch.from_numpy(steering_vector_numpy).float()
    print(f"Original steering vector shape: {steering_vector.shape}")
    
else:
    print("The loaded object is not a dictionary. Using it directly as steering vector.")
    steering_vector = torch.from_numpy(loaded_object).float()

# Scale the vector by the coefficient and move to device
scaled_steering_vector = (steering_vector * STEERING_COEFFICIENT).to(device)
print(f"Scaled steering vector shape: {scaled_steering_vector.shape}")
print(f"Device: {scaled_steering_vector.device}")

# === 4. IMPROVED WRAPPER CLASS ===
print("--- Preparing model for activation steering... ---")

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.add_activations = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        
        if self.add_activations is not None:
            try:
                if isinstance(output, tuple):
                    # Get the hidden states (first element of tuple)
                    hidden_states = output[0]
                    
                    # Handle dimension mismatch
                    if hidden_states.dim() == 3 and self.add_activations.dim() == 1:
                        # Broadcast steering vector to match batch and sequence dimensions
                        # hidden_states shape: [batch_size, seq_len, hidden_dim]
                        # steering_vector shape: [hidden_dim]
                        
                        # Add steering vector to the last token only (most common approach)
                        batch_size, seq_len, hidden_dim = hidden_states.shape
                        
                        # Clone to avoid in-place operations
                        modified_hidden_states = hidden_states.clone()
                        
                        # Add steering vector to last token of each sequence in batch
                        modified_hidden_states[:, -1, :] = (
                            modified_hidden_states[:, -1, :] + 
                            self.add_activations.unsqueeze(0).expand(batch_size, -1)
                        )
                        
                        # Reconstruct the output tuple
                        output = (modified_hidden_states,) + output[1:]
                        
                    elif hidden_states.dim() == 2 and self.add_activations.dim() == 1:
                        # hidden_states shape: [batch_size, hidden_dim]
                        # steering_vector shape: [hidden_dim]
                        batch_size = hidden_states.shape[0]
                        modified_hidden_states = (
                            hidden_states + 
                            self.add_activations.unsqueeze(0).expand(batch_size, -1)
                        )
                        output = (modified_hidden_states,) + output[1:]
                        
                    else:
                        print(f"Warning: Dimension mismatch. Hidden states: {hidden_states.shape}, "
                              f"Steering vector: {self.add_activations.shape}")
                        
                else:
                    # Direct tensor case (less common)
                    if output.dim() == 3 and self.add_activations.dim() == 1:
                        batch_size, seq_len, hidden_dim = output.shape
                        output = output.clone()
                        output[:, -1, :] = (
                            output[:, -1, :] + 
                            self.add_activations.unsqueeze(0).expand(batch_size, -1)
                        )
                    else:
                        print(f"Warning: Unexpected output format or dimension mismatch.")
                        
            except Exception as e:
                print(f"Error in steering application: {e}")
                print(f"Output type: {type(output)}")
                if isinstance(output, tuple) and len(output) > 0:
                    print(f"Hidden states shape: {output[0].shape}")
                print(f"Steering vector shape: {self.add_activations.shape}")
                
        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None

# Wrap the target layer only (more efficient)
target_layer = model.model.layers[TARGET_LAYER]
model.model.layers[TARGET_LAYER] = BlockOutputWrapper(target_layer)
print(f"Wrapped layer {TARGET_LAYER} successfully.")

# === 5. DATA LOADING ===
def load_prompts_from_file(filename, field_name):
    prompts_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if field_name in data:
                    prompts_list.append(data[field_name])
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in {filename}")
    return prompts_list

print(f"--- Loading prompts from {prompts_filename}... ---")
prompts_to_run = load_prompts_from_file(prompts_filename, prompt_field)
print(f"Loaded {len(prompts_to_run)} prompts to process.")

if len(prompts_to_run) == 0:
    print("ERROR: No prompts loaded! Check your file path and field name.")
    exit(1)

# === 6. STEERED GENERATION LOOP ===
all_answers = []
print(f"--- Starting steered generation for {len(prompts_to_run)} prompts ---")
start_time = time.time()

# Get a direct handle to our wrapped target layer
target_layer_wrapper = model.model.layers[TARGET_LAYER]

for i, prompt in enumerate(tqdm(prompts_to_run, desc="Generating Steered Responses")):
    try:
        print(f"Processing prompt {i+1}/{len(prompts_to_run)}")
        
        # Arm the wrapper with our steering vector before generation
        target_layer_wrapper.add(scaled_steering_vector)

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Reset the wrapper to a clean state for the next prompt
        target_layer_wrapper.reset()

        # Decode and save the result
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        all_answers.append(answer.strip())
        
        print(f"Generated answer length: {len(answer)} characters")

        # Memory cleanup for long runs
        del inputs, outputs, generated_tokens
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing prompt {i+1}: {e}")
        all_answers.append("")  # Add empty answer to maintain alignment
        # Reset wrapper even on error
        target_layer_wrapper.reset()
        continue

end_time = time.time()
print(f"--- Steered generation complete! ---")
print(f"Total time taken: {end_time - start_time:.2f} seconds")

# === 7. SAVE RESULTS ===
print(f"--- Saving steered results to {output_filename} ---")
results_to_save = []
for i, answer in enumerate(all_answers):
    results_to_save.append({
        "prompt_index": i,
        "original_prompt": prompts_to_run[i],
        "steered_answer": answer,
        "steering_coefficient": STEERING_COEFFICIENT,
        "target_layer": TARGET_LAYER,
        "model": model_id
    })

with open(output_filename, 'w', encoding='utf-8') as f:
    for result in results_to_save:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Successfully saved {len(results_to_save)} results to {output_filename}")

# Preview results
print("\n--- Preview of results ---")
for i in range(min(2, len(all_answers))):
    print(f"Result {i+1}:")
    print(f"Answer: {all_answers[i][:200]}...")
    print("-" * 50)