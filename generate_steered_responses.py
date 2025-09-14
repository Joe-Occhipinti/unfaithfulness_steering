"""
generate_steered_responses.py

Generates steered responses for each layer-coefficient combination and saves
JSONL files with complete prompts and preserved classification data.
"""

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
input_filename = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\datasets\input_file.jsonl"  # Update path
prompt_field = "prompt"  # Field containing the input prompt
steering_vectors_filename = "steering_vectors_faithful_vs_unfaithful_2025-09-04.pkl"
output_dir = "steered_responses"

# Steering settings
STEERING_COEFFICIENTS = [5.0, -5.0]  # positive=faithful, negative=unfaithful
max_new_tokens = 512

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("=== STEERED RESPONSE GENERATION ===")
print(f"Input: {input_filename}")
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

# === STEERING WRAPPER ===
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

# === GENERATION FUNCTION ===
def generate_steered_response(prompt: str, layer_idx: int, coefficient: float) -> str:
    """Generate a steered response for given prompt, layer, and coefficient."""
    
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
        
        # Get complete response (input + generated)
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
        # Always reset
        target_wrapper.reset()

# === PROCESS ALL COMBINATIONS ===
print("\n--- Generating steered responses... ---")

for layer_idx in test_layers:
    for coefficient in STEERING_COEFFICIENTS:
        print(f"\nProcessing Layer {layer_idx}, Coefficient {coefficient}...")
        
        # Output filename
        coeff_str = f"pos{abs(coefficient)}" if coefficient > 0 else f"neg{abs(coefficient)}"
        output_filename = os.path.join(output_dir, f"steered_layer_{layer_idx}_coeff_{coeff_str}.jsonl")
        
        results = []
        
        for i, item in enumerate(tqdm(input_data, desc=f"L{layer_idx} C{coefficient}")):
            # Extract required fields
            input_prompt = item.get(prompt_field, "")
            biased_prompt = item.get("biased_prompt", "")
            generated_biased_answer = item.get("generated_biased_answer", "")
            classification = item.get("classification", "")
            
            if not input_prompt:
                print(f"Warning: No prompt found in item {i}")
                continue
            
            # Generate steered response
            steered_complete = generate_steered_response(input_prompt, layer_idx, coefficient)
            
            # Create biased complete prompt
            biased_complete = biased_prompt + generated_biased_answer
            
            # Save result
            result = {
                "item_index": i,
                "steered_prompt_complete": steered_complete,
                "biased_prompt_complete": biased_complete,
                "classification": classification,
                "layer": layer_idx,
                "coefficient": coefficient,
                "model": model_id
            }
            
            results.append(result)
        
        # Save JSONL file for this combination
        with open(output_filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(results)} results to {output_filename}")

print(f"\n=== GENERATION COMPLETE ===")
print(f"Output files saved to {output_dir}/")
print(f"Generated {len(test_layers) * len(STEERING_COEFFICIENTS)} files total")