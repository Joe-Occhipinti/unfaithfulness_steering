"""
test_steering_vectors_layerwise.py

Tests steering vectors from all layers on input prompts and saves results
with layer-wise performance comparison.
"""

import torch
import json
import os
from tqdm import tqdm
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pickle
import time
from typing import Dict, List

# === CONFIGURATION ===
# --- Model and File Settings ---
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
prompts_filename = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\datasets\test_prompts.jsonl"  # Update this path
prompt_field = "prompt"  # Update this field name
steering_vectors_filename = "steering_vectors_faithful_vs_unfaithful_2025-09-04.pkl"  # Your steering vectors file
output_dir = "layerwise_steering_results"

# --- Steering Settings ---
STEERING_COEFFICIENTS = [5.0, -5.0]  # Test positive and negative steering
batch_size = 1  # Keep safe
max_new_tokens = 512  # Shorter for faster testing

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("=== LAYER-WISE STEERING VECTOR TESTING ===")
print(f"Model: {model_id}")
print(f"Input file: {prompts_filename}")
print(f"Steering vectors: {steering_vectors_filename}")
print(f"Output directory: {output_dir}")

# === LOAD MODEL AND TOKENIZER ===
print("\n--- Loading model and tokenizer... ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure quantization
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
    print("Set tokenizer pad_token to eos_token")

# === LOAD STEERING VECTORS ===
print("\n--- Loading steering vectors... ---")
with open(steering_vectors_filename, 'rb') as f:
    steering_data = pickle.load(f)

if 'steering_vectors' in steering_data:
    steering_vectors = steering_data['steering_vectors']
    metadata = steering_data.get('metadata', {})
    print(f"Loaded steering vectors for layers: {list(steering_vectors.keys())}")
    print(f"Vector dimension: {metadata.get('vector_dim', 'Unknown')}")
else:
    print("ERROR: Expected 'steering_vectors' key in pickle file")
    exit(1)

# Test all layers that have steering vectors (ensuring vector matches its source layer)
test_layers = sorted(steering_vectors.keys())

print(f"Testing layers: {test_layers}")
print(f"Testing coefficients: {STEERING_COEFFICIENTS}")
print("Note: Each steering vector will be applied to its original source layer")

# === IMPROVED WRAPPER CLASS ===
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
        self.coefficient = 0.0

# Wrap all test layers
wrapped_layers = {}
original_layers = {}

for layer_idx in test_layers:
    original_layer = model.model.layers[layer_idx]
    wrapped_layer = LayerSteeringWrapper(original_layer, layer_idx)
    model.model.layers[layer_idx] = wrapped_layer
    wrapped_layers[layer_idx] = wrapped_layer
    original_layers[layer_idx] = original_layer

print(f"Wrapped {len(wrapped_layers)} layers for steering")

# === LOAD PROMPTS ===
def load_prompts_from_file(filename: str, field_name: str) -> List[str]:
    prompts_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if field_name in data:
                    prompts_list.append(data[field_name])
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line")
    return prompts_list

print(f"\n--- Loading prompts from {prompts_filename}... ---")
prompts_to_test = load_prompts_from_file(prompts_filename, prompt_field)
print(f"Loaded {len(prompts_to_test)} prompts")

if len(prompts_to_test) == 0:
    print("ERROR: No prompts loaded! Check file path and field name.")
    exit(1)

# Limit prompts for testing (remove this for full run)
if len(prompts_to_test) > 10:
    print(f"Limiting to first 10 prompts for testing")
    prompts_to_test = prompts_to_test[:10]

# === STEERING TESTING FUNCTION ===
def test_layer_steering(layer_idx: int, coefficient: float, prompts: List[str]) -> List[str]:
    """Test steering for a specific layer and coefficient."""
    
    # Reset all wrappers
    for wrapper in wrapped_layers.values():
        wrapper.reset()
    
    # Set up steering for target layer
    target_wrapper = wrapped_layers[layer_idx]
    steering_vector = steering_vectors[layer_idx]
    target_wrapper.set_steering(steering_vector, coefficient)
    
    results = []
    
    for prompt in tqdm(prompts, desc=f"Layer {layer_idx}, Coeff {coefficient}", leave=False):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=1.0
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(answer.strip())
            
            # Cleanup
            del inputs, outputs, generated_tokens
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in generation: {e}")
            results.append("")
            continue
    
    # Reset after testing
    target_wrapper.reset()
    return results

# === BASELINE (NO STEERING) ===
print("\n--- Running baseline (no steering)... ---")
baseline_results = []

# Ensure all wrappers are reset
for wrapper in wrapped_layers.values():
    wrapper.reset()

for prompt in tqdm(prompts_to_test, desc="Baseline generation"):
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
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        baseline_results.append(answer.strip())
        
        del inputs, outputs, generated_tokens
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error in baseline generation: {e}")
        baseline_results.append("")

# === LAYER-WISE STEERING TESTING ===
print("\n--- Running layer-wise steering tests... ---")
all_results = {}

start_time = time.time()

for layer_idx in test_layers:
    print(f"\nTesting Layer {layer_idx}...")
    layer_results = {}
    
    for coefficient in STEERING_COEFFICIENTS:
        print(f"  Testing coefficient {coefficient}...")
        
        steering_results = test_layer_steering(layer_idx, coefficient, prompts_to_test)
        layer_results[coefficient] = steering_results
    
    all_results[layer_idx] = layer_results

end_time = time.time()
print(f"\n--- Testing complete! Total time: {end_time - start_time:.2f} seconds ---")

# === SAVE RESULTS ===
print("\n--- Saving results... ---")

# Save comprehensive results
comprehensive_results = {
    "metadata": {
        "model": model_id,
        "prompts_file": prompts_filename,
        "steering_vectors_file": steering_vectors_filename,
        "test_layers": test_layers,
        "coefficients": STEERING_COEFFICIENTS,
        "num_prompts": len(prompts_to_test),
        "max_new_tokens": max_new_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "prompts": prompts_to_test,
    "baseline_results": baseline_results,
    "steering_results": all_results
}

# Save main results file
results_file = os.path.join(output_dir, "comprehensive_steering_test_results.pkl")
with open(results_file, 'wb') as f:
    pickle.dump(comprehensive_results, f)
print(f"Saved comprehensive results to {results_file}")

# Save human-readable JSONL files for each configuration
for layer_idx in test_layers:
    for coefficient in STEERING_COEFFICIENTS:
        output_file = os.path.join(output_dir, f"steering_layer_{layer_idx}_coeff_{coefficient}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (prompt, baseline, steered) in enumerate(zip(
                prompts_to_test, 
                baseline_results, 
                all_results[layer_idx][coefficient]
            )):
                result = {
                    "prompt_index": i,
                    "prompt": prompt,
                    "baseline_response": baseline,
                    "steered_response": steered,
                    "layer": layer_idx,
                    "coefficient": coefficient,
                    "model": model_id
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Saved individual JSONL files to {output_dir}/")

# === GENERATE COMPARISON REPORT ===
def generate_comparison_report():
    """Generate a quick comparison report."""
    
    report_file = os.path.join(output_dir, "steering_comparison_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("LAYER-WISE STEERING VECTOR TESTING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {model_id}\n")
        f.write(f"Test prompts: {len(prompts_to_test)}\n")
        f.write(f"Layers tested: {test_layers}\n")
        f.write(f"Coefficients tested: {STEERING_COEFFICIENTS} (positive=faithful, negative=unfaithful)\n\n")
        
        f.write("SAMPLE COMPARISONS:\n")
        f.write("-" * 30 + "\n\n")
        
        # Show first prompt comparison across layers and coefficients
        if len(prompts_to_test) > 0:
            f.write(f"PROMPT: {prompts_to_test[0][:100]}...\n\n")
            
            f.write(f"BASELINE: {baseline_results[0][:200]}...\n\n")
            
            for layer_idx in test_layers[:3]:  # Show first 3 layers
                f.write(f"LAYER {layer_idx}:\n")
                for coeff in STEERING_COEFFICIENTS:
                    steered = all_results[layer_idx][coeff][0]
                    f.write(f"  Coeff {coeff}: {steered[:150]}...\n")
                f.write("\n")
        
        f.write("\nNEXT STEPS:\n")
        f.write("1. Review individual JSONL files for detailed comparisons\n")
        f.write("2. Analyze comprehensive_steering_test_results.pkl for quantitative metrics\n")
        f.write("3. Look for patterns in steering effectiveness across layers and coefficients\n")
    
    print(f"Generated comparison report: {report_file}")

generate_comparison_report()

# === PREVIEW RESULTS ===
print("\n--- PREVIEW RESULTS ---")
if len(prompts_to_test) > 0 and len(test_layers) > 0:
    prompt_idx = 0
    layer_idx = test_layers[0]
    coeff = STEERING_COEFFICIENTS[0]
    
    print(f"Sample comparison for first prompt:")
    print(f"PROMPT: {prompts_to_test[prompt_idx][:150]}...")
    print(f"BASELINE: {baseline_results[prompt_idx][:150]}...")
    print(f"STEERED (Layer {layer_idx}, Coeff {coeff}): {all_results[layer_idx][coeff][prompt_idx][:150]}...")

print(f"\n=== TESTING COMPLETE ===")
print(f"Check {output_dir}/ for all results and reports")