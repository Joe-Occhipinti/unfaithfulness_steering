#!/usr/bin/env python3
"""
Diagnose model architecture to understand why steering isn't being applied
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

print("=== MODEL ARCHITECTURE DIAGNOSIS ===")

# Load model (same as your config)
print("Loading model...")
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

# Inspect model structure
print(f"\nModel type: {type(model)}")
print(f"Model config: {model.config.model_type}")

# Check if model.model.layers exists
if hasattr(model, 'model'):
    print(f"✓ model.model exists: {type(model.model)}")
    if hasattr(model.model, 'layers'):
        print(f"✓ model.model.layers exists: {len(model.model.layers)} layers")
        print(f"Layer 25 type: {type(model.model.layers[25])}")
    else:
        print("✗ model.model.layers does NOT exist")
        print(f"model.model attributes: {list(model.model.__dict__.keys())}")
else:
    print("✗ model.model does NOT exist")
    print(f"model attributes: {list(model.__dict__.keys())}")

# Test forward hook to see if layers are called during generation
print("\n=== FORWARD HOOK TEST ===")

class ForwardHook:
    def __init__(self, name):
        self.name = name
        self.call_count = 0

    def __call__(self, module, input, output):
        self.call_count += 1
        if self.call_count <= 2:  # Only print first 2 calls
            print(f"HOOK: {self.name} called (#{self.call_count})")

# Add hooks to a few layers
hooks = {}
test_layers = [24, 25, 26]

for layer_idx in test_layers:
    if hasattr(model.model, 'layers') and layer_idx < len(model.model.layers):
        hook = ForwardHook(f"layer_{layer_idx}")
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        hooks[layer_idx] = (hook, handle)
        print(f"Added hook to layer {layer_idx}")

# Test generation
print("\nTesting generation to see which layers get called...")
test_prompt = "The capital of France is"
inputs = tokenizer(test_prompt, return_tensors="pt")
# Move inputs to the same device as the model
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {response}")

# Check hook results
print("\n=== HOOK RESULTS ===")
for layer_idx, (hook, handle) in hooks.items():
    print(f"Layer {layer_idx}: called {hook.call_count} times")
    handle.remove()  # Clean up

# If no layers were called, investigate model structure more
if all(hook.call_count == 0 for hook, _ in hooks.values()):
    print("\n⚠️  NO LAYERS WERE CALLED DURING GENERATION!")
    print("Investigating model structure...")

    # Try to find where the actual layers are
    def explore_model(obj, path="", max_depth=3):
        if max_depth <= 0:
            return

        for name, child in obj.named_children():
            current_path = f"{path}.{name}" if path else name
            print(f"  {current_path}: {type(child)}")

            # Look for anything that looks like transformer layers
            if 'layer' in name.lower() and hasattr(child, '__len__'):
                print(f"    Found layer-like object with {len(child)} items")

            if max_depth > 1:
                explore_model(child, current_path, max_depth-1)

    print("\nModel structure:")
    explore_model(model, max_depth=2)

print("\n=== DIAGNOSIS COMPLETE ===")