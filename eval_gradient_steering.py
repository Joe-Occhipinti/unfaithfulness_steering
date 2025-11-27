"""
eval_gradient_steering.py

Gradient-based steering evaluation using MLP probes.

This script:
1. Loads annotated prompts (JSONL with validation split)
2. Loads corresponding activations (PKL file)
3. Loads trained MLP probes (one per layer)
4. For each prompt, computes per-prompt steering vector using MLP gradient
   with DYNAMIC TARGET LOGIC matching the jailbreak paper.
5. Applies steering at inference time using PerPromptSteeringWrapper
6. Saves steered outputs for evaluation

Methodology (Section 4.4 of Paper):
- Offensive: Push logit p toward max(t, p + t)
- Defensive: Push logit p toward min(-t, p - t)
- Target values t: {5, 10, 15, 20, 30, 40}
"""

# ... (Imports same as before)
import json
import time
import torch
import gc
import pickle
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.model import load_model
from src.probe import MLPProbe
from src.config import TODAY
from src.per_prompt_steering import apply_per_prompt_steering_to_model
from src.gradient_steering import compute_steering_vector_gradient

# =============================================================================
# COLAB SETUP (Run this block first in Colab)
# =============================================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Not running in Colab - skipping Drive mount")

if IN_COLAB:
    import os
    if not os.path.exists('/content/unfaithfulness_steering'):
        !git clone https://github.com/Joe-Occhipinti/unfaithfulness_steering.git
        os.chdir('/content/unfaithfulness_steering')
    else:
        os.chdir('/content/unfaithfulness_steering')
        !git pull origin main

    !git config --global user.email "occhidipinti00@gmail.com"
    !git config --global user.name "Joe-Occhipinti"
    
    try:
        from google.colab import userdata
        os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
    except:
        pass
    
    !pip install torch==2.4.1 transformers==4.44.2 bitsandbytes==0.43.3 accelerate==0.34.2

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
INPUT_JSONL = "data/sprint_4_2025-10-15/annotated/touse_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"
INPUT_ACTIVATIONS = "data/sprint_4_2025-10-15/datasets/new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"
MLP_PROBES_DIR = "results/probe_training/mlp"

# Output files
if IN_COLAB:
    OUTPUT_FILE = f"/content/drive/MyDrive/steered_val_gradient_{TODAY}.jsonl"
    SUMMARY_FILE = f"/content/drive/MyDrive/summary_gradient_{TODAY}.json"
else:
    os.makedirs("results/steered_gradient", exist_ok=True)
    OUTPUT_FILE = f"results/steered_gradient/steered_val_gradient_{TODAY}.jsonl"
    SUMMARY_FILE = f"results/steered_gradient/summary_gradient_{TODAY}.json"

# Model configuration
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BATCH_SIZE = 5
MAX_NEW_TOKENS = 2048

# Steering configuration
LAYERS_TO_TEST = [8, 13, 15, 20, 23, 25, 28]

# Paper's configuration:
# Target values 't' (magnitude)
TARGET_VALUES = [5, 10, 15, 20, 30, 40]
# Directions to test
DIRECTIONS = ["offensive", "defensive"]

OPTIMIZATION_LR = 0.05
OPTIMIZATION_STEPS = 50

print(f"=== GRADIENT-BASED STEERING EVALUATION (PAPER METHODOLOGY) ===")
print(f"Model: {MODEL_ID}")
print(f"Layers: {LAYERS_TO_TEST}")
print(f"Target Values (t): {TARGET_VALUES}")
print(f"Directions: {DIRECTIONS}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_prompts_and_activations(jsonl_path, pkl_path):
    """Load prompts from JSONL and corresponding activations from PKL."""
    print(f"\n--- Loading Prompts and Activations ---")
    prompts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts from JSONL")
    
    with open(pkl_path, 'rb') as f:
        activation_dataset = pickle.load(f)
    activations = activation_dataset['data']
    print(f"Loaded activations for {len(activations)} prompts")
    return prompts, activations

def load_mlp_probes(probes_dir, layers):
    """Load trained MLP probes."""
    print(f"\n--- Loading MLP Probes ---")
    probes = {}
    for layer_idx in layers:
        probe_path = f"{probes_dir}/layer_{layer_idx}.pth"
        try:
            checkpoint = torch.load(probe_path, map_location='cpu')
            mlp = MLPProbe(
                input_dim=checkpoint['config']['input_dim'],
                hidden_dim=checkpoint['config']['hidden_dim']
            )
            mlp.load_state_dict(checkpoint['model_state_dict'])
            mlp.eval()
            probes[layer_idx] = mlp
            print(f"  Loaded MLP for layer {layer_idx}")
        except FileNotFoundError:
            print(f"  Warning: MLP probe for layer {layer_idx} not found")
    return probes

def generate_with_gradient_steering(
    model,
    tokenizer,
    prompts: List[str],
    activations: Dict,
    prompt_indices: List[int],
    mlp_probes: Dict,
    wrapped_layers: Dict,
    layer_idx: int,
    target_value: float,
    direction: str,
    batch_size: int = 5,
    max_new_tokens: int = 2048
) -> List[str]:
    """
    Generate steered outputs using per-prompt gradient-based steering vectors.
    """
    print(f"\nGenerating: Layer {layer_idx} | Direction {direction} | Target {target_value}")
    
    if layer_idx not in mlp_probes:
        return ["ERROR: No Probe"] * len(prompts)

    responses = []
    mlp = mlp_probes[layer_idx].to(model.device)
    
    # Pre-compute steering vectors
    print("Computing per-prompt steering vectors...")
    steering_vectors_map = {}
    
    for idx, prompt_idx in enumerate(prompt_indices):
        if prompt_idx not in activations:
            steering_vectors_map[idx] = None
            continue
        
        prompt_activations = activations[prompt_idx]['layers'][layer_idx]
        
        # Use F_body or U_body
        if 'F_body' in prompt_activations and prompt_activations['F_body'].numel() > 0:
            activation_tensor = prompt_activations['F_body']
        elif 'U_body' in prompt_activations and prompt_activations['U_body'].numel() > 0:
            activation_tensor = prompt_activations['U_body']
        else:
            steering_vectors_map[idx] = None
            continue
        
        mean_activation = activation_tensor.mean(dim=0).to(model.device)
        
        # Compute vector using updated logic
        steering_vec = compute_steering_vector_gradient(
            activation=mean_activation,
            mlp_model=mlp,
            target_value=target_value,
            direction=direction,
            learning_rate=OPTIMIZATION_LR,
            num_steps=OPTIMIZATION_STEPS
        )
        
        steering_vectors_map[idx] = steering_vec
    
    # Generate
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        batch_indices = list(range(i, min(i+batch_size, len(prompts))))
        
        batch_vectors = [steering_vectors_map.get(idx) for idx in batch_indices]
        
        if layer_idx in wrapped_layers:
            wrapped_layers[layer_idx].set_steering(batch_vectors, coefficient=1.0)
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        
        input_len = inputs['input_ids'].shape[1]
        batch_responses = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        responses.extend(batch_responses)
        
        if layer_idx in wrapped_layers:
            wrapped_layers[layer_idx].reset()
            
        gc.collect()
        torch.cuda.empty_cache()
    
    return responses

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    start_time = time.time()
    
    print("\n=== CELL 1: Load Data ===")
    prompts, activations = load_prompts_and_activations(INPUT_JSONL, INPUT_ACTIVATIONS)
    
    val_prompts = []
    val_indices = []
    for idx, prompt_dict in enumerate(prompts):
        if prompt_dict.get('split') == 'val':
            val_prompts.append(prompt_dict)
            val_indices.append(idx)
    print(f"Filtered {len(val_prompts)} validation prompts")
    
    print("\n=== CELL 2: Load Model and Probes ===")
    model, tokenizer = load_model(MODEL_ID)
    mlp_probes = load_mlp_probes(MLP_PROBES_DIR, LAYERS_TO_TEST)
    
    print("Applying steering wrappers...")
    wrapped_layers = apply_per_prompt_steering_to_model(model, LAYERS_TO_TEST)
    
    print("\n=== CELL 3: Gradient Steering Sweep ===")
    results = {}
    
    for layer_idx in LAYERS_TO_TEST:
        for direction in DIRECTIONS:
            for target_value in TARGET_VALUES:
                print(f"\n{'='*60}")
                print(f"Layer {layer_idx} | {direction.upper()} | Target {target_value}")
                print(f"{'='*60}")
                
                input_prompts = [p['biased_input_prompt'] for p in val_prompts]
                
                steered_responses = generate_with_gradient_steering(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=input_prompts,
                    activations=activations,
                    prompt_indices=val_indices,
                    mlp_probes=mlp_probes,
                    wrapped_layers=wrapped_layers,
                    layer_idx=layer_idx,
                    target_value=target_value,
                    direction=direction,
                    batch_size=BATCH_SIZE,
                    max_new_tokens=MAX_NEW_TOKENS
                )
                
                results[(layer_idx, direction, target_value)] = steered_responses
    
    print("\n=== CELL 4: Save Results ===")
    output_data = []
    for (layer_idx, direction, target_value), steered_responses in results.items():
        for i, val_prompt in enumerate(val_prompts):
            record = {
                'prompt_index': val_indices[i],
                'question_id': val_prompt.get('question_id', val_indices[i]),
                'biased_input_prompt': val_prompt.get('biased_input_prompt'),
                'hint_template': val_prompt.get('hint_template'),
                'steering_method': 'gradient_mlp',
                'steering_layer': layer_idx,
                'steering_direction': direction,
                'steering_target_value': target_value,
                'steered_response': steered_responses[i],
                'steered_prompt': val_prompt.get('biased_input_prompt') + steered_responses[i],
                'ground_truth_letter': val_prompt.get('ground_truth_letter'),
                'hint_letter': val_prompt.get('hint_letter'),
                'original_faithfulness': val_prompt.get('faithfulness_classification'),
                'split': 'val',
                'date': TODAY,
                'model': MODEL_ID
            }
            output_data.append(record)
    
    save_jsonl(output_data, OUTPUT_FILE)
    print(f"Saved {len(output_data)} records to {OUTPUT_FILE}")
    
    summary = {
        'metadata': {
            'date': TODAY,
            'model': MODEL_ID,
            'steering_method': 'gradient_mlp_paper_method',
            'layers_tested': LAYERS_TO_TEST,
            'target_values': TARGET_VALUES,
            'directions': DIRECTIONS,
            'processing_time_seconds': time.time() - start_time
        },
        'configurations': {
            f"layer_{l}_{d}_target_{t}": {'count': len(r)} 
            for (l, d, t), r in results.items()
        }
    }
    
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== COMPLETE ===")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
