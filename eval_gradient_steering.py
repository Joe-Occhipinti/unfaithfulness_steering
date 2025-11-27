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

import json
import time
import torch
import gc
import pickle
import os
import math
import shutil
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
    # Clone the repo to import in Colab its packages from GitHub
    import os
    if not os.path.exists('/content/unfaithfulness_steering'):
        !git clone https://github.com/Joe-Occhipinti/unfaithfulness_steering.git
        os.chdir('/content/unfaithfulness_steering')
    else:
        os.chdir('/content/unfaithfulness_steering')
        !git pull origin main

    # Authenticate in GitHub
    !git config --global user.email "occhidipinti00@gmail.com"
    !git config --global user.name "Joe-Occhipinti"
    
    # Put your GitHub token in Colab secrets
    from google.colab import userdata
    try:
        GITHUB_TOKEN = userdata.get('Colab')
        if GITHUB_TOKEN:
            # Build authenticated repo url
            repo_url = f"https://{GITHUB_TOKEN}@github.com/Joe-Occhipinti/unfaithfulness_steering.git"
            !git remote set-url origin {repo_url}
    except:
        print("GitHub token not found in secrets (optional for pulling public repo)")

    # Install required packages
    !pip install torch==2.4.1 transformers==4.44.2 bitsandbytes==0.43.3 accelerate==0.34.2

    # Set up OpenRouter API environment variables from Colab secrets
    try:
        os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
    except:
        pass

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
INPUT_JSONL = "data/sprint_4_2025-10-15/annotated/touse_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"
INPUT_ACTIVATIONS = "data/sprint_4_2025-10-15/datasets/new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"
MLP_PROBES_DIR = "results/probe_training/mlp"

# Parallelization Config (Sharding)
# CHANGE THIS FOR EACH NOTEBOOK: 0, 1, or 2
SHARD_ID = 0  
NUM_SHARDS = 3

# Output files
if IN_COLAB:
    OUTPUT_FILE = f"/content/drive/MyDrive/steered_val_gradient_{TODAY}_shard_{SHARD_ID}.jsonl"
    SUMMARY_FILE = f"/content/drive/MyDrive/summary_gradient_{TODAY}_shard_{SHARD_ID}.json"
else:
    os.makedirs("results/steered_gradient", exist_ok=True)
    OUTPUT_FILE = f"results/steered_gradient/steered_val_gradient_{TODAY}_shard_{SHARD_ID}.jsonl"
    SUMMARY_FILE = f"results/steered_gradient/summary_gradient_{TODAY}_shard_{SHARD_ID}.json"

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
print(f"Running Shard {SHARD_ID + 1}/{NUM_SHARDS}")

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
    
    # Check if probes exist locally; if not, try to copy from Drive
    if IN_COLAB and (not os.path.exists(probes_dir) or not os.listdir(probes_dir)):
        drive_probes_path = "/content/drive/MyDrive/results/probe_training/mlp"
        print(f"Probes not found in {probes_dir}. Attempting to copy from Drive: {drive_probes_path}")
        
        if os.path.exists(drive_probes_path):
            os.makedirs(probes_dir, exist_ok=True)
            # Copy only the needed layers
            for layer in layers:
                src = f"{drive_probes_path}/layer_{layer}.pth"
                dst = f"{probes_dir}/layer_{layer}.pth"
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"  Copied layer_{layer}.pth from Drive")
                else:
                    print(f"  Warning: layer_{layer}.pth not found in Drive")
        else:
            print(f"  Drive path {drive_probes_path} does not exist")

    probes = {}
    missing_layers = []
    
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
            print(f"  ERROR: MLP probe for layer {layer_idx} not found at {probe_path}")
            missing_layers.append(layer_idx)
            
    if missing_layers:
        raise FileNotFoundError(
            f"Could not find MLP probes for layers: {missing_layers}. "
            "Please ensure trained probes are in 'results/probe_training/mlp' "
            "or uploaded to Google Drive at '/content/drive/MyDrive/results/probe_training/mlp'."
        )
            
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
    # Ensure MLP is on correct device and in float32 for stable optimization
    mlp = mlp_probes[layer_idx].to(model.device).float()
    
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
        
        # Average over tokens, move to device, and CAST TO FLOAT32
        mean_activation = activation_tensor.mean(dim=0).to(model.device).float()
        
        # Compute vector using updated logic
        steering_vec = compute_steering_vector_gradient(
            activation=mean_activation,
            mlp_model=mlp,
            target_value=target_value,
            direction=direction,
            learning_rate=OPTIMIZATION_LR,
            num_steps=OPTIMIZATION_STEPS
        )
        
        # Cast back to model's dtype (likely float16/bfloat16) for generation
        steering_vec = steering_vec.to(model.dtype)
        
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
    
    # Filter for validation split AND ground_truth != hint
    val_prompts = []
    val_indices = []
    for idx, prompt_dict in enumerate(prompts):
        if prompt_dict.get('split') == 'val':
            # Check if ground_truth_letter != hint_letter
            gt = prompt_dict.get('ground_truth_letter')
            hint = prompt_dict.get('hint_letter')
            
            # Only include if both exist and are different (biased hint)
            if gt and hint and gt != hint:
                val_prompts.append(prompt_dict)
                val_indices.append(idx)
    
    total_val = len(val_prompts)
    print(f"Total Validation Prompts (Biased Only): {total_val}")
    
    # --- SHARDING LOGIC ---
    shard_size = math.ceil(total_val / NUM_SHARDS)
    start_idx = SHARD_ID * shard_size
    end_idx = min(start_idx + shard_size, total_val)
    
    val_prompts = val_prompts[start_idx:end_idx]
    val_indices = val_indices[start_idx:end_idx]
    
    print(f"Processing Shard {SHARD_ID + 1}/{NUM_SHARDS}")
    print(f"Prompts {start_idx} to {end_idx} (Count: {len(val_prompts)})")
    # ----------------------
    
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
                
                # Verify input is biased_input_prompt
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
                'model': MODEL_ID,
                'shard_id': SHARD_ID
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
            'shard_id': SHARD_ID,
            'num_shards': NUM_SHARDS,
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
    
    # CELL 6: Verify Results Saved to Google Drive
    print("\n=== CELL 6: Verify Results Saved to Google Drive ===")
    
    # Check if files exist in Drive
    if os.path.exists(OUTPUT_FILE):
        print(f"Output file saved to Drive: {OUTPUT_FILE}")
        print(f"  Size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")
    else:
        print(f"Warning: Output file not found at {OUTPUT_FILE}")
    
    if os.path.exists(SUMMARY_FILE):
        print(f"Summary file saved to Drive: {SUMMARY_FILE}")
        print(f"  Size: {os.path.getsize(SUMMARY_FILE) / 1024:.2f} KB")
    else:
        print(f"Warning: Summary file not found at {SUMMARY_FILE}")
    
    print(f"\n=== EXPERIMENT COMPLETE ===")
    print(f"Results are saved in your Google Drive (MyDrive root):")
    print(f"  - {os.path.basename(OUTPUT_FILE)}")
    print(f"  - {os.path.basename(SUMMARY_FILE)}")
    print(f"\nNext steps:")
    print(f"  1. Access files from your Google Drive on any device")
    print(f"  2. Move files to your local repo and commit to GitHub when convenient")
    print(f"  3. Run eval_faithfulness.py to evaluate faithfulness of steered outputs")

if __name__ == "__main__":
    main()
