"""
eval_steering_easysteer.py

Dual-mode steering evaluation script using EasySteer + vLLM.

Modes:
- LINEAR: Pre-computed mean-diff vectors (batch processing, fast)
- MLP: Per-prompt gradient-optimized vectors via MLP probes (sequential, per-prompt)

This script:
1. Auto-discovers input JSONL files based on model name and dataset type
2. Loads model with EasySteer steering support (vLLM backend)
3. Runs steering sweep based on selected mode
4. Saves steered prompts with metrics to JSONL
5. Generates summary with performance statistics

Usage:
    # Linear mode (pre-computed vectors)
    python eval_steering_easysteer.py \\
        --mode linear \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \\
        --dataset-type "annotated" \\
        --layers 8 13 15 \\
        --coefficients 0.75 -0.75 1 -1

    # MLP mode (gradient-based per-prompt)
    python eval_steering_easysteer.py \\
        --mode mlp \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \\
        --dataset-type "annotated" \\
        --layers 8 13 15 \\
        --directions offensive defensive \\
        --target-values 5 10 15 \\
        --shard 3 0  # Optional: (num_shards, shard_id)
"""

import argparse
import json
import time
import gc
import os
import re
import math
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import torch

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.model import load_model_easysteer
from src.config import TODAY

# Import conversion utility
from utilities.convert_pkl_to_gguf import convert_pkl_to_gguf


# =============================================================================
# FILE DISCOVERY UTILITIES
# =============================================================================

def get_model_short_name(model_id: str) -> str:
    """Extract a short name from a model ID for file matching."""
    short_name = model_id.split("/")[-1]
    short_name = short_name.replace("-Instruct", "").replace("-instruct", "")
    return short_name


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYY-MM-DD date from filename."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None


def find_input_file(model_id: str, dataset_type: str, base_dir: Path) -> Path:
    """Search entire repo for matching JSONL files."""
    model_short = get_model_short_name(model_id)
    model_lower = model_short.lower().replace("-", "").replace("_", "")
    
    print(f"Searching for input file...")
    print(f"  Model: {model_id} -> short: {model_short}")
    print(f"  Dataset type: {dataset_type}")
    
    all_jsonl = list(base_dir.rglob("*.jsonl"))
    print(f"  Found {len(all_jsonl)} total JSONL files")
    
    model_matches = [f for f in all_jsonl 
                     if model_lower in str(f).lower().replace("-", "").replace("_", "")]
    print(f"  After model filter: {len(model_matches)} files")
    
    dataset_matches = [f for f in model_matches if dataset_type.lower() in f.stem.lower()]
    print(f"  After dataset filter: {len(dataset_matches)} files")
    
    if not dataset_matches:
        raise FileNotFoundError(
            f"No JSONL files found matching model='{model_short}' and dataset_type='{dataset_type}'"
        )
    
    def get_date_key(path: Path) -> str:
        date = extract_date_from_filename(path.stem)
        return date if date else "0000-00-00"
    
    dataset_matches.sort(key=get_date_key, reverse=True)
    selected = dataset_matches[0]
    print(f"  Selected: {selected}")
    return selected
    
    # hi, unlucky person looking at this project


def find_vectors_file(input_dir: Path, model_id: str) -> Path:
    """Search for vectors PKL matching model name in input_dir or parent directories."""
    model_short = get_model_short_name(model_id)
    model_lower = model_short.lower().replace("-", "").replace("_", "")
    
    print(f"Searching for vectors file...")
    print(f"  Model: {model_short}")
    
    search_dirs = [input_dir]
    current = input_dir
    for _ in range(4):
        current = current.parent
        if current.name == "":
            break
        search_dirs.append(current)
    
    all_pkl_files = []
    for search_dir in search_dirs:
        all_pkl_files.extend(list(search_dir.rglob("*vectors*.pkl")))
    
    print(f"  Found {len(all_pkl_files)} total vectors PKL files")
    
    # Filter by model name
    model_matches = [f for f in all_pkl_files 
                     if model_lower in str(f).lower().replace("-", "").replace("_", "")]
    
    print(f"  After model filter: {len(model_matches)} files")
    
    if not model_matches:
        raise FileNotFoundError(
            f"No vectors PKL files found matching model '{model_short}' in or above: {input_dir}\n"
            f"Expected pattern: *vectors*{model_short}*.pkl"
        )
    
    def get_date_key(path: Path) -> str:
        date = extract_date_from_filename(path.stem)
        return date if date else "0000-00-00"
    
    model_matches.sort(key=get_date_key, reverse=True)
    selected = model_matches[0]
    print(f"  Selected: {selected}")
    return selected


def find_off_policy_vectors_file(model_id: str, base_dir: Path) -> Path:
    """Search for off-policy vectors PKL file matching model name."""
    model_short = get_model_short_name(model_id)
    model_lower = model_short.lower().replace("-", "").replace("_", "")
    
    print(f"Searching for off-policy vectors file...")
    print(f"  Model: {model_short}")
    
    # Search for files with 'vectors', 'off_policy' (or 'off-policy'), and model name
    all_pkl = list(base_dir.rglob("*vectors*.pkl"))
    
    # Filter for off_policy in path
    off_policy_matches = []
    for f in all_pkl:
        path_str = str(f).lower().replace("-", "").replace("_", "")
        if "offpolicy" in path_str and model_lower in path_str:
            off_policy_matches.append(f)
    
    print(f"  Found {len(off_policy_matches)} off-policy vectors files matching model")
    
    if not off_policy_matches:
        raise FileNotFoundError(
            f"No off-policy vectors PKL found for model '{model_short}'\n"
            f"Expected pattern: *vectors*off_policy*{model_short}*.pkl"
        )
    
    def get_date_key(path: Path) -> str:
        date = extract_date_from_filename(path.stem)
        return date if date else "0000-00-00"
    
    off_policy_matches.sort(key=get_date_key, reverse=True)
    selected = off_policy_matches[0]
    print(f"  Selected: {selected}")
    return selected


def find_activations_file(model_id: str, base_dir: Path) -> Path:
    """Search for activations PKL file matching model name."""
    model_short = get_model_short_name(model_id)
    model_lower = model_short.lower().replace("-", "").replace("_", "")
    
    print(f"Searching for activations file...")
    
    all_pkl = list(base_dir.rglob("*activations*.pkl"))
    model_matches = [f for f in all_pkl 
                     if model_lower in str(f).lower().replace("-", "").replace("_", "")]
    
    print(f"  Found {len(model_matches)} activations files matching model")
    
    if not model_matches:
        raise FileNotFoundError(f"No activations PKL found for model '{model_short}'")
    
    def get_date_key(path: Path) -> str:
        date = extract_date_from_filename(path.stem)
        return date if date else "0000-00-00"
    
    model_matches.sort(key=get_date_key, reverse=True)
    selected = model_matches[0]
    print(f"  Selected: {selected}")
    return selected


def find_probes_dir(model_id: str, base_dir: Path) -> Path:
    """Search for MLP probes directory matching model name."""
    model_short = get_model_short_name(model_id)
    model_lower = model_short.lower().replace("-", "").replace("_", "")
    
    print(f"Searching for MLP probes directory...")
    
    # Look for directories with 'probe' and 'mlp' containing layer_*.pth files
    probe_dirs = []
    for d in base_dir.rglob("*probe*"):
        if d.is_dir():
            mlp_subdir = d / "mlp"
            if mlp_subdir.exists() and list(mlp_subdir.glob("layer_*.pth")):
                probe_dirs.append(mlp_subdir)
    
    print(f"  Found {len(probe_dirs)} probe directories with MLP files")
    
    if not probe_dirs:
        raise FileNotFoundError(f"No MLP probe directories found in {base_dir}")
    
    # Prefer those matching model name
    model_matches = [d for d in probe_dirs 
                     if model_lower in str(d).lower().replace("-", "").replace("_", "")]
    
    if model_matches:
        selected = model_matches[0]
    else:
        selected = probe_dirs[0]  # Fall back to first found
    
    print(f"  Selected: {selected}")
    return selected


# =============================================================================
# CLI ARGUMENTS
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dual-mode steering evaluation using EasySteer + vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Linear mode
    python eval_steering_easysteer.py --mode linear \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \\
        --dataset-type "annotated" \\
        --coefficients 0.75 -0.75 1 -1

    # MLP mode with sharding
    python eval_steering_easysteer.py --mode mlp \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \\
        --dataset-type "annotated" \\
        --directions offensive defensive \\
        --target-values 5 10 15 \\
        --shard 3 0
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["linear", "mlp", "off-policy", "random"],
        required=True,
        help="Steering mode: 'linear' (pre-computed vectors), 'mlp' (gradient-based per-prompt), or 'random' (sanity check)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model ID"
    )
    
    # File discovery (shared)
    parser.add_argument("--dataset-type", type=str, required=True,
                        help="Dataset type handle (e.g., 'annotated')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as input file)")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 13, 15, 18, 23, 28],
                        help="Layers to test")
    
    # Linear mode specific
    parser.add_argument("--input-vectors", type=str, default=None,
                        help="[linear] Path to steering vectors PKL")
    parser.add_argument("--coefficients", type=float, nargs="+", 
                        default=[0.6, -0.6, 0.75, -0.75, 1, -1],
                        help="[linear/random] Coefficients to test (for random: recommended 0.6, 1, 2)")
    
    # MLP mode specific
    parser.add_argument("--input-activations", type=str, default=None,
                        help="[mlp] Path to activations PKL")
    parser.add_argument("--probes-dir", type=str, default=None,
                        help="[mlp] Path to MLP probes directory")
    parser.add_argument("--directions", type=str, nargs="+", 
                        default=["offensive", "defensive"],
                        help="[mlp] Directions to test")
    parser.add_argument("--target-values", type=float, nargs="+",
                        default=[5, 10, 15, 20, 30, 40],
                        help="[mlp] Target values for gradient optimization")
    parser.add_argument("--shard", type=int, nargs=2, default=None,
                        metavar=("NUM_SHARDS", "SHARD_ID"),
                        help="[mlp] Optional sharding: (num_shards, shard_id)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="[mlp] Learning rate for gradient optimization (default: 0.05)")
    parser.add_argument("--opt-steps", type=int, default=50,
                        help="[mlp] Number of optimization steps (default: 50)")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[mlp] Batch size for generation (default: 32)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    
    return parser.parse_args()


# =============================================================================
# LINEAR MODE
# =============================================================================

def run_linear_mode(args, llm, val_data, input_prompts, output_dir, base_dir):
    """Run linear mode: pre-computed vectors with coefficient sweep."""
    from vllm import SamplingParams
    from vllm.steer_vectors.request import SteerVectorRequest
    
    # Find vectors file - different logic for off-policy mode
    input_file = find_input_file(args.model, args.dataset_type, base_dir)
    if args.input_vectors:
        vectors_file = Path(args.input_vectors)
    elif args.mode == "off-policy":
        vectors_file = find_off_policy_vectors_file(args.model, base_dir)
    else:
        vectors_file = find_vectors_file(input_file.parent, args.model)

    
    # Convert PKL to GGUF
    print(f"\nConverting steering vectors to GGUF format...")
    gguf_dir = output_dir / "gguf_vectors"
    gguf_paths = convert_pkl_to_gguf(
        pkl_path=str(vectors_file),
        output_dir=str(gguf_dir),
        layers=args.layers
    )
    
    layers_to_test = [l for l in args.layers if l in gguf_paths]
    print(f"Will test {len(layers_to_test)} layers: {layers_to_test}")
    
    # Steering sweep
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0, repetition_penalty=1.1)
    evaluation_results = {}
    config_count = 0
    total_configs = len(layers_to_test) * len(args.coefficients)
    
    for layer_idx in layers_to_test:
        for coeff in args.coefficients:
            config_count += 1
            print(f"\n[{config_count}/{total_configs}] Layer {layer_idx}, Coefficient {coeff:+.2f}")
            
            steer_request = SteerVectorRequest(
                steer_vector_name=f"layer_{layer_idx}_coeff_{coeff}",
                steer_vector_int_id=config_count,
                steer_vector_local_path=gguf_paths[layer_idx],
                scale=coeff,
                target_layers=[layer_idx],
                prefill_trigger_tokens=[],
                generate_trigger_tokens=[-1],
            )
            
            outputs = llm.generate(input_prompts, steer_vector_request=steer_request,
                                   sampling_params=sampling_params)
            
            steered_responses = [output.outputs[0].text.strip() for output in outputs]
            steered_prompts = [p + r for p, r in zip(input_prompts, steered_responses)]
            
            evaluation_results[(layer_idx, coeff)] = {
                'steered_prompts': steered_prompts,
                'total_prompts': len(val_data)
            }
            print(f"  Generated {len(steered_responses)} steered responses")
            gc.collect()
    
    # Build output records
    output_data = []
    for (layer_idx, coeff), results in evaluation_results.items():
        for i, orig_item in enumerate(val_data):
            record = {
                'hinted_id': orig_item.get('hinted_id', i),
                'steering_layer': layer_idx,
                'steering_coefficient': coeff,
                'steered_prompt': results['steered_prompts'][i],
                'hint_template': orig_item.get('hint_template'),
                'ground_truth_letter': orig_item.get('ground_truth_letter'),
                'hint_letter': orig_item.get('hint_letter'),
                'biased_answer_letter': orig_item.get('biased_answer_letter'),
                'original_faithfulness_classification': orig_item.get('faithfulness_classification'),
                'split': 'val',
                'date': TODAY,
                'model': args.model,
                'steering_mode': 'linear',
                'backend': 'easysteer_vllm'
            }
            output_data.append(record)
    
    return output_data, {
        'vectors_file': str(vectors_file),
        'layers_tested': layers_to_test,
        'coefficients_tested': args.coefficients
    }


# =============================================================================
# RANDOM MODE (SANITY CHECK)
# =============================================================================

def run_random_mode(args, llm, val_data, input_prompts, output_dir, base_dir):
    """Run random mode: unit-normalized random vectors as sanity check.
    
    Generates one unit-normalized random vector per layer using seed=42.
    Vectors are normalized to L2 norm = 1, so coefficients directly control magnitude.
    This provides a direct comparison with linear steering methods.
    """
    from vllm import SamplingParams
    from vllm.steer_vectors.request import SteerVectorRequest
    import gguf
    import numpy as np
    
    # Get hidden dimension from model config
    hidden_dim = llm.llm_engine.model_config.hf_config.hidden_size
    print(f"Model hidden dimension: {hidden_dim}")
    
    # Generate one unit-normalized random vector per layer (seed=42)
    rng = np.random.default_rng(seed=42)
    random_vectors = {}
    for layer_idx in args.layers:
        v = rng.standard_normal(hidden_dim).astype(np.float32)
        v = v / np.linalg.norm(v)  # Unit normalize (L2 norm = 1)
        random_vectors[layer_idx] = v
        print(f"  Layer {layer_idx}: generated unit-normalized random vector (L2 norm: {np.linalg.norm(v):.4f})")
    
    # Save to GGUF (one file per layer)
    gguf_dir = output_dir / "gguf_random_vectors"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_paths = {}
    
    for layer_idx, vec in random_vectors.items():
        gguf_path = gguf_dir / f"random_layer_{layer_idx}.gguf"
        writer = gguf.GGUFWriter(str(gguf_path), "controlvector")
        writer.add_string("controlvector.model_hint", "llama")
        writer.add_string("controlvector.method", "random")
        writer.add_uint32("controlvector.layer_count", 1)
        writer.add_tensor(f"direction.{layer_idx}", vec)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        gguf_paths[layer_idx] = str(gguf_path)
        print(f"  Saved: {gguf_path}")
    
    layers_to_test = list(gguf_paths.keys())
    print(f"Will test {len(layers_to_test)} layers: {layers_to_test}")
    
    # Steering sweep (identical to linear mode)
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0, repetition_penalty=1.1)
    evaluation_results = {}
    config_count = 0
    total_configs = len(layers_to_test) * len(args.coefficients)
    
    for layer_idx in layers_to_test:
        for coeff in args.coefficients:
            config_count += 1
            print(f"\n[{config_count}/{total_configs}] Layer {layer_idx}, Coefficient {coeff:+.2f}")
            
            steer_request = SteerVectorRequest(
                steer_vector_name=f"random_layer_{layer_idx}_coeff_{coeff}",
                steer_vector_int_id=config_count,
                steer_vector_local_path=gguf_paths[layer_idx],
                scale=coeff,
                target_layers=[layer_idx],
                prefill_trigger_tokens=[],
                generate_trigger_tokens=[-1],
            )
            
            outputs = llm.generate(input_prompts, steer_vector_request=steer_request,
                                   sampling_params=sampling_params)
            
            steered_responses = [output.outputs[0].text.strip() for output in outputs]
            steered_prompts = [p + r for p, r in zip(input_prompts, steered_responses)]
            
            evaluation_results[(layer_idx, coeff)] = {
                'steered_prompts': steered_prompts,
                'total_prompts': len(val_data)
            }
            print(f"  Generated {len(steered_responses)} steered responses")
            gc.collect()
    
    # Build output records
    output_data = []
    for (layer_idx, coeff), results in evaluation_results.items():
        for i, orig_item in enumerate(val_data):
            record = {
                'hinted_id': orig_item.get('hinted_id', i),
                'steering_layer': layer_idx,
                'steering_coefficient': coeff,
                'steered_prompt': results['steered_prompts'][i],
                'hint_template': orig_item.get('hint_template'),
                'ground_truth_letter': orig_item.get('ground_truth_letter'),
                'hint_letter': orig_item.get('hint_letter'),
                'biased_answer_letter': orig_item.get('biased_answer_letter'),
                'original_faithfulness_classification': orig_item.get('faithfulness_classification'),
                'split': 'val',
                'date': TODAY,
                'model': args.model,
                'steering_mode': 'random',
                'backend': 'easysteer_vllm'
            }
            output_data.append(record)
    
    return output_data, {
        'random_seed': 42,
        'layers_tested': layers_to_test,
        'coefficients_tested': args.coefficients,
        'vector_norms': {l: float(np.linalg.norm(v)) for l, v in random_vectors.items()}
    }


# =============================================================================
# MLP MODE
# =============================================================================

def run_mlp_mode(args, llm, val_data, val_indices, input_prompts, output_dir, base_dir):
    """Run MLP mode: per-prompt gradient-optimized steering via batched processing (HF backend)."""
    from src.probe import MLPProbe
    from src.model import load_model
    from src.per_prompt_steering import apply_per_prompt_steering_to_model
    from tqdm import tqdm
    import torch
    
    # NOTE: 'llm' argument is ignored here as we load a fresh HF model for steering
    # This is because the main() function loads vLLM by default which we don't use for this mode anymore
    
    # Find activations and probes
    if args.input_activations:
        activations_file = Path(args.input_activations)
    else:
        activations_file = find_activations_file(args.model, base_dir)
    
    if args.probes_dir:
        probes_dir = Path(args.probes_dir)
    else:
        probes_dir = find_probes_dir(args.model, base_dir)
    
    # Load activations
    print(f"\nLoading activations from {activations_file}...")
    with open(activations_file, 'rb') as f:
        activation_dataset = pickle.load(f)
    activations = activation_dataset['data']
    print(f"Loaded activations for {len(activations)} prompts")
    
    # Load MLP probes
    print(f"\nLoading MLP probes from {probes_dir}...")
    mlp_probes = {}
    for layer_idx in args.layers:
        probe_path = probes_dir / f"layer_{layer_idx}.pth"
        if probe_path.exists():
            checkpoint = torch.load(probe_path, map_location='cpu', weights_only=False)
            mlp = MLPProbe(
                input_dim=checkpoint['config']['input_dim'],
                hidden_dim=checkpoint['config']['hidden_dim'],
                num_hidden_layers=checkpoint['config'].get('num_hidden_layers', 2)
            )
            mlp.load_state_dict(checkpoint['model_state_dict'])
            mlp.eval()
            mlp_probes[layer_idx] = mlp
            print(f"  Loaded layer {layer_idx}")
        else:
            print(f"  Warning: Probe for layer {layer_idx} not found")
    
    layers_to_test = [l for l in args.layers if l in mlp_probes]
    print(f"Will test {len(layers_to_test)} layers: {layers_to_test}")
    
    # =========================================================================
    # PHASE 1: PRE-COMPUTE STEERING VECTORS (GPU-BATCHED)
    # =========================================================================
    print("\n=== PHASE 1: Pre-computing Steering Vectors (GPU-Batched) ===")
    
    # Import GPU-batched function
    from src.gradient_steering import compute_steering_vectors_gpu_batched
    
    # Structure: precomputed_vectors[(layer, direction, target)][prompt_index_in_input_list] = vector
    precomputed_vectors = {}
    
    total_configs = len(layers_to_test) * len(args.directions) * len(args.target_values)
    config_count = 0
    
    for layer_idx in layers_to_test:
        # Load probe for this layer to GPU
        mlp = mlp_probes[layer_idx].float().cuda()
        print(f"\n--- Layer {layer_idx}: Probe loaded to GPU ---")
        
        # Extract all activations for this layer (stay on CPU for now)
        layer_activations = []
        for prompt_idx in val_indices:
            if prompt_idx not in activations:
                layer_activations.append(None)
                continue
                
            prompt_activations = activations[prompt_idx]['layers'].get(layer_idx, {})
            
            # Get F_body or U_body activation
            f_body = prompt_activations.get('F_body')
            u_body = prompt_activations.get('U_body')
            
            if f_body is not None and f_body.numel() > 0:
                activation_tensor = f_body.mean(dim=0).float()  # Average over tokens
            elif u_body is not None and u_body.numel() > 0:
                activation_tensor = u_body.mean(dim=0).float()
            else:
                layer_activations.append(None)
                continue
            
            layer_activations.append(activation_tensor)
        
        print(f"  Extracted {sum(1 for a in layer_activations if a is not None)} valid activations")
        
        # Process all direction × target configs for this layer
        for direction in args.directions:
            for target_value in args.target_values:
                config_count += 1
                config_key = (layer_idx, direction, target_value)
                print(f"[{config_count}/{total_configs}] Computing vectors: Layer {layer_idx} | {direction} | Target {target_value}")
                
                # GPU-batched computation
                import time
                start_time = time.time()
                
                vectors_list = compute_steering_vectors_gpu_batched(
                    activations_list=layer_activations,
                    mlp_model=mlp,
                    target_value=target_value,
                    direction=direction,
                    learning_rate=args.lr,
                    num_steps=args.opt_steps,
                    device="cuda"
                )
                
                elapsed = time.time() - start_time
                valid_count = sum(1 for v in vectors_list if v is not None)
                print(f"  Computed {valid_count} vectors in {elapsed:.2f}s ({elapsed/max(valid_count, 1)*1000:.1f}ms/prompt)")
                
                precomputed_vectors[config_key] = vectors_list
        
        # Cleanup: Remove probe from GPU after all configs for this layer
        mlp.cpu()
        del mlp
        torch.cuda.empty_cache()
        gc.collect()
        print(f"--- Layer {layer_idx}: Probe unloaded from GPU ---")

    # =========================================================================
    # PHASE 2: BATCHED GENERATION (HF BACKEND)
    # =========================================================================
    print("\n=== PHASE 2: Batched Generation (HF Backend) ===")
    
    # Load HF Model with Flash Attention 2
    # Note: We unload the vLLM model first if possible, but here we just load a new one
    # Ideally main() should not load vLLM if mode is MLP, but we handle that by just loading a second model
    # (Assuming enough VRAM or vLLM was cleared. The script structure makes this tricky, 
    #  but we'll assume the user has resources or we should modify main to not load vLLM)
    
    print("Loading Hugging Face model with SDPA...")
    model, tokenizer = load_model(args.model, attn_implementation="sdpa")
    
    # Set tokenizer limits
    tokenizer.model_max_length = 1024  # Enforce max input tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Required for decoder-only models
        
    # Apply Per-Prompt Steering Wrapper
    print("Applying per-prompt steering wrappers...")
    wrappers = apply_per_prompt_steering_to_model(model, layers_to_test)
    
    evaluation_results = {}
    batch_size = args.batch_size
    
    # Iterate through configs again for generation
    config_count = 0
    
    for layer_idx in layers_to_test:
        wrapper = wrappers[layer_idx]
        
        for direction in args.directions:
            for target_value in args.target_values:
                config_count += 1
                config_key = (layer_idx, direction, target_value)
                print(f"\n[{config_count}/{total_configs}] Generating: Layer {layer_idx} | {direction} | Target {target_value}")
                
                vectors = precomputed_vectors[config_key]
                steered_responses = []
                
                # Batched generation
                for i in tqdm(range(0, len(input_prompts), batch_size), desc="Generating batches"):
                    batch_prompts = input_prompts[i:i + batch_size]
                    batch_vectors = vectors[i:i + batch_size]
                    
                    # Tokenize
                    inputs = tokenizer(
                        batch_prompts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=1024
                    ).to(model.device)
                    
                    # Set steering vectors for this batch
                    # Note: batch_vectors contains Tensors or None. Wrapper handles None.
                    wrapper.set_steering(batch_vectors)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            repetition_penalty=1.1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    # Reset wrapper immediately
                    wrapper.reset()
                    
                    # Decode
                    input_len = inputs['input_ids'].shape[1]
                    batch_decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
                    steered_responses.extend([r.strip() for r in batch_decoded])
                    
                    del inputs, outputs
                    torch.cuda.empty_cache()
                
                # Store results
                full_steered_prompts = [p + r for p, r in zip(input_prompts, steered_responses)]
                evaluation_results[config_key] = {
                    'steered_prompts': full_steered_prompts,
                    'total_prompts': len(val_data)
                }
                
    # Build output records
    output_data = []
    for (layer_idx, direction, target_value), results in evaluation_results.items():
        for i, orig_item in enumerate(val_data):
            record = {
                'hinted_id': orig_item.get('hinted_id', i),
                'steering_layer': layer_idx,
                'steering_direction': direction,
                'steering_target_value': target_value,
                'steered_prompt': results['steered_prompts'][i],
                'hint_template': orig_item.get('hint_template'),
                'ground_truth_letter': orig_item.get('ground_truth_letter'),
                'hint_letter': orig_item.get('hint_letter'),
                'biased_answer_letter': orig_item.get('biased_answer_letter'),
                'original_faithfulness_classification': orig_item.get('faithfulness_classification'),
                'split': 'val',
                'date': TODAY,
                'model': args.model,
                'steering_mode': 'mlp',
                'backend': 'hf_custom_hooks'
            }
            output_data.append(record)
            
    return output_data, {
        'activations_file': str(activations_file),
        'probes_dir': str(probes_dir),
        'layers_tested': layers_to_test,
        'directions_tested': args.directions,
        'target_values_tested': args.target_values
    }


def _save_single_vector_gguf(vector_np, layer_idx: int, output_path: str):
    """Save a single steering vector to GGUF format."""
    import gguf
    
    writer = gguf.GGUFWriter(output_path, "steering_vector")
    writer.add_tensor(f"direction.{layer_idx}", vector_np)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    args = parse_args()
    start_time = time.time()
    base_dir = Path(__file__).parent.resolve()
    
    print("=" * 60)
    print(f"STEERING EVALUATION ({args.mode.upper()} MODE)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Mode: {args.mode}")
    print(f"Layers: {args.layers}")
    
    if args.mode == "linear":
        print(f"Coefficients: {args.coefficients}")
    else:
        print(f"Directions: {args.directions}")
        print(f"Target Values: {args.target_values}")
        if args.shard:
            print(f"Shard: {args.shard[1]} of {args.shard[0]}")
    
    # =========================================================================
    # STEP 1: Discover Input Files and Prepare Data
    # =========================================================================
    print("\n=== STEP 1: Discover Input Files ===")
    input_file = find_input_file(args.model, args.dataset_type, base_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 2: Load Model
    # =========================================================================
    print("\n=== STEP 2: Load Model ===")
    if args.mode == "mlp":
        print("Skipping vLLM load for MLP mode (will load HF model later)")
        llm = None
    else:
        print(f"Loading model with EasySteer: {args.model}")
        llm = load_model_easysteer(
            model_id=args.model,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len
        )
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # =========================================================================
    # STEP 3: Load and Filter Data
    # =========================================================================
    print("\n=== STEP 3: Load and Filter Data ===")
    annotated_data = load_jsonl(str(input_file))
    print(f"Loaded {len(annotated_data)} examples")
    
    # Filter for validation split and incorrect hints
    val_data = []
    val_indices = []
    for idx, item in enumerate(annotated_data):
        if item.get('split') == 'val':
            gt = item.get('ground_truth_letter')
            hint = item.get('hint_letter')
            if gt and hint and gt != hint:
                val_data.append(item)
                val_indices.append(idx)
    
    print(f"After filtering: {len(val_data)} validation examples with biased hints")
    
    # Apply sharding (MLP mode only)
    if args.mode == "mlp" and args.shard:
        num_shards, shard_id = args.shard
        shard_size = math.ceil(len(val_data) / num_shards)
        start_idx = shard_id * shard_size
        end_idx = min(start_idx + shard_size, len(val_data))
        val_data = val_data[start_idx:end_idx]
        val_indices = val_indices[start_idx:end_idx]
        print(f"Shard {shard_id}/{num_shards}: Processing prompts {start_idx} to {end_idx} ({len(val_data)} prompts)")
    
    # Apply num_samples limit
    if args.num_samples is not None:
        val_data = val_data[:args.num_samples]
        val_indices = val_indices[:args.num_samples]
        print(f"Limited to first {args.num_samples} samples")
    
    # Extract prompts
    input_prompts = [item.get('biased_input_prompt') for item in val_data]
    if None in input_prompts:
        raise ValueError("Missing 'biased_input_prompt' in some items")
    print(f"Extracted {len(input_prompts)} prompts")
    
    # =========================================================================
    # STEP 4: Run Steering (Mode-Specific)
    # =========================================================================
    print(f"\n=== STEP 4: Run {args.mode.upper()} Mode Steering ===")
    
    if args.mode in ["linear", "off-policy"]:
        output_data, mode_metadata = run_linear_mode(
            args, llm, val_data, input_prompts, output_dir, base_dir
        )
    elif args.mode == "random":
        output_data, mode_metadata = run_random_mode(
            args, llm, val_data, input_prompts, output_dir, base_dir
        )
    else:  # mlp mode
        output_data, mode_metadata = run_mlp_mode(
            args, llm, val_data, val_indices, input_prompts, output_dir, base_dir
        )
    
    # =========================================================================
    # STEP 5: Save Results
    # =========================================================================
    print("\n=== STEP 5: Save Results ===")
    
    model_short = get_model_short_name(args.model)
    
    # Build output filename - mode name in output (off-policy becomes off_policy)
    mode_for_filename = args.mode.replace("-", "_")
    
    if args.mode == "mlp" and args.shard:
        shard_suffix = f"_shard_{args.shard[1]}_of_{args.shard[0]}"
    else:
        shard_suffix = ""
    
    output_file = output_dir / f"steered_{mode_for_filename}_{model_short}_{TODAY}{shard_suffix}.jsonl"
    summary_file = output_dir / f"summary_{mode_for_filename}_{model_short}_{TODAY}{shard_suffix}.json"
    
    save_jsonl(output_data, str(output_file))
    print(f"Saved {len(output_data)} records to {output_file}")
    
    # Summary
    end_time = time.time()
    summary = {
        'metadata': {
            'date': TODAY,
            'model': args.model,
            'mode': args.mode,
            'backend': 'easysteer_vllm',
            'input_file': str(input_file),
            'output_file': str(output_file),
            'num_examples': len(val_data),
            'processing_time_seconds': end_time - start_time,
            'shard': args.shard if args.mode == "mlp" else None
        },
        'mode_config': mode_metadata
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_file}")
    
    # Final summary
    print(f"\n=== COMPLETE ===")
    print(f"Processing time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Results: {output_file}")
    
    # Cleanup
    # Cleanup
    if llm is not None:
        del llm
    gc.collect()


if __name__ == "__main__":
    main()
