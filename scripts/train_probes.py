"""
train_layer_probes.py

Train linear and non-linear probes for all layers to detect faithfulness vs unfaithfulness
in LLM activations. Saves trained models and generates performance analysis.

Usage:
    python train_layer_probes.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    python train_layer_probes.py --model "Qwen/Qwen3-32B" --input-activations path/to/activations.pkl
"""

import sys
import os
# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import re
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.probe import (
    load_dataset,
    load_balanced_data_for_layer,
    train_logistic_probe,
    train_mlp_probe,
    compute_per_template_performance,
    print_layer_results
)
from src.config import TODAY


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


def find_activations_file(model_id: str, base_dir: Path) -> Path:
    """Search for activations PKL file matching model name."""
    model_short = get_model_short_name(model_id)
    model_lower = model_short.lower().replace("-", "").replace("_", "")
    
    print(f"Searching for activations file...")
    print(f"  Model: {model_short}")
    
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


# =============================================================================
# CLI ARGUMENTS
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train layer-wise probes for faithfulness detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_layer_probes.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    python train_layer_probes.py --model "Qwen/Qwen3-32B" --layers 8 13 15 20
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Model ID (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')"
    )
    
    # File paths
    parser.add_argument(
        "--input-activations", 
        type=str, 
        default=None,
        help="Path to activations PKL (default: auto-discover based on model name)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (default: same directory as input activations)"
    )
    
    # Training configuration
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to train (default: inferred from activations dataset)"
    )
    parser.add_argument(
        "--hyper",
        type=int,
        nargs=2,
        default=[2, 8],
        metavar=("NUM_LAYERS", "NEURONS"),
        help="MLP architecture: (num_hidden_layers, neurons_per_layer). Default: 2 8"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Skip options
    parser.add_argument(
        "--skip-logreg",
        action="store_true",
        help="Skip logistic regression training (train only MLP)"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    return parser.parse_args()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_logreg_model(model, layer_idx: int, output_dir: Path):
    """Save logistic regression model to pickle file."""
    logreg_dir = output_dir / "logreg"
    logreg_dir.mkdir(parents=True, exist_ok=True)
    path = logreg_dir / f"layer_{layer_idx}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path


def save_mlp_model(model, metrics: dict, layer_idx: int, output_dir: Path, 
                   hidden_dim: int, num_hidden_layers: int, input_dim: int = 4096):
    """Save MLP model checkpoint with config and metrics."""
    mlp_dir = output_dir / "mlp"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    path = mlp_dir / f"layer_{layer_idx}.pth"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_hidden_layers': num_hidden_layers,
        },
        'metrics': metrics,
        'layer_idx': layer_idx,
    }
    
    torch.save(checkpoint, path)
    return path


def plot_layer_performance(all_results: dict, output_dir: Path):
    """Plot performance comparison across all layers."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted(all_results['mlp'].keys())
    
    # Check if we have logreg results
    has_logreg = len(all_results.get('logreg', {})) > 0
    
    mlp_val_accs = [all_results['mlp'][l]['val_accuracy'] for l in layers]
    
    # Plot 1: Performance across layers
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if has_logreg:
        logreg_val_accs = [all_results['logreg'][l]['val_accuracy'] for l in layers]
        ax.plot(layers, logreg_val_accs, marker='o', label='Logistic Regression', linewidth=2)
    
    ax.plot(layers, mlp_val_accs, marker='s', label='MLP', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Probe Performance Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'layer_performance.png', dpi=150)
    plt.close()
    
    print(f"\nPlots saved to: {plots_dir}/")


def save_results_summary(all_results: dict, output_dir: Path, args):
    """Save comprehensive results summary to JSON."""
    summary_path = output_dir / 'results_summary.json'
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {'logreg': {}, 'mlp': {}}
    
    for model_type in ['logreg', 'mlp']:
        for layer_idx, metrics in all_results.get(model_type, {}).items():
            serializable_metrics = {}
            for key, value in metrics.items():
                if key in ['train_predictions', 'val_predictions']:
                    continue
                
                if isinstance(value, (int, float, str, bool)):
                    serializable_metrics[key] = value
                elif isinstance(value, (list, tuple)):
                    try:
                        if len(value) > 0 and hasattr(value[0], 'item'):
                            serializable_metrics[key] = [float(v.item()) if hasattr(v, 'item') else float(v) for v in value]
                        else:
                            serializable_metrics[key] = list(value)
                    except:
                        serializable_metrics[key] = list(value)
                elif hasattr(value, 'item'):
                    try:
                        serializable_metrics[key] = float(value.item())
                    except:
                        continue
                else:
                    try:
                        serializable_metrics[key] = float(value)
                    except:
                        continue
            
            serializable_results[model_type][str(layer_idx)] = serializable_metrics
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'config': {
            'random_seed': args.random_seed,
            'num_hidden_layers': args.hyper[0],
            'hidden_dim': args.hyper[1],
            'layers_trained': args.layers,
        },
        'results': serializable_results,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to: {summary_path}")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    args = parse_args()
    
    # Determine base directory
    base_dir = Path(__file__).parent.resolve()
    
    print("="*80)
    print("LAYER-WISE PROBE TRAINING")
    print("="*80)
    print(f"Model: {args.model}")
    
    # Find activations file
    if args.input_activations:
        activations_file = Path(args.input_activations)
        if not activations_file.exists():
            raise FileNotFoundError(f"Activations file not found: {activations_file}")
    else:
        activations_file = find_activations_file(args.model, base_dir)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Save in same directory as activations, in a probes subfolder
        model_short = get_model_short_name(args.model)
        output_dir = activations_file.parent / f"probes_{model_short}_{TODAY}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset to infer layers and dimensions
    print("\nLoading dataset...")
    dataset = load_dataset(str(activations_file))
    
    # Infer available layers from dataset
    sample_data = list(dataset['data'].values())[0]
    available_layers = sorted(sample_data['layers'].keys())
    
    # Infer input dimension from first layer
    first_layer = available_layers[0]
    f_body = sample_data['layers'][first_layer].get('F_body')
    u_body = sample_data['layers'][first_layer].get('U_body')
    
    # Check tensors properly - can't use `or` with tensors
    if f_body is not None and f_body.numel() > 0:
        sample_tensor = f_body
    elif u_body is not None and u_body.numel() > 0:
        sample_tensor = u_body
    else:
        sample_tensor = None
    
    if sample_tensor is not None:
        input_dim = sample_tensor.shape[-1]
    else:
        input_dim = 4096
    
    # Determine layers to train
    if args.layers is None:
        layer_range = available_layers
    else:
        # Filter to only layers that exist in dataset
        layer_range = [l for l in args.layers if l in available_layers]
        if len(layer_range) != len(args.layers):
            missing = set(args.layers) - set(layer_range)
            print(f"Warning: Layers {missing} not found in dataset, skipping")
    
    # Extract hyper params
    num_hidden_layers, hidden_dim = args.hyper
    
    print(f"Activations: {activations_file}")
    print(f"Output directory: {output_dir}")
    print(f"Training {len(layer_range)} layers: {layer_range}")
    print(f"Input dimension: {input_dim}")
    print(f"Architecture: {num_hidden_layers} hidden layers × {hidden_dim} neurons")
    print(f"Random seed: {args.random_seed}")
    print(f"Skip LogReg: {args.skip_logreg}")
    
    # Training configs
    logreg_config = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': args.random_seed
    }
    
    mlp_config = {
        'hidden_dim': hidden_dim,
        'num_hidden_layers': num_hidden_layers,
        'learning_rate': 0.001,
        'batch_size': 32,
        'max_epochs': 200,
        'weight_decay': 0.01,
        'patience': 20,
        'min_delta': 0.0001,
        'random_seed': args.random_seed,
        'verbose': False
    }

    
    # Storage for all results
    all_results = {
        'logreg': {},
        'mlp': {},
    }
    
    # Train probes for each layer
    for layer_idx in layer_range:
        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*80}")
        
        # Load and balance data for this layer
        print("Loading and balancing data...")
        train_X, train_y, val_X, val_y, train_metadata, val_metadata = load_balanced_data_for_layer(
            dataset, layer_idx, args.random_seed
        )
        
        print(f"Train: {train_X.shape[0]} samples ({(train_y == 0).sum().item()} F, {(train_y == 1).sum().item()} U)")
        print(f"Val:   {val_X.shape[0]} samples ({(val_y == 0).sum().item()} F, {(val_y == 1).sum().item()} U)")
        
        # Train Logistic Regression (if not skipped)
        if not args.skip_logreg:
            print("\nTraining Logistic Regression...")
            logreg_model, logreg_metrics = train_logistic_probe(
                train_X, train_y, val_X, val_y, **logreg_config
            )
            logreg_path = save_logreg_model(logreg_model, layer_idx, output_dir)
            print(f"  Saved to: {logreg_path}")
            all_results['logreg'][layer_idx] = logreg_metrics
        else:
            logreg_metrics = None
        
        # Train MLP
        print("\nTraining MLP Probe...")
        mlp_model, mlp_metrics = train_mlp_probe(
            train_X, train_y, val_X, val_y, 
            **mlp_config
        )
        
        mlp_path = save_mlp_model(mlp_model, mlp_metrics, layer_idx, output_dir, 
                                  hidden_dim, num_hidden_layers, input_dim)
        print(f"  Saved to: {mlp_path}")
        
        all_results['mlp'][layer_idx] = mlp_metrics
        
        # Print results
        print_layer_results(layer_idx, logreg_metrics, mlp_metrics)
    
    # =========================================================================
    # POST-TRAINING ANALYSIS
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("POST-TRAINING ANALYSIS")
    print(f"{'='*80}")
    
    # Find top 5 layers by MLP val accuracy
    mlp_val_accs = {layer: metrics['val_accuracy'] for layer, metrics in all_results['mlp'].items()}
    top_5_layers = sorted(mlp_val_accs.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\nTop 5 Layers (by MLP validation accuracy):")
    for i, (layer, acc) in enumerate(top_5_layers, 1):
        logreg_acc = all_results['logreg'].get(layer, {}).get('val_accuracy', 'N/A')
        if isinstance(logreg_acc, float):
            print(f"  {i}. Layer {layer:2d}: MLP={acc:.4f}, LogReg={logreg_acc:.4f}")
        else:
            print(f"  {i}. Layer {layer:2d}: MLP={acc:.4f}")
    
    # Generate plots (if not skipped) - save in activations directory
    if not args.skip_plots:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")
        plot_layer_performance(all_results, activations_file.parent)
    
    # Save results summary
    print(f"\n{'='*80}")
    print("SAVING RESULTS SUMMARY")
    print(f"{'='*80}")
    args.layers = layer_range  # Store actual layers trained
    save_results_summary(all_results, output_dir, args)
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nModels saved to: {output_dir}/")
    if not args.skip_logreg:
        print(f"  - Logistic Regression: logreg/ ({len(layer_range)} .pkl files)")
    print(f"  - MLP: mlp/ ({len(layer_range)} .pth files)")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
