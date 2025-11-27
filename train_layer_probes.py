"""
train_layer_probes.py

Train linear and non-linear probes for all 32 layers to detect faithfulness vs unfaithfulness
in LLM activations. Saves trained models and generates performance analysis.

Usage:
    python train_layer_probes.py
"""

import os
import json
import pickle
from datetime import datetime

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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset
DATASET_PATH = "data/sprint_4_2025-10-15/datasets/new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"

# Output directories
OUTPUT_DIR = "results/probe_training"
LOGREG_DIR = os.path.join(OUTPUT_DIR, "logreg")
MLP_DIR = os.path.join(OUTPUT_DIR, "mlp")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Layer range
LAYER_RANGE = range(32)  # Train all 32 layers

# Random seed
RANDOM_SEED = 42

# Logistic Regression config
LOGREG_CONFIG = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': RANDOM_SEED
}

# MLP config
MLP_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'max_epochs': 200,
    'weight_decay': 0.01,
    'patience': 20,
    'min_delta': 0.0001,
    'verbose': False  # Set to True for detailed training output
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_logreg_model(model, layer_idx: int):
    """Save logistic regression model to pickle file."""
    os.makedirs(LOGREG_DIR, exist_ok=True)
    path = os.path.join(LOGREG_DIR, f"layer_{layer_idx}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path


def save_mlp_model(model, metrics: dict, layer_idx: int):
    """Save MLP model checkpoint with config and metrics."""
    os.makedirs(MLP_DIR, exist_ok=True)
    path = os.path.join(MLP_DIR, f"layer_{layer_idx}.pth")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 4096,
            'hidden_dim': 8,
        },
        'metrics': metrics,
        'layer_idx': layer_idx,
    }
    
    torch.save(checkpoint, path)
    return path


def plot_layer_performance(all_results: dict):
    """
    Plot performance comparison across all layers.
    
    Creates two plots:
    1. Val accuracy vs layer for both models
    2. MLP vs LogReg comparison
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    layers = sorted(all_results['logreg'].keys())
    
    logreg_val_accs = [all_results['logreg'][l]['val_accuracy'] for l in layers]
    mlp_val_accs = [all_results['mlp'][l]['val_accuracy'] for l in layers]
    
    # Plot 1: Performance across layers
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(layers, logreg_val_accs, marker='o', label='Logistic Regression', linewidth=2)
    ax.plot(layers, mlp_val_accs, marker='s', label='MLP (8 neurons)', linewidth=2)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Probe Performance Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'layer_performance.png'), dpi=150)
    plt.close()
    
    # Plot 2: MLP vs LogReg scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(logreg_val_accs, mlp_val_accs, s=100, alpha=0.6, edgecolors='black')
    
    # Add diagonal line (equal performance)
    min_val = min(min(logreg_val_accs), min(mlp_val_accs))
    max_val = max(max(logreg_val_accs), max(mlp_val_accs))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Equal Performance')
    
    ax.set_xlabel('Logistic Regression Val Accuracy', fontsize=12)
    ax.set_ylabel('MLP Val Accuracy', fontsize=12)
    ax.set_title('MLP vs Logistic Regression Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'mlp_vs_logreg.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to: {PLOTS_DIR}/")


def save_results_summary(all_results: dict):
    """Save comprehensive results summary to JSON."""
    summary_path = os.path.join(OUTPUT_DIR, 'results_summary.json')
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {'logreg': {}, 'mlp': {}}
    
    for model_type in ['logreg', 'mlp']:
        for layer_idx, metrics in all_results[model_type].items():
            # Create serializable copy
            serializable_metrics = {}
            for key, value in metrics.items():
                # Skip predictions (too large for JSON)
                if key in ['train_predictions', 'val_predictions']:
                    continue
                
                # Handle different value types
                if isinstance(value, (int, float, str, bool)):
                    serializable_metrics[key] = value
                elif isinstance(value, (list, tuple)):
                    # Already serializable or needs conversion
                    if len(value) > 0:
                        try:
                            # Try to convert first element
                            if hasattr(value[0], 'item'):
                                serializable_metrics[key] = [float(v.item()) if hasattr(v, 'item') else float(v) for v in value]
                            else:
                                serializable_metrics[key] = list(value)
                        except:
                            serializable_metrics[key] = list(value)
                    else:
                        serializable_metrics[key] = list(value)
                elif hasattr(value, 'item'):
                    # Numpy scalar - single element
                    try:
                        serializable_metrics[key] = float(value.item())
                    except:
                        # Skip if can't convert
                        continue
                else:
                    # Try direct conversion
                    try:
                        serializable_metrics[key] = float(value)
                    except:
                        # Skip if can't convert
                        continue
            
            serializable_results[model_type][str(layer_idx)] = serializable_metrics
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': DATASET_PATH,
        'config': {
            'random_seed': RANDOM_SEED,
            'logreg': LOGREG_CONFIG,
            'mlp': MLP_CONFIG,
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
    print("="*80)
    print("LAYER-WISE PROBE TRAINING")
    print("Faithfulness Detection via Linear and Non-linear Probes")
    print("="*80)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training {len(LAYER_RANGE)} layers (0-{max(LAYER_RANGE)})")
    print(f"Random seed: {RANDOM_SEED}\n")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGREG_DIR, exist_ok=True)
    os.makedirs(MLP_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load dataset once
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    
    # Storage for all results
    all_results = {
        'logreg': {},
        'mlp': {},
    }
    
    # Train probes for each layer
    for layer_idx in LAYER_RANGE:
        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*80}")
        
        # Load and balance data for this layer
        print("Loading and balancing data...")
        train_X, train_y, val_X, val_y, train_metadata, val_metadata = load_balanced_data_for_layer(
            dataset, layer_idx, RANDOM_SEED
        )
        
        print(f"Train: {train_X.shape[0]} samples ({(train_y == 0).sum().item()} F, {(train_y == 1).sum().item()} U)")
        print(f"Val:   {val_X.shape[0]} samples ({(val_y == 0).sum().item()} F, {(val_y == 1).sum().item()} U)")
        
        # Train Logistic Regression
        print("\nTraining Logistic Regression...")
        logreg_model, logreg_metrics = train_logistic_probe(
            train_X, train_y, val_X, val_y, **LOGREG_CONFIG
        )
        
        # Save LogReg model
        logreg_path = save_logreg_model(logreg_model, layer_idx)
        print(f"  Saved to: {logreg_path}")
        
        # Train MLP
        print("\nTraining MLP Probe...")
        mlp_model, mlp_metrics = train_mlp_probe(
            train_X, train_y, val_X, val_y, **MLP_CONFIG
        )
        
        # Save MLP model
        mlp_path = save_mlp_model(mlp_model, mlp_metrics, layer_idx)
        print(f"  Saved to: {mlp_path}")
        
        # Store results
        all_results['logreg'][layer_idx] = logreg_metrics
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
        logreg_acc = all_results['logreg'][layer]['val_accuracy']
        print(f"  {i}. Layer {layer:2d}: MLP={acc:.4f}, LogReg={logreg_acc:.4f}")
    
    # Per-template analysis for top 5 layers
    print(f"\n{'='*80}")
    print("PER-TEMPLATE ANALYSIS (Top 5 Layers)")
    print(f"{'='*80}")
    
    for layer, _ in top_5_layers:
        print(f"\n--- Layer {layer} ---")
        
        # Reload data for this layer
        train_X, train_y, val_X, val_y, train_metadata, val_metadata = load_balanced_data_for_layer(
            dataset, layer, RANDOM_SEED
        )
        
        # Load models
        logreg_model = pickle.load(open(os.path.join(LOGREG_DIR, f"layer_{layer}.pkl"), 'rb'))
        mlp_checkpoint = torch.load(os.path.join(MLP_DIR, f"layer_{layer}.pth"))
        mlp_model = torch.nn.Module()  # Placeholder
        from src.probe import MLPProbe
        mlp_model = MLPProbe(4096, 8)
        mlp_model.load_state_dict(mlp_checkpoint['model_state_dict'])
        mlp_model.eval()
        
        # Compute per-template performance
        logreg_template_results = compute_per_template_performance(
            logreg_model, val_X, val_y, val_metadata, model_type='logreg'
        )
        mlp_template_results = compute_per_template_performance(
            mlp_model, val_X, val_y, val_metadata, model_type='mlp'
        )
        
        print("\nLogistic Regression (per-template val accuracy):")
        for template in sorted(logreg_template_results.keys()):
            acc = logreg_template_results[template]['accuracy']
            total = logreg_template_results[template]['total']
            print(f"  {template:20s}: {acc:.4f} ({total} samples)")
        
        print("\nMLP Probe (per-template val accuracy):")
        for template in sorted(mlp_template_results.keys()):
            acc = mlp_template_results[template]['accuracy']
            total = mlp_template_results[template]['total']
            print(f"  {template:20s}: {acc:.4f} ({total} samples)")
    
    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    plot_layer_performance(all_results)
    
    # Save results summary
    print(f"\n{'='*80}")
    print("SAVING RESULTS SUMMARY")
    print(f"{'='*80}")
    save_results_summary(all_results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nModels saved:")
    print(f"  - Logistic Regression: {LOGREG_DIR}/ (32 .pkl files)")
    print(f"  - MLP: {MLP_DIR}/ (32 .pth files)")
    print(f"\nResults:")
    print(f"  - Plots: {PLOTS_DIR}/")
    print(f"  - Summary: {OUTPUT_DIR}/results_summary.json")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
