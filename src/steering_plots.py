"""
Steering Results Plotting Module

Loads Qwen3-32B summary JSON files and generates plot variations
showing steering effects on faithfulness, correctness, and hint-mentioning.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MetricSet:
    """Metrics extracted from a single steering configuration."""
    faithful_pct: float = 0.0  # F%: stable_faithful rate
    unfaithful_pct: float = 0.0  # U%: stable_unfaithful rate
    correct_pct: float = 0.0  # C%: wrong_to_correct rate
    incomplete_pct: float = 0.0  # I%: incomplete rate
    non_hint_error_pct: float = 0.0  # NHE%: hint_error + error rates
    hint_mentioning_pct: float = 0.0  # Hm%: sum of *_mentioning_hint
    n: int = 0  # sample size


@dataclass
class ConfigResult:
    """Results for a single (layer, coefficient) configuration."""
    layer: int
    coefficient: float
    # Positive steering (toward faithfulness)
    positive_WF: MetricSet = field(default_factory=MetricSet)
    positive_WU: MetricSet = field(default_factory=MetricSet)
    # Negative steering (toward unfaithfulness)
    negative_WF: MetricSet = field(default_factory=MetricSet)
    negative_WU: MetricSet = field(default_factory=MetricSet)


@dataclass
class ApproachData:
    """Data for a steering approach (linear, mlp, off_policy)."""
    best_config: Optional[tuple[int, float]] = None
    best_result: Optional[ConfigResult] = None
    all_configs: list[ConfigResult] = field(default_factory=list)


@dataclass
class ModelData:
    """Data for a single model across all approaches."""
    linear: ApproachData = field(default_factory=ApproachData)
    mlp: ApproachData = field(default_factory=ApproachData)
    off_policy: ApproachData = field(default_factory=ApproachData)


# =============================================================================
# Data Loading
# =============================================================================

def extract_metrics(transitions: dict) -> MetricSet:
    """Extract metrics from a transitions dict."""
    def get_rate(key: str) -> float:
        return transitions.get(key, {}).get("rate", 0.0)
    
    # Sum all *_mentioning_hint rates
    hint_mentioning = sum(
        v.get("rate", 0.0) for k, v in transitions.items()
        if "mentioning_hint" in k and isinstance(v, dict)
    )
    
    return MetricSet(
        faithful_pct=get_rate("stable_faithful") * 100,
        unfaithful_pct=get_rate("stable_unfaithful") * 100,
        correct_pct=get_rate("wrong_to_correct") * 100,
        incomplete_pct=get_rate("incomplete") * 100,
        non_hint_error_pct=(get_rate("hint_error") + get_rate("error")) * 100,
        hint_mentioning_pct=hint_mentioning * 100,
    )


def load_summary_json(filepath: Path) -> dict:
    """Load and return a summary JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_configurations(
    summary: dict,
    hint_template: str = "grader_hacking"
) -> list[ConfigResult]:
    """Parse configurations from a summary JSON for a given hint template."""
    configs = summary.get("configurations_by_hint", {}).get(hint_template, [])
    results = []
    
    for cfg in configs:
        layer = cfg.get("layer")
        coeff = cfg.get("coefficient_magnitude")
        
        result = ConfigResult(layer=layer, coefficient=coeff)
        
        # Extract all 4 steering directions
        for direction, attr in [
            ("positive_on_WF", "positive_WF"),
            ("positive_on_WU", "positive_WU"),
            ("negative_on_WF", "negative_WF"),
            ("negative_on_WU", "negative_WU"),
        ]:
            data = cfg.get(direction, {})
            metrics = extract_metrics(data.get("transitions", {}))
            metrics.n = data.get("n", 0)
            setattr(result, attr, metrics)
        
        results.append(result)
    
    return results


def find_best_config(configs: list[ConfigResult], eps: float = 1e-6) -> ConfigResult:
    """
    Find the best configuration by maximizing:
        score = (+WU → F%) / (+WF → U% + eps)
    
    This rewards making unfaithful inputs faithful 
    while penalizing making faithful inputs unfaithful.
    """
    best_score = float("-inf")
    best = None
    
    for cfg in configs:
        wu_to_f = cfg.positive_WU.faithful_pct
        wf_to_u = cfg.positive_WF.unfaithful_pct
        score = wu_to_f / (wf_to_u + eps)
        
        if score > best_score:
            best_score = score
            best = cfg
    
    return best


def load_approach_data(
    data_dir: Path,
    mode: str,
    model_suffix: str,
    hint_template: str = "grader_hacking"
) -> ApproachData:
    """Load data for a single approach (mode) from the most recent summary file."""
    # Find summary files matching pattern
    pattern = f"summary_steered_{mode}_{model_suffix}_*.json"
    files = list(data_dir.glob(pattern))
    
    if not files:
        print(f"Warning: No files matching {pattern} in {data_dir}")
        return ApproachData()
    
    # Use the most recent file (by filename date)
    latest_file = sorted(files)[-1]
    print(f"Loading: {latest_file.name}")
    
    summary = load_summary_json(latest_file)
    configs = parse_configurations(summary, hint_template)
    
    if not configs:
        return ApproachData()
    
    best = find_best_config(configs)
    
    return ApproachData(
        best_config=(best.layer, best.coefficient) if best else None,
        best_result=best,
        all_configs=configs
    )


def load_qwen3_32b_data(
    base_dir: Path,
    hint_template: str = "grader_hacking"
) -> ModelData:
    """Load all Qwen3-32B steering data."""
    data_dir = base_dir / "data" / "definitive_pipeline_data" / "Qwen3-32B"
    
    model = ModelData()
    model.linear = load_approach_data(data_dir, "linear", "Qwen3-32B", hint_template)
    model.mlp = load_approach_data(data_dir, "mlp", "Qwen3-32B", hint_template)
    model.off_policy = load_approach_data(data_dir, "off_policy", "Qwen3-32B", hint_template)
    
    return model


# =============================================================================
# Placeholder Data Generation
# =============================================================================

def generate_placeholder_metrics() -> MetricSet:
    """Generate random placeholder metrics for visualization testing."""
    faithful = random.uniform(10, 40)
    unfaithful = random.uniform(20, 50)
    correct = random.uniform(15, 45)
    incomplete = random.uniform(0, 10)
    nhe = random.uniform(0, 15)
    hm = random.uniform(10, 30)
    
    return MetricSet(
        faithful_pct=faithful,
        unfaithful_pct=unfaithful,
        correct_pct=correct,
        incomplete_pct=incomplete,
        non_hint_error_pct=nhe,
        hint_mentioning_pct=hm,
        n=30
    )


def generate_placeholder_config() -> ConfigResult:
    """Generate a placeholder configuration result."""
    return ConfigResult(
        layer=random.choice([13, 24, 31, 40]),
        coefficient=random.choice([0.6, 0.75, 1.0]),
        positive_WF=generate_placeholder_metrics(),
        positive_WU=generate_placeholder_metrics(),
        negative_WF=generate_placeholder_metrics(),
        negative_WU=generate_placeholder_metrics()
    )


def generate_placeholder_approach() -> ApproachData:
    """Generate placeholder approach data."""
    cfg = generate_placeholder_config()
    return ApproachData(
        best_config=(cfg.layer, cfg.coefficient),
        best_result=cfg,
        all_configs=[cfg]
    )


def generate_placeholder_model() -> ModelData:
    """Generate placeholder model data."""
    return ModelData(
        linear=generate_placeholder_approach(),
        mlp=generate_placeholder_approach(),
        off_policy=generate_placeholder_approach()
    )


# =============================================================================
# Plotting Functions
# =============================================================================

# Color palette for approaches
APPROACH_COLORS = {
    "linear": "#3498db",    # Blue
    "mlp": "#e74c3c",       # Red
    "off_policy": "#2ecc71" # Green
}

# Color palette for models (lighter shades within approach groups)
MODEL_SHADES = {
    "Qwen3-32B": 1.0,           # Full saturation
    "DeepSeek-R1-Llama-8B": 0.6,  # Medium
    "Qwen3-14B": 0.3            # Light
}


def get_bar_color(approach: str, model: str) -> str:
    """Get color for a bar based on approach and model."""
    base_color = APPROACH_COLORS.get(approach, "#95a5a6")
    shade = MODEL_SHADES.get(model, 1.0)
    # For simplicity, just return base color with alpha adjustment via hex
    # In actual plotting, we'll use alpha parameter
    return base_color


def plot_variation_1(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 1: 1 row with 2 barplots
    - Left: +WU → F% (intended effect)
    - Right: +WF → U% (collateral damage)
    
    9 bars each: 3 approaches × 3 models
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    # Left plot: +WU → F%
    ax = axes[0]
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                values.append(approach_data.best_result.positive_WU.faithful_pct)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)
    
    ax.set_ylabel("Faithful %")
    ax.set_title("+Steer Wrong-Unfaithful → Faithful %\n(Intended Effect)")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ").title() for a in approaches])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    
    # Right plot: +WF → U%
    ax = axes[1]
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                values.append(approach_data.best_result.positive_WF.unfaithful_pct)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)
    
    ax.set_ylabel("Unfaithful %")
    ax.set_title("+Steer Wrong-Faithful → Unfaithful %\n(Collateral Damage)")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ").title() for a in approaches])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_4(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 4: 2 rows
    - Row 1: F% and U% (faithfulness)
    - Row 2: C%, I%, NHE% (other metrics)
    
    Each subplot has 9 bars: 3 approaches × 3 models
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    plot_configs = [
        (axes[0, 0], "positive_WU", "faithful_pct", "+WU → F%", "Faithful %"),
        (axes[0, 1], "positive_WF", "unfaithful_pct", "+WF → U%", "Unfaithful %"),
        (axes[1, 0], "positive_WU", "correct_pct", "+WU → C%", "Correct %"),
        (axes[1, 1], "positive_WU", "incomplete_pct", "+WU → I%", "Incomplete %"),
    ]
    
    for ax, direction, metric, title, ylabel in plot_configs:
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    metrics = getattr(approach_data.best_result, direction, None)
                    values.append(getattr(metrics, metric, 0) if metrics else 0)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", " ").title() for a in approaches])
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for generating steering result plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate steering result plots")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"c:\Users\occhi\Desktop\unfaithfulness_steering"),
        help="Base directory of the project"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: base_dir/plots)"
    )
    parser.add_argument(
        "--hint-template",
        type=str,
        default="grader_hacking",
        choices=["grader_hacking", "metadata", "professor"],
        help="Hint template to use"
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir or (args.base_dir / "plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load Qwen3-32B data
    print("Loading Qwen3-32B data...")
    qwen_data = load_qwen3_32b_data(args.base_dir, args.hint_template)
    
    # Generate placeholder data for other models
    print("Generating placeholder data for other models...")
    random.seed(42)  # Reproducible placeholders
    
    all_data = {
        "Qwen3-32B": qwen_data,
        "DeepSeek-R1-Llama-8B": generate_placeholder_model(),
        "Qwen3-14B": generate_placeholder_model()
    }
    
    # Print best configurations
    print("\n=== Best Configurations (Qwen3-32B) ===")
    for approach in ["off_policy", "linear", "mlp"]:
        approach_data = getattr(qwen_data, approach)
        if approach_data.best_config:
            layer, coeff = approach_data.best_config
            result = approach_data.best_result
            print(f"\n{approach.upper()}:")
            print(f"  Best config: layer={layer}, coeff={coeff}")
            print(f"  +WU → F%: {result.positive_WU.faithful_pct:.1f}%")
            print(f"  +WF → U%: {result.positive_WF.unfaithful_pct:.1f}%")
            print(f"  +WU → C%: {result.positive_WU.correct_pct:.1f}%")
    
    # Generate plots
    print("\n=== Generating Plots ===")
    
    plot_variation_1(all_data, output_dir / "variation_1_intended_vs_collateral.png")
    plot_variation_4(all_data, output_dir / "variation_4_faithfulness_and_metrics.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
