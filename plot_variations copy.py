"""
Variation 1: Two Rows — Good Behavior of Both Steering Directions

Narrative: "Both steering directions work as intended."

| Row             | Left (9 bars)       | Right (9 bars)      |
|-----------------|---------------------|---------------------|
| Faithfulness    | +steer WU→F%        | −steer WF→U%        |
| Hint-Mentioning | +steer WU→Hm%       | −steer WF→Hnm%      |

Each panel: 9 bars = 3 approaches × 3 models (grouped by approach)
Using top configuration per (approach × model) via argmax.
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
    """Metrics extracted from one steering direction."""
    faithful_pct: float = 0.0      # F%: stable_faithful rate
    unfaithful_pct: float = 0.0    # U%: stable_unfaithful rate
    correct_pct: float = 0.0       # C%: wrong_to_correct rate
    hint_mentioning_pct: float = 0.0  # Hm%: sum of *_mentioning_hint
    n: int = 0


@dataclass
class ConfigResult:
    """Results for a single (layer, coefficient) configuration."""
    layer: int
    coefficient: float
    positive_WF: MetricSet = field(default_factory=MetricSet)
    positive_WU: MetricSet = field(default_factory=MetricSet)
    negative_WF: MetricSet = field(default_factory=MetricSet)
    negative_WU: MetricSet = field(default_factory=MetricSet)


@dataclass
class ApproachData:
    """Data for a steering approach (off_policy, linear, mlp)."""
    best_config: Optional[tuple[int, float]] = None
    best_result: Optional[ConfigResult] = None
    all_configs: list[ConfigResult] = field(default_factory=list)


@dataclass
class ModelData:
    """Data for a single model across all approaches."""
    off_policy: ApproachData = field(default_factory=ApproachData)
    linear: ApproachData = field(default_factory=ApproachData)
    mlp: ApproachData = field(default_factory=ApproachData)


# =============================================================================
# Data Loading
# =============================================================================

def extract_metrics(transitions: dict) -> MetricSet:
    """Extract metrics from a transitions dictionary."""
    def get_rate(key: str) -> float:
        return transitions.get(key, {}).get("rate", 0.0)
    
    hint_mentioning = sum(
        v.get("rate", 0.0) for k, v in transitions.items()
        if "mentioning_hint" in k and isinstance(v, dict)
    )
    
    return MetricSet(
        faithful_pct=get_rate("stable_faithful") * 100,
        unfaithful_pct=get_rate("stable_unfaithful") * 100,
        correct_pct=get_rate("wrong_to_correct") * 100,
        hint_mentioning_pct=hint_mentioning * 100,
    )


def parse_configurations(summary: dict) -> list[ConfigResult]:
    """Parse all configurations from summary JSON, appearing in ANY hint template, and average metrics."""
    all_configs_by_hint = summary.get("configurations_by_hint", {})
    
    # Dictionary to collect list of MetricSets for each component of each config
    # Key: (layer, coefficient)
    # Value: { 'positive_WF': [MetricSet], 'positive_WU': [MetricSet], ... }
    aggregated_data = {}

    # 1. Collect all metrics
    for hint_template, configs in all_configs_by_hint.items():
        for cfg in configs:
            key = (cfg.get("layer"), cfg.get("coefficient_magnitude"))
            
            if key not in aggregated_data:
                aggregated_data[key] = {
                    "positive_WF": [], "positive_WU": [],
                    "negative_WF": [], "negative_WU": []
                }
            
            for json_key, list_key in [
                ("positive_on_WF", "positive_WF"),
                ("positive_on_WU", "positive_WU"),
                ("negative_on_WF", "negative_WF"),
                ("negative_on_WU", "negative_WU"),
            ]:
                data = cfg.get(json_key, {})
                metrics = extract_metrics(data.get("transitions", {}))
                metrics.n = data.get("n", 0)
                aggregated_data[key][list_key].append(metrics)

    # 2. Average metrics across templates (Equal contribution)
    results = []
    for (layer, coeff), data in aggregated_data.items():
        result = ConfigResult(layer=layer, coefficient=coeff)
        
        for list_key in ["positive_WF", "positive_WU", "negative_WF", "negative_WU"]:
            metric_list = data[list_key]
            if not metric_list:
                continue
            
            # Calculate simple average of rates (counts are summed just for info)
            count = len(metric_list)
            avg_metrics = MetricSet()
            
            if count > 0:
                avg_metrics.faithful_pct = sum(m.faithful_pct for m in metric_list) / count
                avg_metrics.unfaithful_pct = sum(m.unfaithful_pct for m in metric_list) / count
                avg_metrics.correct_pct = sum(m.correct_pct for m in metric_list) / count
                avg_metrics.hint_mentioning_pct = sum(m.hint_mentioning_pct for m in metric_list) / count
                avg_metrics.n = sum(m.n for m in metric_list) # Sum N to show total samples involved
            
            setattr(result, list_key, avg_metrics)
        
        results.append(result)
    
    return results


def find_best_config(configs: list[ConfigResult]) -> Optional[ConfigResult]:
    """
    Select best (layer, coefficient) by maximizing:
        score = (+WU → F%) / (+WF → U% + eps)
    
    Rewards intended effect, penalizes collateral damage.
    """
    best_score = float("-inf")
    best = None
    eps = 1e-6  # Prevent division by zero
    
    for cfg in configs:
        intended = cfg.positive_WU.faithful_pct    # +steer WU → F%
        collateral = cfg.positive_WF.unfaithful_pct  # +steer WF → U%
        score = intended / (collateral + eps)
        
        if score > best_score:
            best_score = score
            best = cfg
    
    return best


def load_approach(data_dir: Path, mode: str, model_suffix: str) -> ApproachData:
    """Load data for a single approach from summary file."""
    pattern = f"summary_steered_{mode}_{model_suffix}_*.json"
    files = list(data_dir.glob(pattern))
    
    if not files:
        print(f"  Warning: No files matching {pattern}")
        return ApproachData()
    
    latest = sorted(files)[-1]
    print(f"  Loading {mode}: {latest.name}")
    
    with open(latest, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    configs = parse_configurations(summary)
    best = find_best_config(configs) if configs else None
    
    return ApproachData(
        best_config=(best.layer, best.coefficient) if best else None,
        best_result=best,
        all_configs=configs
    )


def load_qwen3_32b(base_dir: Path) -> ModelData:
    """Load all Qwen3-32B data."""
    data_dir = base_dir / "data" / "definitive_pipeline_data" / "Qwen3-32B"
    print("Loading Qwen3-32B...")
    
    model = ModelData()
    model.off_policy = load_approach(data_dir, "off_policy", "Qwen3-32B")
    model.linear = load_approach(data_dir, "linear", "Qwen3-32B")
    model.mlp = load_approach(data_dir, "mlp", "Qwen3-32B")
    
    return model


def load_qwen3_14b(base_dir: Path) -> ModelData:
    """Load all Qwen3-14B data."""
    data_dir = base_dir / "data" / "definitive_pipeline_data" / "Qwen3-14B"
    print("Loading Qwen3-14B...")
    
    model = ModelData()
    model.off_policy = load_approach(data_dir, "off_policy", "Qwen3-14B")
    model.linear = load_approach(data_dir, "linear", "Qwen3-14B")
    model.mlp = load_approach(data_dir, "mlp", "Qwen3-14B")
    
    return model


def load_deepseek_r1_distill_llama_8b(base_dir: Path) -> ModelData:
    """Load all DeepSeek-R1-Distill-Llama-8B data."""
    data_dir = base_dir / "data" / "definitive_pipeline_data" / "DeepSeek-R1-Distill-Llama-8B"
    print("Loading DeepSeek-R1-Distill-Llama-8B...")
    
    model = ModelData()
    model.off_policy = load_approach(data_dir, "off_policy", "DeepSeek-R1-Distill-Llama-8B")
    model.linear = load_approach(data_dir, "linear", "DeepSeek-R1-Distill-Llama-8B")
    model.mlp = load_approach(data_dir, "mlp", "DeepSeek-R1-Distill-Llama-8B")
    
    return model


# =============================================================================
# Placeholder Generation
# =============================================================================

def generate_placeholder_metrics() -> MetricSet:
    """Generate random placeholder metrics."""
    return MetricSet(
        faithful_pct=random.uniform(10, 40),
        unfaithful_pct=random.uniform(15, 45),
        correct_pct=random.uniform(20, 50),
        hint_mentioning_pct=random.uniform(10, 35),
        n=30
    )


def generate_placeholder_model() -> ModelData:
    """Generate placeholder data for a model."""
    model = ModelData()
    for approach in ["off_policy", "linear", "mlp"]:
        cfg = ConfigResult(
            layer=random.choice([13, 24, 31, 40]),
            coefficient=random.choice([0.6, 0.75, 1.0]),
            positive_WF=generate_placeholder_metrics(),
            positive_WU=generate_placeholder_metrics(),
            negative_WF=generate_placeholder_metrics(),
            negative_WU=generate_placeholder_metrics()
        )
        setattr(model, approach, ApproachData(
            best_config=(cfg.layer, cfg.coefficient),
            best_result=cfg,
            all_configs=[cfg]
        ))
    return model


# =============================================================================
# Plotting: Variation 1
# =============================================================================

def plot_variation_1(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 1: Two Rows — Good Behavior of Both Directions
    
    | Row             | Left (9 bars)  | Right (9 bars) |
    |-----------------|----------------|----------------|
    | Faithfulness    | +steer WU→F%   | −steer WF→U%   |
    | Hint-Mentioning | +steer WU→Hm%  | −steer WF→Hnm% |
    """
    fig, axes = plt.subplots(2, 2, figsize=(28, 24))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    # Swapped order: Linear first, then Off-Policy
    approaches = ["linear", "off_policy", "mlp"]
    approach_labels = ["Linear", "Off-Policy", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    # Define color palettes (Light -> Medium -> Dark)
    # 0 (8B), 1 (14B), 2 (32B)
    palettes = {
        "good": ["#A5D6A7", "#4CAF50", "#1B5E20"],      # Green
        "bad": ["#EF9A9A", "#F44336", "#B71C1C"],       # Red
        "meh_good": ["#81D4FA", "#29B6F6", "#01579B"],  # Light Blue
        "meh_bad": ["#FFF59D", "#FBC02D", "#F57F17"],   # Yellow/Orange
    }
    
    # Map 'models' index to shade index: 
    # models[0]=32B -> Dark (2)
    # models[1]=8B  -> Light (0)
    # models[2]=14B -> Medium (1)
    shade_indices = [2, 0, 1]
    
    # Panel specifications: (row, col), direction_attr, metric_attr, title, ylabel, is_hnm, palette_key
    panels = [
        # Row 0: Faithfulness
        (0, 0, "positive_WU", "faithful_pct", 
         "+ Steering on Unfaithful Answers:\nFaithfulness Rate", "Faithful %", False, "good"),
        (0, 1, "negative_WF", "unfaithful_pct", 
         "- Steering on Faithful Answers:\nUnfaithfulness Rate", "Unfaithful %", False, "bad"),
        # Row 1: Hint-Mentioning
        (1, 0, "positive_WU", "hint_mentioning_pct", 
         "+ Steering on Unfaithful Answers:\nHint-mentioning Rate", "Hint-Mentioning %", False, "meh_good"),
        (1, 1, "negative_WF", "hint_mentioning_pct", 
         "- Steering on Faithful Answers:\nHint-mentioning-ablation Rate", "Hint-mentioning-ablation %", True, "meh_bad"),
    ]
    
    for row, col, direction, metric, title, ylabel, is_hnm, palette_key in panels:
        ax = axes[row, col]
        
        current_palette = palettes[palette_key]
        
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    metrics = getattr(approach_data.best_result, direction, None)
                    value = getattr(metrics, metric, 0) if metrics else 0
                    # Hnm% = 100 - Hm%
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            
            # Determine color based on model size shade
            shade_idx = shade_indices[i]
            color = current_palette[shade_idx]
            
            # Use shorter display labels for legend
            model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
            ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
        
        # 2x font sizes for print readability
        ax.set_ylabel(ylabel, fontsize=42, fontweight="bold")
        ax.set_title(title, fontsize=36, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=48)
        ax.tick_params(axis="y", labelsize=28)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=36)
    
    # Main title
    fig.suptitle("Steering Performance Towards and Away From Faithfulness", 
                 fontsize=40, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.3)  # Add vertical and horizontal spacing
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_2(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 2: Two Rows — Good vs Bad Behavior of +Steering Only
    
    Narrative: "Here's what +steering achieves, and here's the price we pay."
    
    | Row             | Left (9 bars)       | Right (9 bars)      |
    |-----------------|---------------------|---------------------|
    | Faithfulness    | +steer WU→F% (good) | +steer WF→U% (bad)  |
    | Hint-Mentioning | +steer WU→Hm% (good)| +steer WF→Hnm% (bad)|
    """
    fig, axes = plt.subplots(2, 2, figsize=(28, 24))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    # Shade indices: 32B=Dark(2), 8B=Light(0), 14B=Medium(1)
    shade_indices = [2, 0, 1]
    
    # Define palettes (Light, Medium, Dark)
    palettes = {
        "good": ["#a8e6cf", "#2ecc71", "#219150"],      # Greens
        "bad": ["#ffb3b3", "#ff4d4d", "#b30000"],        # Reds
        "meh_good": ["#AED6F1", "#3498db", "#1F618D"],   # Blues
        "meh_bad": ["#F9E79F", "#F1C40F", "#B7950B"],    # Yellows
    }
    
    x = np.arange(len(approaches))
    width = 0.25
    
    # Panel specifications: (row, col), direction_attr, metric_attr, title, ylabel, is_hnm, palette_key
    panels = [
        # Row 0: Faithfulness
        (0, 0, "positive_WU", "faithful_pct", 
         "+Steering on Unfaithful Answers:\nFaithfulness Rate (Good)", "Faithful %", False, "good"),
        (0, 1, "positive_WF", "unfaithful_pct", 
         "+Steering on Faithful Answers:\nUnfaithfulness Rate (Bad)", "Unfaithful %", False, "bad"),
        # Row 1: Hint-Mentioning
        (1, 0, "positive_WU", "hint_mentioning_pct", 
         "+Steering on Unfaithful Answers:\nHint-mentioning Rate (Good)", "Hint-Mentioning %", False, "meh_good"),
        (1, 1, "positive_WF", "hint_mentioning_pct", 
         "+Steering on Faithful Answers:\nHint-not-mentioning Rate (Bad)", "Hint-Not-Mentioning %", True, "meh_bad"),
    ]
    
    for row, col, direction, metric, title, ylabel, is_hnm, palette_key in panels:
        ax = axes[row, col]
        current_palette = palettes[palette_key]
        
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    metrics = getattr(approach_data.best_result, direction, None)
                    value = getattr(metrics, metric, 0) if metrics else 0
                    # Hnm% = 100 - Hm%
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            
            # Determine color based on model size shade
            shade_idx = shade_indices[i]
            color = current_palette[shade_idx]
            
            # Use shorter display labels for legend
            model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
            ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
        
        # 2x font sizes for print readability
        ax.set_ylabel(ylabel, fontsize=42, fontweight="bold")
        ax.set_title(title, fontsize=36, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=48)
        ax.tick_params(axis="y", labelsize=28)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=36)
    
    # Main title
    fig.suptitle("Variation 2: Good vs Bad Behavior of +Steering", 
                 fontsize=40, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_3a(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 3A: One Row — Compressed Version of V1 (Both Directions)
    
    4 panels in a single row: +WU→F%, −WF→U%, +WU→Hm%, −WF→Hnm%
    """
    # 4 columns, so we need huge width
    fig, axes = plt.subplots(1, 4, figsize=(56, 12))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    shade_indices = [2, 0, 1]
    palettes = {
        "good": ["#a8e6cf", "#2ecc71", "#219150"],
        "bad": ["#ffb3b3", "#ff4d4d", "#b30000"],
        "meh_good": ["#AED6F1", "#3498db", "#1F618D"],
        "meh_bad": ["#F9E79F", "#F1C40F", "#B7950B"],
    }
    
    x = np.arange(len(approaches))
    width = 0.25
    
    panels = [
        (0, "positive_WU", "faithful_pct", "+WU → F%", "Faithful %", False, "good"),
        (1, "negative_WF", "unfaithful_pct", "−WF → U%", "Unfaithful %", False, "bad"),
        (2, "positive_WU", "hint_mentioning_pct", "+WU → Hm%", "Hint-Mentioning %", False, "meh_good"),
        (3, "negative_WF", "hint_mentioning_pct", "−WF → Hnm%", "Hint-Not-Mentioning %", True, "meh_bad"),
    ]
    
    for col, direction, metric, title, ylabel, is_hnm, palette_key in panels:
        ax = axes[col]
        current_palette = palettes[palette_key]
        
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    metrics = getattr(approach_data.best_result, direction, None)
                    value = getattr(metrics, metric, 0) if metrics else 0
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            shade_idx = shade_indices[i]
            color = current_palette[shade_idx]
            model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
            
            ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
        
        ax.set_ylabel(ylabel, fontsize=36, fontweight="bold")
        ax.set_title(title, fontsize=36, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=42)
        ax.tick_params(axis="y", labelsize=28)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if col == 3:
            ax.legend(loc="upper right", fontsize=32)
    
    fig.suptitle("Variation 3A: Both Steering Directions (Compressed)", 
                 fontsize=40, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_3b(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 3B: One Row — Compressed Version of V2 (+Steering Only)
    
    4 panels: +WU→F% (good), +WF→U% (bad), +WU→Hm% (good), +WF→Hnm% (bad)
    """
    fig, axes = plt.subplots(1, 4, figsize=(56, 12))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    shade_indices = [2, 0, 1]
    palettes = {
        "good": ["#a8e6cf", "#2ecc71", "#219150"],
        "bad": ["#ffb3b3", "#ff4d4d", "#b30000"],
        "meh_good": ["#AED6F1", "#3498db", "#1F618D"],
        "meh_bad": ["#F9E79F", "#F1C40F", "#B7950B"],
    }
    
    x = np.arange(len(approaches))
    width = 0.25
    
    panels = [
        (0, "positive_WU", "faithful_pct", "+WU → F% (Good)", "Faithful %", False, "good"),
        (1, "positive_WF", "unfaithful_pct", "+WF → U% (Bad)", "Unfaithful %", False, "bad"),
        (2, "positive_WU", "hint_mentioning_pct", "+WU → Hm% (Good)", "Hint-Mentioning %", False, "meh_good"),
        (3, "positive_WF", "hint_mentioning_pct", "+WF → Hnm% (Bad)", "Hint-Not-Mentioning %", True, "meh_bad"),
    ]
    
    for col, direction, metric, title, ylabel, is_hnm, palette_key in panels:
        ax = axes[col]
        current_palette = palettes[palette_key]
        
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    metrics = getattr(approach_data.best_result, direction, None)
                    value = getattr(metrics, metric, 0) if metrics else 0
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            shade_idx = shade_indices[i]
            color = current_palette[shade_idx]
            model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
            
            ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
        
        ax.set_ylabel(ylabel, fontsize=36, fontweight="bold")
        ax.set_title(title, fontsize=36, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=42)
        ax.tick_params(axis="y", labelsize=28)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if col == 3:
            ax.legend(loc="upper right", fontsize=32)
    
    fig.suptitle("Variation 3B: +Steering Good vs Bad (Compressed)", 
                 fontsize=40, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_4(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 4: Three Rows — Add Correctness to V1
    
    | Row             | Left (9 bars)  | Right (9 bars) |
    |-----------------|----------------|----------------|
    | Faithfulness    | +steer WU→F%   | −steer WF→U%   |
    | Hint-Mentioning | +steer WU→Hm%  | −steer WF→Hnm% |
    | Correctness     | +steer I→C%    | −steer I→C%    |
    
    Note: I→C% pools WF and WU as "Incorrect" initial state.
    """
    fig, axes = plt.subplots(3, 2, figsize=(28, 36))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    # Shade indices: 32B=Dark(2), 8B=Light(0), 14B=Medium(1)
    shade_indices = [2, 0, 1]
    
    # Define palettes
    palettes = {
        "good": ["#a8e6cf", "#2ecc71", "#219150"],      # Greens
        "bad": ["#ffb3b3", "#ff4d4d", "#b30000"],        # Reds
        "meh_good": ["#AED6F1", "#3498db", "#1F618D"],   # Blues
        "meh_bad": ["#F9E79F", "#F1C40F", "#B7950B"],    # Yellows
    }
    
    x = np.arange(len(approaches))
    width = 0.25
    
    panels = [
        (0, 0, "positive_WU", "faithful_pct", "+Steer WU → F%", "Faithful %", False, "good"),
        (0, 1, "negative_WF", "unfaithful_pct", "−Steer WF → U%", "Unfaithful %", False, "bad"),
        (1, 0, "positive_WU", "hint_mentioning_pct", "+Steer WU → Hm%", "Hint-M %", False, "meh_good"),
        (1, 1, "negative_WF", "hint_mentioning_pct", "−Steer WF → Hnm%", "Hint-nM %", True, "meh_bad"),
        (2, 0, "positive_WU", "correct_pct", "+Steer I → C%", "Correctness %", False, "good"),
        (2, 1, "negative_WU", "correct_pct", "−Steer I → C%", "Correctness %", False, "good"),
    ]
    
    for row, col, direction, metric, title, ylabel, is_hnm, palette_key in panels:
        ax = axes[row, col]
        current_palette = palettes[palette_key]
        
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    metrics = getattr(approach_data.best_result, direction, None)
                    value = getattr(metrics, metric, 0) if metrics else 0
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            shade_idx = shade_indices[i]
            color = current_palette[shade_idx]
            model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
            
            ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
        
        ax.set_ylabel(ylabel, fontsize=42, fontweight="bold")
        ax.set_title(title, fontsize=36, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=48)
        ax.tick_params(axis="y", labelsize=28)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=36)
        
    fig.suptitle("Variation 4: Adding Correctness", 
                 fontsize=40, fontweight="bold", y=1.01)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_5(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 5: Three Rows — Full 4-Scenario Comparison
    
    | Row             | 4 Groups (9 bars each)                      |
    |-----------------|---------------------------------------------|
    | Faithfulness    | +WU→F%, −WF→U%, +WF→U%, −WU→F%              |
    | Hint-Mentioning | +WU→Hm%, −WF→Hnm%, +WF→Hnm%, −WU→Hm%        |
    | Correctness     | +I→C%, −I→C%                                |
    """
    # Use GridSpec for different column counts per row
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1])
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    # Row 0: Faithfulness (4 panels)
    faith_panels = [
        (0, "positive_WU", "faithful_pct", "+WU → F%", "F%", False),
        (1, "negative_WF", "unfaithful_pct", "−WF → U%", "U%", False),
        (2, "positive_WF", "unfaithful_pct", "+WF → U%", "U%", False),
        (3, "negative_WU", "faithful_pct", "−WU → F%", "F%", False),
    ]
    
    for col, direction, metric, title, ylabel, is_hnm in faith_panels:
        ax = fig.add_subplot(gs[0, col])
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
            ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.85)
        
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if col == 3:
            ax.legend(loc="upper right", fontsize=9)
    
    # Row 1: Hint-Mentioning (4 panels)
    hint_panels = [
        (0, "positive_WU", "hint_mentioning_pct", "+WU → Hm%", "Hm%", False),
        (1, "negative_WF", "hint_mentioning_pct", "−WF → Hnm%", "Hnm%", True),
        (2, "positive_WF", "hint_mentioning_pct", "+WF → Hnm%", "Hnm%", True),
        (3, "negative_WU", "hint_mentioning_pct", "−WU → Hm%", "Hm%", False),
    ]
    
    for col, direction, metric, title, ylabel, is_hnm in hint_panels:
        ax = fig.add_subplot(gs[1, col])
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    metrics = getattr(approach_data.best_result, direction, None)
                    value = getattr(metrics, metric, 0) if metrics else 0
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.85)
        
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if col == 3:
            ax.legend(loc="upper right", fontsize=9)
    
    # Row 2: Correctness (2 panels, span 2 cols each)
    correct_panels = [
        (slice(0, 2), "positive_WU", "correct_pct", "+Steer I → C%", "Correct %"),
        (slice(2, 4), "negative_WU", "correct_pct", "−Steer I → C%", "Correct %"),
    ]
    
    for col_slice, direction, metric, title, ylabel in correct_panels:
        ax = fig.add_subplot(gs[2, col_slice])
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
            ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.85)
        
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=10)
    
    fig.suptitle("Variation 5: Full 4-Scenario Comparison", 
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_6(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 6: Four Rows — Grouped by Desirability
    
    | Row | Left (9 bars)  | Right (9 bars) | Desirability    |
    |-----|----------------|----------------|-----------------|
    | 1   | +WU→F%         | −WF→U%         | Higher = better |
    | 2   | +WU→Hm%        | −WF→Hnm%       | Higher = better |
    | 3   | +WF→U%         | −WU→F%         | Lower = better  |
    | 4   | +WF→Hnm%       | −WU→Hm%        | Lower = better  |
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    # Panel specs: row, col, direction, metric, title, ylabel, is_hnm
    panels = [
        # Row 0: Higher = better (Faithfulness intended)
        (0, 0, "positive_WU", "faithful_pct", "+WU → F% ↑", "F%", False),
        (0, 1, "negative_WF", "unfaithful_pct", "−WF → U% ↑", "U%", False),
        # Row 1: Higher = better (Hint-Mentioning intended)
        (1, 0, "positive_WU", "hint_mentioning_pct", "+WU → Hm% ↑", "Hm%", False),
        (1, 1, "negative_WF", "hint_mentioning_pct", "−WF → Hnm% ↑", "Hnm%", True),
        # Row 2: Lower = better (Faithfulness collateral)
        (2, 0, "positive_WF", "unfaithful_pct", "+WF → U% ↓", "U%", False),
        (2, 1, "negative_WU", "faithful_pct", "−WU → F% ↓", "F%", False),
        # Row 3: Lower = better (Hint-Mentioning collateral)
        (3, 0, "positive_WF", "hint_mentioning_pct", "+WF → Hnm% ↓", "Hnm%", True),
        (3, 1, "negative_WU", "hint_mentioning_pct", "−WU → Hm% ↓", "Hm%", False),
    ]
    
    for row, col, direction, metric, title, ylabel, is_hnm in panels:
        ax = axes[row, col]
        
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    metrics = getattr(approach_data.best_result, direction, None)
                    value = getattr(metrics, metric, 0) if metrics else 0
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.85)
        
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if col == 1:
            ax.legend(loc="upper right", fontsize=10)
    
    # Row labels
    fig.text(0.02, 0.88, "↑ Higher = Better", fontsize=12, fontweight="bold", rotation=90, va="center", color="#27ae60")
    fig.text(0.02, 0.62, "(Intended)", fontsize=10, rotation=90, va="center", color="#27ae60")
    fig.text(0.02, 0.38, "↓ Lower = Better", fontsize=12, fontweight="bold", rotation=90, va="center", color="#c0392b")
    fig.text(0.02, 0.12, "(Collateral)", fontsize=10, rotation=90, va="center", color="#c0392b")
    
    fig.suptitle("Variation 6: Grouped by Desirability", 
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Variation 1 plot")
    parser.add_argument("--base-dir", type=Path, 
                        default=Path(r"c:\Users\occhi\Desktop\unfaithfulness_steering"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hint-template", type=str, default="grader_hacking")
    args = parser.parse_args()
    
    output_dir = args.output_dir or (args.base_dir / "plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load real data for all models
    qwen_data = load_qwen3_32b(args.base_dir)
    qwen14b_data = load_qwen3_14b(args.base_dir)
    deepseek_data = load_deepseek_r1_distill_llama_8b(args.base_dir)
    
    all_data = {
        "Qwen3-32B": qwen_data,
        "DeepSeek-R1-Distill-Llama-8B": deepseek_data,
        "Qwen3-14B": qwen14b_data
    }
    
    # Print best configs and save to file
    config_file_path = output_dir / "best_configurations.txt"
    print(f"\n=== Saving Best Configurations to {config_file_path} ===")
    
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write("=== Best Steering Configurations ===\n\n")
        
        for model_name, model_data in all_data.items():
            f.write(f"Model: {model_name}\n")
            print(f"\nModel: {model_name}")
            
            for approach in ["off_policy", "linear", "mlp"]:
                ad = getattr(model_data, approach)
                if ad.best_config:
                    layer, coeff = ad.best_config
                    r = ad.best_result
                    
                    # Handle potential list/array for coeff
                    if isinstance(coeff, list):
                        coeff = coeff[0]
                        
                    line = f"  {approach.upper()}: layer={layer}, coeff={coeff}\n"
                    line += f"    +WU→F%: {r.positive_WU.faithful_pct:.1f}%\n"
                    line += f"    −WF→U%: {r.negative_WF.unfaithful_pct:.1f}%\n"
                    line += f"    +WU→Hm%: {r.positive_WU.hint_mentioning_pct:.1f}%\n"
                    
                    f.write(line)
                    print(line.strip())
            f.write("\n")
    
    # Generate plots
    print("\n=== Generating Plots ===")
    plot_variation_1(all_data, output_dir / "variation_1.png")
    plot_variation_2(all_data, output_dir / "variation_2.png")
    plot_variation_3a(all_data, output_dir / "variation_3a.png")
    plot_variation_3b(all_data, output_dir / "variation_3b.png")
    plot_variation_4(all_data, output_dir / "variation_4.png")
    plot_variation_5(all_data, output_dir / "variation_5.png")
    plot_variation_6(all_data, output_dir / "variation_6.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
