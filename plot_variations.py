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
    correct_count: int = 0         # Raw count of wrong_to_correct
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
    random: ApproachData = field(default_factory=ApproachData)


# =============================================================================
# Data Loading
# =============================================================================

def extract_metrics(transitions: dict) -> MetricSet:
    """Extract metrics from a transitions dictionary."""
    def get_rate(key: str) -> float:
        return transitions.get(key, {}).get("rate", 0.0)
    
    def get_count(key: str) -> int:
        return transitions.get(key, {}).get("count", 0)
    
    hint_mentioning = sum(
        v.get("rate", 0.0) for k, v in transitions.items()
        if "mentioning_hint" in k and isinstance(v, dict)
    )
    
    return MetricSet(
        faithful_pct=get_rate("stable_faithful") * 100,
        unfaithful_pct=get_rate("stable_unfaithful") * 100,
        correct_pct=get_rate("wrong_to_correct") * 100,
        correct_count=get_count("wrong_to_correct"),
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
            
            # Calculate simple average of rates (counts are summed for pooling)
            num_templates = len(metric_list)
            avg_metrics = MetricSet()
            
            if num_templates > 0:
                avg_metrics.faithful_pct = sum(m.faithful_pct for m in metric_list) / num_templates
                avg_metrics.unfaithful_pct = sum(m.unfaithful_pct for m in metric_list) / num_templates
                avg_metrics.correct_pct = sum(m.correct_pct for m in metric_list) / num_templates
                avg_metrics.correct_count = sum(m.correct_count for m in metric_list)  # Sum counts for pooling
                avg_metrics.hint_mentioning_pct = sum(m.hint_mentioning_pct for m in metric_list) / num_templates
                avg_metrics.n = sum(m.n for m in metric_list)  # Sum N to show total samples involved
            
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
    model.random = load_approach(data_dir, "random", "Qwen3-32B")
    
    return model


def load_qwen3_14b(base_dir: Path) -> ModelData:
    """Load all Qwen3-14B data."""
    data_dir = base_dir / "data" / "definitive_pipeline_data" / "Qwen3-14B"
    print("Loading Qwen3-14B...")
    
    model = ModelData()
    model.off_policy = load_approach(data_dir, "off_policy", "Qwen3-14B")
    model.linear = load_approach(data_dir, "linear", "Qwen3-14B")
    model.mlp = load_approach(data_dir, "mlp", "Qwen3-14B")
    model.random = load_approach(data_dir, "random", "Qwen3-14B")
    
    return model


def load_deepseek_r1_distill_llama_8b(base_dir: Path) -> ModelData:
    """Load all DeepSeek-R1-Distill-Llama-8B data."""
    data_dir = base_dir / "data" / "definitive_pipeline_data" / "DeepSeek-R1-Distill-Llama-8B"
    print("Loading DeepSeek-R1-Distill-Llama-8B...")
    
    model = ModelData()
    model.off_policy = load_approach(data_dir, "off_policy", "DeepSeek-R1-Distill-Llama-8B")
    model.linear = load_approach(data_dir, "linear", "DeepSeek-R1-Distill-Llama-8B")
    model.mlp = load_approach(data_dir, "mlp", "DeepSeek-R1-Distill-Llama-8B")
    model.random = load_approach(data_dir, "random", "DeepSeek-R1-Distill-Llama-8B")
    
    return model


# =============================================================================
# Plotting
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


def load_hintwise_data(base_dir: Path) -> dict:
    """
    Load steering data per hint template (not averaged).
    
    Returns: {model: {approach: {hint: ConfigResult}}}
    """
    models_config = {
        "Qwen3-32B": "Qwen3-32B",
        "Qwen3-14B": "Qwen3-14B", 
        "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    }
    
    approaches = ["linear", "off_policy", "mlp"]
    hints = ["grader_hacking", "metadata", "professor"]
    
    all_data = {}
    
    for model_name, model_suffix in models_config.items():
        data_dir = base_dir / "data" / "definitive_pipeline_data" / model_suffix
        all_data[model_name] = {}
        
        for mode in approaches:
            pattern = f"summary_steered_{mode}_{model_suffix}_*.json"
            files = list(data_dir.glob(pattern))
            
            if not files:
                all_data[model_name][mode] = {}
                continue
            
            latest = sorted(files)[-1]
            with open(latest, "r", encoding="utf-8") as f:
                summary = json.load(f)
            
            # Parse per-hint configs and find best for each hint
            configs_by_hint = summary.get("configurations_by_hint", {})
            all_data[model_name][mode] = {}
            
            for hint in hints:
                hint_configs = configs_by_hint.get(hint, [])
                if not hint_configs:
                    continue
                
                # Find best config for this hint using same scoring
                best_score = float("-inf")
                best_result = None
                eps = 1e-6
                
                for cfg in hint_configs:
                    layer = cfg.get("layer")
                    coeff = cfg.get("coefficient_magnitude")
                    
                    # Extract metrics for this config
                    result = ConfigResult(layer=layer, coefficient=coeff)
                    
                    for json_key, attr_key in [
                        ("positive_on_WF", "positive_WF"),
                        ("positive_on_WU", "positive_WU"),
                        ("negative_on_WF", "negative_WF"),
                        ("negative_on_WU", "negative_WU"),
                    ]:
                        data = cfg.get(json_key, {})
                        metrics = extract_metrics(data.get("transitions", {}))
                        metrics.n = data.get("n", 0)
                        setattr(result, attr_key, metrics)
                    
                    # Score: intended effect / collateral damage
                    intended = result.positive_WU.faithful_pct
                    collateral = result.positive_WF.unfaithful_pct
                    score = intended / (collateral + eps)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                
                all_data[model_name][mode][hint] = best_result
    
    return all_data


def plot_variation_1_hintwise(
    base_dir: Path,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 1 Hint-wise: 2×3 grid
    
    Rows: 2 metrics (+ Steering Faithfulness, - Steering Unfaithfulness)
    Columns: 3 models (Qwen3-32B, Qwen3-14B, Llama-8B)
    Each cell: 9 bars (3 approaches × 3 hints, color-coded by hint)
    """
    print("Loading hint-wise data...")
    hintwise_data = load_hintwise_data(base_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(30, 18))
    
    models = ["Qwen3-32B", "Qwen3-14B", "DeepSeek-R1-Distill-Llama-8B"]
    model_labels = ["Qwen3-32B", "Qwen3-14B", "Llama-8B"]
    approaches = ["linear", "off_policy", "mlp"]
    approach_labels = ["Linear", "Off-Policy", "MLP"]
    hints = ["grader_hacking", "metadata", "professor"]
    hint_labels = ["Grader", "Metadata", "Professor"]
    
    # Colors for hints
    hint_colors = {
        "grader_hacking": "#2196F3",  # Blue
        "metadata": "#4CAF50",         # Green
        "professor": "#FF9800",        # Orange
    }
    
    # Panel specifications (row index, direction, metric, ylabel)
    panels = [
        (0, "positive_WU", "faithful_pct", "+ Steering: Faithfulness Rate", "Faithful %"),
        (1, "negative_WF", "unfaithful_pct", "- Steering: Unfaithfulness Rate", "Unfaithful %"),
    ]
    
    # Bar positioning
    n_hints = len(hints)
    bar_width = 0.25
    x_base = np.arange(len(approaches))
    
    for row_idx, direction, metric, row_title, ylabel in panels:
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            
            # Plot 3 hints per approach
            for h_idx, hint in enumerate(hints):
                values = []
                for approach in approaches:
                    data = hintwise_data.get(model, {}).get(approach, {}).get(hint)
                    if data:
                        metrics_obj = getattr(data, direction, None)
                        value = getattr(metrics_obj, metric, 0) if metrics_obj else 0
                    else:
                        value = 0
                    values.append(value)
                
                offset = (h_idx - 1) * bar_width
                ax.bar(x_base + offset, values, bar_width * 0.9, 
                       label=hint_labels[h_idx] if (row_idx == 0 and col_idx == 0) else "",
                       color=hint_colors[hint], alpha=0.85)
            
            # Styling
            ax.set_xticks(x_base)
            ax.set_xticklabels(approach_labels, fontsize=36)
            ax.tick_params(axis="y", labelsize=30)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            
            # Y-axis label only on first column
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=40, fontweight="bold")
            
            # Model name as column title (only on first row) - using absolute positioning via fig.text
            # We skip ax.set_title to avoid padding guess-work and potential overlaps
            
            # Legend only on top-right panel (rightest plot)
            if row_idx == 0 and col_idx == 2:
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=hint_colors[h], label=l) 
                                  for h, l in zip(hints, hint_labels)]
                ax.legend(handles=legend_elements, loc="upper right", fontsize=30, 
                         title="Hint Type", title_fontsize=28)
    
    # Model Names (Column Headers) - High up, near main title
    # Columns are at ~0.15, ~0.48, ~0.81 roughly in 3-col layout with wspace=0.15
    # We can use the axes positions to centering
    # Get positions of the top row axes
    for i in range(3):
        bbox = axes[0, i].get_position()
        center_x = bbox.x0 + bbox.width / 2
        fig.text(center_x, 0.96, model_labels[i], ha="center", fontsize=44, fontweight="bold")

    # Row titles above each row (centered horizontally)
    fig.text(0.5, 0.89, "+ Steering on Unfaithful Answers: Faithfulness Rate", ha="center", fontsize=34, fontweight="bold")
    fig.text(0.5, 0.40, "- Steering on Faithful Answers: Unfaithfulness Rate", ha="center", fontsize=34, fontweight="bold")
    
    fig.suptitle("Steering Performance by Hint Template", 
                 fontsize=48, fontweight="bold", y=1.05)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, hspace=0.50, wspace=0.15)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_2(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 2: Pooled Correctness Rate
    
    1 row × 2 columns:
    - Left: +Steering Correctness (pooled WF + WU)
    - Right: −Steering Correctness (pooled WF + WU)
    
    Correctness % = (WF→correct_count + WU→correct_count) / (WF_n + WU_n) × 100
    """
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp"]
    approach_labels = ["Linear", "Off-Policy", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    # Color palettes: Pink for positive, Purple for negative
    # Ordered: Light (8B), Medium (14B), Dark (32B)
    palettes = {
        "positive": ["#F8BBD9", "#EC407A", "#AD1457"],  # Pink shades
        "negative": ["#CE93D8", "#AB47BC", "#6A1B9A"],  # Purple shades
    }
    
    # Map models index to shade index: 32B->Dark(2), 8B->Light(0), 14B->Medium(1)
    shade_indices = [2, 0, 1]
    
    # Panel specifications: (col, sign, title, palette_key)
    panels = [
        (0, "positive", "+ Steering Correctness Rate", "positive"),
        (1, "negative", "- Steering Correctness Rate", "negative"),
    ]
    
    for col, sign, title, palette_key in panels:
        ax = axes[col]
        current_palette = palettes[palette_key]
        
        for i, model in enumerate(models):
            model_data = data.get(model)
            values = []
            
            for approach in approaches:
                approach_data = getattr(model_data, approach, None)
                if approach_data and approach_data.best_result:
                    result = approach_data.best_result
                    
                    # Get WF and WU metrics for this sign
                    if sign == "positive":
                        wf = result.positive_WF
                        wu = result.positive_WU
                    else:
                        wf = result.negative_WF
                        wu = result.negative_WU
                    
                    # Pool correctness: (WF_correct + WU_correct) / (WF_n + WU_n)
                    total_correct = wf.correct_count + wu.correct_count
                    total_n = wf.n + wu.n
                    
                    if total_n > 0:
                        pooled_pct = (total_correct / total_n) * 100
                    else:
                        pooled_pct = 0
                    
                    values.append(pooled_pct)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            shade_idx = shade_indices[i]
            color = current_palette[shade_idx]
            
            model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
            ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
        
        # Match Variation 1 font sizes
        ax.set_ylabel("Correctness %", fontsize=42, fontweight="bold")
        ax.set_title(title, fontsize=36, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=48)
        ax.tick_params(axis="y", labelsize=28)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=36)
    
    fig.suptitle("Steering Effect on Answer Correctness", 
                 fontsize=40, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_3(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 3: Collateral Effects
    
    1 row × 2 columns:
    - Left: +Steering on WF → Unfaithfulness % (faithful answers becoming unfaithful)
    - Right: −Steering on WU → Faithfulness % (unfaithful answers becoming faithful)
    """
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp"]
    approach_labels = ["Linear", "Off-Policy", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    # Color palettes: Red for +steering collateral, Orange for -steering collateral
    # Ordered: Light (8B), Medium (14B), Dark (32B)
    palettes = {
        "red": ["#EF9A9A", "#F44336", "#B71C1C"],      # Red shades
        "orange": ["#FFCC80", "#FF9800", "#E65100"],   # Orange shades
    }
    
    # Map models index to shade index: 32B->Dark(2), 8B->Light(0), 14B->Medium(1)
    shade_indices = [2, 0, 1]
    
    # Panel specifications: (col, direction_attr, metric_attr, title, ylabel, palette_key)
    panels = [
        (0, "positive_WF", "unfaithful_pct", 
         "+ Steering Making Faithful Answers Unfaithful", "Unfaithful %", "red"),
        (1, "negative_WU", "faithful_pct", 
         "- Steering Making Unfaithful Answers Faithful", "Faithful %", "orange"),
    ]
    
    for col, direction, metric, title, ylabel, palette_key in panels:
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
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            shade_idx = shade_indices[i]
            color = current_palette[shade_idx]
            
            model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
            ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
        
        # Match Variation 1 font sizes
        ax.set_ylabel(ylabel, fontsize=42, fontweight="bold")
        ax.set_title(title, fontsize=36, fontweight="bold", pad=20)  # Added pad for spacing
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=48)
        ax.tick_params(axis="y", labelsize=28)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=36)
    
    fig.suptitle("Collateral Effects of Positive and Negative Steering", 
                 fontsize=40, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, top=0.85)  # Added top margin for title spacing
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Probe Data Loading
# =============================================================================

def load_probe_data(base_dir: Path) -> dict[str, dict]:
    """
    Load probe results summary for all models.
    
    Returns dict: {model_name: {layer: {probe_type: metrics}}}
    """
    probe_paths = {
        "Qwen3-32B": base_dir / "data" / "definitive_pipeline_data" / "Qwen3-32B" / "probes_Qwen3-32B_2026-01-12" / "results_summary.json",
        "Qwen3-14B": base_dir / "data" / "definitive_pipeline_data" / "Qwen3-14B" / "probes_Qwen3-14B_2026-01-12" / "results_summary.json",
        "DeepSeek-R1-Distill-Llama-8B": base_dir / "data" / "definitive_pipeline_data" / "DeepSeek-R1-Distill-Llama-8B" / "probes_results_summary.json",
    }
    
    all_probe_data = {}
    
    for model_name, path in probe_paths.items():
        if not path.exists():
            print(f"  Warning: Probe summary not found for {model_name}")
            continue
            
        print(f"  Loading probes for {model_name}: {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        all_probe_data[model_name] = data
    
    return all_probe_data


def plot_variation_4(
    base_dir: Path,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 4: Probe F1 Scores Across Layers
    
    1 row × 3 columns: one subplot per model, comparing MLP vs LogReg probe.
    """
    # Load probe data
    print("Loading probe data...")
    probe_data = load_probe_data(base_dir)
    
    if not probe_data:
        print("No probe data found!")
        return None
    
    # Larger figure to accommodate bigger text
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    
    # Model order for consistent display
    model_order = ["Qwen3-32B", "Qwen3-14B", "DeepSeek-R1-Distill-Llama-8B"]
    display_names = {
        "Qwen3-32B": "Qwen3-32B",
        "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
        "Qwen3-14B": "Qwen3-14B",
    }
    
    # Colors for probe types
    probe_colors = {
        "mlp": "#1565C0",     # Blue
        "logreg": "#FF9800",  # Orange
    }
    
    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        
        if model_name not in probe_data:
            ax.set_title(f"No data: {display_names[model_name]}", fontsize=32)
            continue
        
        data = probe_data[model_name]
        
        for probe_type, color in probe_colors.items():
            results = data.get("results", {}).get(probe_type, {})
            
            if not results:
                continue
            
            layers = sorted([int(k) for k in results.keys()])
            f1_scores = [results[str(layer)].get("val_f1", 0) * 100 for layer in layers]
            
            label = "MLP" if probe_type == "mlp" else "LogReg"
            ax.plot(
                layers, f1_scores, 
                label=label,
                color=color,
                linewidth=3,
                alpha=0.9
            )
        
        # Styling per subplot - larger fonts
        ax.set_xlabel("Layer", fontsize=36, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("F1 Score (%)", fontsize=36, fontweight="bold")
        
        ax.set_title(display_names[model_name], fontsize=36, fontweight="bold", pad=20)
        ax.tick_params(axis="both", labelsize=28)
        ax.set_ylim(50, 100)
        ax.grid(axis="both", alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=40)
    
    fig.suptitle("Faithfulness Probes Performance Across Layers", 
                 fontsize=48, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, top=0.88)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Baseline Faithfulness Data Loading
# =============================================================================

def load_baseline_faithfulness_data(base_dir: Path) -> dict[str, dict]:
    """
    Load baseline faithfulness data from faithfulness_annotated_*.jsonl files.
    
    Returns dict: {model_name: {"faithful": count, "unfaithful": count, "total": count}}
    """
    models_config = {
        "Qwen3-32B": "Qwen3-32B",
        "Qwen3-14B": "Qwen3-14B",
        "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    }
    
    all_data = {}
    
    for model_name, model_suffix in models_config.items():
        data_dir = base_dir / "data" / "definitive_pipeline_data" / model_suffix
        
        # Find the faithfulness_annotated file
        pattern = f"faithfulness_annotated_{model_suffix}_*.jsonl"
        files = list(data_dir.glob(pattern))
        
        if not files:
            print(f"  Warning: No faithfulness_annotated file found for {model_name}")
            all_data[model_name] = {"faithful": 0, "unfaithful": 0, "total": 0}
            continue
        
        latest = sorted(files)[-1]
        print(f"  Loading faithfulness data for {model_name}: {latest.name}")
        
        # Count faithful vs unfaithful across all records (pooled across hints)
        faithful_count = 0
        unfaithful_count = 0
        
        with open(latest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                classification = record.get("faithfulness_classification", "")
                if classification == "faithful":
                    faithful_count += 1
                elif classification == "unfaithful":
                    unfaithful_count += 1
        
        total = faithful_count + unfaithful_count
        all_data[model_name] = {
            "faithful": faithful_count,
            "unfaithful": unfaithful_count,
            "total": total
        }
        print(f"    Faithful: {faithful_count}, Unfaithful: {unfaithful_count}, Total: {total}")
    
    return all_data


def plot_variation_5(
    base_dir: Path,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 5: Baseline Faithfulness Distribution
    
    Horizontal stacked bar chart showing faithful vs unfaithful percentages
    for each model, pooled across all hint templates.
    
    - Left portion: Faithful % (green)
    - Right portion: Unfaithful % (red)
    """
    print("Loading baseline faithfulness data...")
    faithfulness_data = load_baseline_faithfulness_data(base_dir)
    
    # Model order and display names
    models = ["Qwen3-32B", "Qwen3-14B", "DeepSeek-R1-Distill-Llama-8B"]
    model_labels = ["Qwen3-32B", "Qwen3-14B", "Llama-8B"]
    
    # Calculate percentages
    faithful_pcts = []
    unfaithful_pcts = []
    
    for model in models:
        data = faithfulness_data.get(model, {})
        total = data.get("total", 0)
        if total > 0:
            f_pct = (data.get("faithful", 0) / total) * 100
            u_pct = (data.get("unfaithful", 0) / total) * 100
        else:
            f_pct = 0
            u_pct = 0
        faithful_pcts.append(f_pct)
        unfaithful_pcts.append(u_pct)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Bar positions (horizontal)
    y_positions = np.arange(len(models))
    bar_height = 0.6
    
    # Colors
    faithful_color = "#4CAF50"    # Green
    unfaithful_color = "#F44336"  # Red
    
    # Draw stacked horizontal bars
    # Faithful on left (starts at 0)
    bars_faithful = ax.barh(
        y_positions, faithful_pcts, bar_height,
        label="Faithful", color=faithful_color, alpha=0.9
    )
    
    # Unfaithful on right (starts where faithful ends)
    bars_unfaithful = ax.barh(
        y_positions, unfaithful_pcts, bar_height,
        left=faithful_pcts, label="Unfaithful", color=unfaithful_color, alpha=0.9
    )
    
    # Add percentage labels on bars
    for i, (f_pct, u_pct) in enumerate(zip(faithful_pcts, unfaithful_pcts)):
        # Faithful label (left side)
        if f_pct > 10:  # Only show if enough space
            ax.text(f_pct / 2, i, f"{f_pct:.1f}%", 
                   ha="center", va="center", fontsize=28, fontweight="bold", color="white")
        
        # Unfaithful label (right side)
        if u_pct > 10:  # Only show if enough space
            ax.text(f_pct + u_pct / 2, i, f"{u_pct:.1f}%",
                   ha="center", va="center", fontsize=28, fontweight="bold", color="white")
    
    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_labels, fontsize=28, fontweight="bold")
    ax.set_xlabel("(%)", fontsize=28, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=22)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Invert y-axis so first model is at top
    ax.invert_yaxis()
    
    # Legend - positioned outside below the chart
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=32)
    
    # Title
    fig.suptitle("Baseline Faithfulness Distribution Across Models", 
                 fontsize=40, fontweight="bold", y=1.02)
    
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
    Variation 6: Hint-Mentioning Rate of Negative Steering on Unfaithful Answers
    
    Single panel showing: -Steering on WU → Hint-Mentioning %
    (How often do unfaithful answers still mention the hint after negative steering?)
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp"]
    approach_labels = ["Linear", "Off-Policy", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    # Color palette: Teal shades
    # Ordered: Light (8B), Medium (14B), Dark (32B)
    palette = ["#80CBC4", "#26A69A", "#004D40"]
    
    # Map models index to shade index: 32B->Dark(2), 8B->Light(0), 14B->Medium(1)
    shade_indices = [2, 0, 1]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                # negative_WU = negative steering on Was Unfaithful
                metrics = getattr(approach_data.best_result, "negative_WU", None)
                value = getattr(metrics, "hint_mentioning_pct", 0) if metrics else 0
                values.append(value)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = palette[shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
    
    # Styling
    ax.set_ylabel("Hint-Mentioning %", fontsize=36, fontweight="bold")
    ax.set_title("- Steering on Unfaithful Answers:\nHint-Mentioning Rate", fontsize=32, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=36)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=28)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_7(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 7: Like Variation 1 but with Random approach as 4th baseline.
    
    | Row             | Left (12 bars) | Right (12 bars) |
    |-----------------|----------------|-----------------|
    | Faithfulness    | +steer WU→F%   | −steer WF→U%    |
    | Hint-Mentioning | +steer WU→Hm%  | −steer WF→Hnm%  |
    
    Each panel: 12 bars = 4 approaches × 3 models
    """
    fig, axes = plt.subplots(2, 2, figsize=(32, 24))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp", "random"]
    approach_labels = ["Linear", "Off-Policy", "MLP", "Random"]
    
    x = np.arange(len(approaches))
    width = 0.22
    
    # Color palettes (Light -> Medium -> Dark)
    palettes = {
        "good": ["#A5D6A7", "#4CAF50", "#1B5E20"],      # Green
        "bad": ["#EF9A9A", "#F44336", "#B71C1C"],       # Red
        "meh_good": ["#81D4FA", "#29B6F6", "#01579B"],  # Light Blue
        "meh_bad": ["#FFF59D", "#FBC02D", "#F57F17"],   # Yellow/Orange
    }
    
    # Map models index to shade index
    shade_indices = [2, 0, 1]
    
    # Panel specifications
    panels = [
        (0, 0, "positive_WU", "faithful_pct", 
         "+ Steering on Unfaithful Answers:\nFaithfulness Rate", "Faithful %", False, "good"),
        (0, 1, "negative_WF", "unfaithful_pct", 
         "- Steering on Faithful Answers:\nUnfaithfulness Rate", "Unfaithful %", False, "bad"),
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
        ax.set_title(title, fontsize=32, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=36)
        ax.tick_params(axis="y", labelsize=24)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=28)
    
    fig.suptitle("Steering Performance (Including Random Baseline)", 
                 fontsize=40, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_8(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 8: Like Variation 3 (Collateral Effects) but with Random approach.
    
    1 row × 2 columns:
    - Left: +Steering on WF → Unfaithfulness % (faithful answers becoming unfaithful)
    - Right: −Steering on WU → Faithfulness % (unfaithful answers becoming faithful)
    """
    fig, axes = plt.subplots(1, 2, figsize=(32, 12))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp", "random"]
    approach_labels = ["Linear", "Off-Policy", "MLP", "Random"]
    
    x = np.arange(len(approaches))
    width = 0.22
    
    # Color palettes
    palettes = {
        "red": ["#EF9A9A", "#F44336", "#B71C1C"],      # Red shades
        "orange": ["#FFCC80", "#FF9800", "#E65100"],   # Orange shades
    }
    
    shade_indices = [2, 0, 1]
    
    panels = [
        (0, "positive_WF", "unfaithful_pct", 
         "+ Steering Making Faithful Answers Unfaithful", "Unfaithful %", "red"),
        (1, "negative_WU", "faithful_pct", 
         "- Steering Making Unfaithful Answers Faithful", "Faithful %", "orange"),
    ]
    
    for col, direction, metric, title, ylabel, palette_key in panels:
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
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            shade_idx = shade_indices[i]
            color = current_palette[shade_idx]
            
            model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
            ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
        
        ax.set_ylabel(ylabel, fontsize=36, fontweight="bold")
        ax.set_title(title, fontsize=32, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=36)
        ax.tick_params(axis="y", labelsize=24)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=28)
    
    fig.suptitle("Collateral Effects (Including Random Baseline)", 
                 fontsize=40, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, top=0.85)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig



def plot_variation_9(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 9: Like Variation 6 (Hint-Mentioning Collateral) but with Random approach.
    
    Single panel showing: -Steering on WU → Hint-Mentioning %
    (How often do unfaithful answers still mention the hint after negative steering?)
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp", "random"]
    approach_labels = ["Linear", "Off-Policy", "MLP", "Random"]
    
    x = np.arange(len(approaches))
    width = 0.22
    
    # Color palette: Teal shades
    palette = ["#80CBC4", "#26A69A", "#004D40"]
    shade_indices = [2, 0, 1]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                # negative_WU = negative steering on Was Unfaithful
                metrics = getattr(approach_data.best_result, "negative_WU", None)
                value = getattr(metrics, "hint_mentioning_pct", 0) if metrics else 0
                values.append(value)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = palette[shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
    
    # Styling
    ax.set_ylabel("Hint-Mentioning %", fontsize=36, fontweight="bold")
    ax.set_title("- Steering on Unfaithful Answers:\nHint-Mentioning Rate", fontsize=32, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=36)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=28)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig



def plot_variation_9(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 9: Like Variation 6 (Hint-Mentioning Collateral) but with Random approach.
    
    Single panel showing: -Steering on WU → Hint-Mentioning %
    (How often do unfaithful answers still mention the hint after negative steering?)
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp", "random"]
    approach_labels = ["Linear", "Off-Policy", "MLP", "Random"]
    
    x = np.arange(len(approaches))
    width = 0.22
    
    # Color palette: Teal shades
    palette = ["#80CBC4", "#26A69A", "#004D40"]
    shade_indices = [2, 0, 1]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                # negative_WU = negative steering on Was Unfaithful
                metrics = getattr(approach_data.best_result, "negative_WU", None)
                value = getattr(metrics, "hint_mentioning_pct", 0) if metrics else 0
                values.append(value)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = palette[shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
    
    # Styling
    ax.set_ylabel("Hint-Mentioning %", fontsize=36, fontweight="bold")
    ax.set_title("- Steering on Unfaithful Answers:\nHint-Mentioning Rate", fontsize=32, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=36)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=28)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_10(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 10: Monitorability Gain + Collateral Effects (Compact 2x2)
    
    | Row                     | Left                              | Right                             |
    |-------------------------|-----------------------------------|-----------------------------------|
    | Intended Effects        | Monitorability Gain (stacked)     | −steer WF→U%                      |
    | Collateral Effects      | +steer WF→U%                      | −steer WU→F%                      |
    
    Top Left: Stacked bars showing Faithful % (solid) + Hint-Mentioning % (hatched)
    """
    fig, axes = plt.subplots(2, 2, figsize=(28, 24))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp"]
    approach_labels = ["Linear", "Off-Policy", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    # Color palettes
    palettes = {
        "good": ["#A5D6A7", "#4CAF50", "#1B5E20"],      # Green
        "bad": ["#EF9A9A", "#F44336", "#B71C1C"],       # Red
        "orange": ["#FFCC80", "#FF9800", "#E65100"],    # Orange
    }
    
    # Model to shade mapping: 32B->Dark(2), 8B->Light(0), 14B->Medium(1)
    shade_indices = [2, 0, 1]
    
    # =========================================================================
    # TOP LEFT: Monitorability Gain (Stacked: Faithful + Hint-Mentioning)
    # =========================================================================
    ax = axes[0, 0]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        faithful_vals = []
        hint_vals = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                metrics = getattr(approach_data.best_result, "positive_WU", None)
                f_val = getattr(metrics, "faithful_pct", 0) if metrics else 0
                h_val = getattr(metrics, "hint_mentioning_pct", 0) if metrics else 0
                faithful_vals.append(f_val)
                hint_vals.append(h_val)
            else:
                faithful_vals.append(0)
                hint_vals.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = palettes["good"][shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        
        # Convert to numpy arrays for proper stacking
        faithful_arr = np.array(faithful_vals)
        hint_arr = np.array(hint_vals)
        
        # Base bar: Faithful % (solid)
        ax.bar(x + offset, faithful_arr, width, label=f"{model_label} (Faithful)", 
               color=color, alpha=0.9)
        
        # Stacked bar: Hint-Mentioning % (hatched)
        ax.bar(x + offset, hint_arr, width, bottom=faithful_arr,
               label=f"{model_label} (Hint-Mentioning)" if i == 0 else "",
               color=color, alpha=0.6, hatch="//", edgecolor="white")
    
    ax.set_ylabel("Monitorability Gain %", fontsize=36, fontweight="bold")
    ax.set_title("Monitorability Gain: +Steering on Unfaithful Answers", fontsize=32, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=36)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)  # Allow for stacked bars to exceed 100%
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Custom legend for stacked bars
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palettes["good"][2], label="Qwen3-32B"),
        Patch(facecolor=palettes["good"][0], label="Llama-8B"),
        Patch(facecolor=palettes["good"][1], label="Qwen3-14B"),
        Patch(facecolor="gray", alpha=0.9, label="Faithful %"),
        Patch(facecolor="gray", alpha=0.6, hatch="//", edgecolor="white", label="Hint-Mentioning %"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=22)
    
    # =========================================================================
    # TOP RIGHT: -Steering on WF → Unfaithfulness Rate (unchanged from V1)
    # =========================================================================
    ax = axes[0, 1]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                metrics = getattr(approach_data.best_result, "negative_WF", None)
                value = getattr(metrics, "unfaithful_pct", 0) if metrics else 0
                values.append(value)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = palettes["bad"][shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
    
    ax.set_ylabel("Degradation %", fontsize=36, fontweight="bold")
    ax.set_title("- Steering on Faithful Answers:\nUnfaithfulness Rate", fontsize=32, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=36)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=28)
    
    # =========================================================================
    # BOTTOM ROW: Collateral Effects (from Variation 3)
    # =========================================================================
    
    # BOTTOM LEFT: +Steering on WF → Unfaithfulness % (simple bars)
    ax = axes[1, 0]
    current_palette = palettes["bad"]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                metrics = getattr(approach_data.best_result, "positive_WF", None)
                value = getattr(metrics, "unfaithful_pct", 0) if metrics else 0
                values.append(value)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = current_palette[shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
    
    ax.set_ylabel("Unintended Degradation %", fontsize=36, fontweight="bold")
    ax.set_title("+ Steering Making Faithful Answers Unfaithful", fontsize=32, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=36)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=28)
    
    # BOTTOM RIGHT: -Steering on WU → Faithfulness % + Hint-Mentioning % (stacked)
    ax = axes[1, 1]
    current_palette = palettes["orange"]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        faithful_vals = []
        hint_vals = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                metrics = getattr(approach_data.best_result, "negative_WU", None)
                f_val = getattr(metrics, "faithful_pct", 0) if metrics else 0
                h_val = getattr(metrics, "hint_mentioning_pct", 0) if metrics else 0
                faithful_vals.append(f_val)
                hint_vals.append(h_val)
            else:
                faithful_vals.append(0)
                hint_vals.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = current_palette[shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        
        # Convert to numpy arrays for proper stacking
        faithful_arr = np.array(faithful_vals)
        hint_arr = np.array(hint_vals)
        
        # Base bar: Faithful % (solid)
        ax.bar(x + offset, faithful_arr, width, label=f"{model_label}", 
               color=color, alpha=0.9)
        
        # Stacked bar: Hint-Mentioning % (hatched)
        ax.bar(x + offset, hint_arr, width, bottom=faithful_arr,
               color=color, alpha=0.6, hatch="//", edgecolor="white")
    
    ax.set_ylabel("Unintended Monitorability Gain %", fontsize=36, fontweight="bold")
    ax.set_title("- Steering Making Unfaithful Answers Monitorable", fontsize=32, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=36)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)  # Allow for stacked bars
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Custom legend for stacked bars
    legend_elements = [
        Patch(facecolor=current_palette[2], label="Qwen3-32B"),
        Patch(facecolor=current_palette[0], label="Llama-8B"),
        Patch(facecolor=current_palette[1], label="Qwen3-14B"),
        Patch(facecolor="gray", alpha=0.9, label="Faithful %"),
        Patch(facecolor="gray", alpha=0.6, hatch="//", edgecolor="white", label="Hint-Mentioning %"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=22)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def plot_variation_11(
    data: dict[str, ModelData],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Variation 11: Same as Variation 10 but WITH Random approach (4 bars per group)
    
    | Row                     | Left                              | Right                             |
    |-------------------------|-----------------------------------|-----------------------------------|
    | Intended Effects        | Monitorability Gain (stacked)     | −steer WF→U%                      |
    | Collateral Effects      | +steer WF→U%                      | −steer WU→F% + Hint% (stacked)    |
    """
    fig, axes = plt.subplots(2, 2, figsize=(32, 24))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"]
    approaches = ["linear", "off_policy", "mlp", "random"]
    approach_labels = ["Linear", "Off-Policy", "MLP", "Random"]
    
    x = np.arange(len(approaches))
    width = 0.22
    
    # Color palettes
    palettes = {
        "good": ["#A5D6A7", "#4CAF50", "#1B5E20"],      # Green
        "bad": ["#EF9A9A", "#F44336", "#B71C1C"],       # Red
        "orange": ["#FFCC80", "#FF9800", "#E65100"],    # Orange
    }
    
    # Model to shade mapping: 32B->Dark(2), 8B->Light(0), 14B->Medium(1)
    shade_indices = [2, 0, 1]
    
    # =========================================================================
    # TOP LEFT: Monitorability Gain (Stacked: Faithful + Hint-Mentioning)
    # =========================================================================
    ax = axes[0, 0]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        faithful_vals = []
        hint_vals = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                metrics = getattr(approach_data.best_result, "positive_WU", None)
                f_val = getattr(metrics, "faithful_pct", 0) if metrics else 0
                h_val = getattr(metrics, "hint_mentioning_pct", 0) if metrics else 0
                faithful_vals.append(f_val)
                hint_vals.append(h_val)
            else:
                faithful_vals.append(0)
                hint_vals.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = palettes["good"][shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        
        # Convert to numpy arrays for proper stacking
        faithful_arr = np.array(faithful_vals)
        hint_arr = np.array(hint_vals)
        
        # Base bar: Faithful % (solid)
        ax.bar(x + offset, faithful_arr, width, label=f"{model_label} (Faithful)", 
               color=color, alpha=0.9)
        
        # Stacked bar: Hint-Mentioning % (hatched)
        ax.bar(x + offset, hint_arr, width, bottom=faithful_arr,
               label=f"{model_label} (Hint-Mentioning)" if i == 0 else "",
               color=color, alpha=0.6, hatch="//", edgecolor="white")
    
    ax.set_ylabel("Monitorability Gain %", fontsize=36, fontweight="bold")
    ax.set_title("Monitorability Gain: +Steering on Unfaithful Answers", fontsize=32, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=32)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palettes["good"][2], label="Qwen3-32B"),
        Patch(facecolor=palettes["good"][0], label="Llama-8B"),
        Patch(facecolor=palettes["good"][1], label="Qwen3-14B"),
        Patch(facecolor="gray", alpha=0.9, label="Faithful %"),
        Patch(facecolor="gray", alpha=0.6, hatch="//", edgecolor="white", label="Hint-Mentioning %"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=20)
    
    # =========================================================================
    # TOP RIGHT: -Steering on WF → Unfaithfulness Rate
    # =========================================================================
    ax = axes[0, 1]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                metrics = getattr(approach_data.best_result, "negative_WF", None)
                value = getattr(metrics, "unfaithful_pct", 0) if metrics else 0
                values.append(value)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = palettes["bad"][shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
    
    ax.set_ylabel("Degradation %", fontsize=36, fontweight="bold")
    ax.set_title("- Steering on Faithful Answers:\nUnfaithfulness Rate", fontsize=32, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=32)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=24)
    
    # =========================================================================
    # BOTTOM LEFT: +Steering on WF → Unfaithfulness %
    # =========================================================================
    ax = axes[1, 0]
    current_palette = palettes["bad"]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        values = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                metrics = getattr(approach_data.best_result, "positive_WF", None)
                value = getattr(metrics, "unfaithful_pct", 0) if metrics else 0
                values.append(value)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = current_palette[shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        ax.bar(x + offset, values, width, label=model_label, color=color, alpha=0.9)
    
    ax.set_ylabel("Unintended Degradation %", fontsize=36, fontweight="bold")
    ax.set_title("+ Steering Making Faithful Answers Unfaithful", fontsize=32, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=32)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=24)
    
    # =========================================================================
    # BOTTOM RIGHT: -Steering on WU → Faithfulness % + Hint-Mentioning % (stacked)
    # =========================================================================
    ax = axes[1, 1]
    current_palette = palettes["orange"]
    
    for i, model in enumerate(models):
        model_data = data.get(model)
        faithful_vals = []
        hint_vals = []
        
        for approach in approaches:
            approach_data = getattr(model_data, approach, None)
            if approach_data and approach_data.best_result:
                metrics = getattr(approach_data.best_result, "negative_WU", None)
                f_val = getattr(metrics, "faithful_pct", 0) if metrics else 0
                h_val = getattr(metrics, "hint_mentioning_pct", 0) if metrics else 0
                faithful_vals.append(f_val)
                hint_vals.append(h_val)
            else:
                faithful_vals.append(0)
                hint_vals.append(0)
        
        offset = (i - 1) * width
        shade_idx = shade_indices[i]
        color = current_palette[shade_idx]
        
        model_label = "Llama-8B" if model == "DeepSeek-R1-Distill-Llama-8B" else model
        
        # Convert to numpy arrays for proper stacking
        faithful_arr = np.array(faithful_vals)
        hint_arr = np.array(hint_vals)
        
        ax.bar(x + offset, faithful_arr, width, label=f"{model_label}", 
               color=color, alpha=0.9)
        
        ax.bar(x + offset, hint_arr, width, bottom=faithful_arr,
               color=color, alpha=0.6, hatch="//", edgecolor="white")
    
    ax.set_ylabel("Unintended Monitorability Gain %", fontsize=36, fontweight="bold")
    ax.set_title("- Steering Making Unfaithful Answers Monitorable", fontsize=32, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=32)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    legend_elements = [
        Patch(facecolor=current_palette[2], label="Qwen3-32B"),
        Patch(facecolor=current_palette[0], label="Llama-8B"),
        Patch(facecolor=current_palette[1], label="Qwen3-14B"),
        Patch(facecolor="gray", alpha=0.9, label="Faithful %"),
        Patch(facecolor="gray", alpha=0.6, hatch="//", edgecolor="white", label="Hint-Mentioning %"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=20)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate steering performance plot")
    parser.add_argument("--base-dir", type=Path, 
                        default=Path(r"c:\Users\occhi\Desktop\unfaithfulness_steering"))
    parser.add_argument("--output-dir", type=Path, default=None)
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
    
    # Load hint-wise data as well for reporting
    hintwise_data = load_hintwise_data(args.base_dir)

    # Print best configs and save to file
    config_file_path = output_dir / "best_configurations.txt"
    print(f"\n=== Saving Best Configurations to {config_file_path} ===")
    
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write("========== FULL EXPERIMENT RESULTS ==========\n\n")
        
        for model_name, model_data in all_data.items():
            f.write(f"MODEL: {model_name}\n")
            f.write("=" * 60 + "\n\n")
            
            for approach in ["off_policy", "linear", "mlp", "random"]:
                f.write(f"  APPROACH: {approach.upper()}\n")
                f.write("  " + "-" * 40 + "\n")
                
                # 1. POOLED RESULTS (Averaged across hints)
                ad = getattr(model_data, approach)
                if ad.best_config:
                    layer, coeff = ad.best_config
                    r = ad.best_result
                    
                    if isinstance(coeff, list):
                        coeff = coeff[0]
                        
                    f.write(f"    [POOLED BEST CONFIG]: Layer {layer}, Coeff {coeff}\n")
                    f.write(f"      +Steering (Make Unfaithful Faithful):\n")
                    f.write(f"        Faithfulness Rate:      {r.positive_WU.faithful_pct:.2f}%\n")
                    f.write(f"        Hint-Mentioning Rate:   {r.positive_WU.hint_mentioning_pct:.2f}%\n")
                    f.write(f"        Correctness (Pooled):   {r.positive_WU.correct_pct:.2f}% (WU) / {r.positive_WF.correct_pct:.2f}% (WF)\n")
                    
                    f.write(f"      -Steering (Make Faithful Unfaithful - Collateral):\n")
                    f.write(f"        Unfaithfulness Rate:    {r.negative_WF.unfaithful_pct:.2f}%\n")
                    f.write(f"        Hint-Mentioning Rate:   {r.negative_WF.hint_mentioning_pct:.2f}%\n")
                    f.write(f"        Correctness (Pooled):   {r.negative_WF.correct_pct:.2f}% (WF) / {r.negative_WU.correct_pct:.2f}% (WU)\n")
                else:
                    f.write("    [POOLED]: No data found\n")
                
                f.write("\n")
                
                # 2. HINT-WISE RESULTS
                hw_data = hintwise_data.get(model_name, {}).get(approach, {})
                if hw_data:
                    f.write(f"    [HINT-WISE BREAKDOWN]\n")
                    for hint_name, res in hw_data.items():
                        if not res:
                            continue
                        f.write(f"      Hint: {hint_name}\n")
                        f.write(f"        Best Config: Layer {res.layer}, Coeff {res.coefficient}\n")
                        f.write(f"        +Steering Faithfulness (WU->F): {res.positive_WU.faithful_pct:.2f}%\n")
                        f.write(f"        -Steering Unfaithfulness (WF->U): {res.negative_WF.unfaithful_pct:.2f}%\n")
                        f.write(f"        +Steering Hint-Mentioning: {res.positive_WU.hint_mentioning_pct:.2f}%\n")
                        f.write(f"        -Steering Hint-Mentioning: {res.negative_WF.hint_mentioning_pct:.2f}%\n")
                else:
                     f.write("    [HINT-WISE]: No data found\n")
                
                f.write("\n")
            f.write("\n")
    
    # Generate plots
    print("\n=== Generating Plots ===")
    plot_variation_1_hintwise(args.base_dir, output_dir / "variation_1_hintwise.png")
    plot_variation_1(all_data, output_dir / "variation_1.png")
    plot_variation_2(all_data, output_dir / "variation_2.png")
    plot_variation_3(all_data, output_dir / "variation_3.png")
    plot_variation_4(args.base_dir, output_dir / "variation_4.png")
    plot_variation_5(args.base_dir, output_dir / "variation_5_faithfulness.png")
    plot_variation_6(all_data, output_dir / "variation_6_hint_mentioning.png")
    plot_variation_7(all_data, output_dir / "variation_7_with_random.png")
    plot_variation_8(all_data, output_dir / "variation_8_collateral_with_random.png")
    plot_variation_9(all_data, output_dir / "variation_9_hint_collateral_with_random.png")
    plot_variation_10(all_data, output_dir / "variation_10_monitorability.png")
    plot_variation_11(all_data, output_dir / "variation_11_monitorability_with_random.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

