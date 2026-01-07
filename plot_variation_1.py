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


def parse_configurations(summary: dict, hint_template: str = "grader_hacking") -> list[ConfigResult]:
    """Parse all configurations from summary JSON."""
    configs = summary.get("configurations_by_hint", {}).get(hint_template, [])
    results = []
    
    for cfg in configs:
        result = ConfigResult(
            layer=cfg.get("layer"),
            coefficient=cfg.get("coefficient_magnitude")
        )
        
        for json_key, attr in [
            ("positive_on_WF", "positive_WF"),
            ("positive_on_WU", "positive_WU"),
            ("negative_on_WF", "negative_WF"),
            ("negative_on_WU", "negative_WU"),
        ]:
            data = cfg.get(json_key, {})
            metrics = extract_metrics(data.get("transitions", {}))
            metrics.n = data.get("n", 0)
            setattr(result, attr, metrics)
        
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


def load_approach(data_dir: Path, mode: str, model_suffix: str, hint_template: str) -> ApproachData:
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
    
    configs = parse_configurations(summary, hint_template)
    best = find_best_config(configs) if configs else None
    
    return ApproachData(
        best_config=(best.layer, best.coefficient) if best else None,
        best_result=best,
        all_configs=configs
    )


def load_qwen3_32b(base_dir: Path, hint_template: str = "grader_hacking") -> ModelData:
    """Load all Qwen3-32B data."""
    data_dir = base_dir / "data" / "definitive_pipeline_data" / "Qwen3-32B"
    print("Loading Qwen3-32B...")
    
    model = ModelData()
    model.off_policy = load_approach(data_dir, "off_policy", "Qwen3-32B", hint_template)
    model.linear = load_approach(data_dir, "linear", "Qwen3-32B", hint_template)
    model.mlp = load_approach(data_dir, "mlp", "Qwen3-32B", hint_template)
    
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # Green, Blue, Red for models
    
    # Panel specifications: (row, col), direction_attr, metric_attr, title, ylabel, is_hnm
    panels = [
        # Row 0: Faithfulness
        (0, 0, "positive_WU", "faithful_pct", "+Steer WU → F%", "Faithful %", False),
        (0, 1, "negative_WF", "unfaithful_pct", "−Steer WF → U%", "Unfaithful %", False),
        # Row 1: Hint-Mentioning
        (1, 0, "positive_WU", "hint_mentioning_pct", "+Steer WU → Hm%", "Hint-Mentioning %", False),
        (1, 1, "negative_WF", "hint_mentioning_pct", "−Steer WF → Hnm%", "Hint-Not-Mentioning %", True),
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
                    # Hnm% = 100 - Hm%
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.85)
        
        # Large font sizes for print readability
        ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=12)
    
    # Main title
    fig.suptitle("Variation 1: Good Behavior of Both Steering Directions", 
                 fontsize=20, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # Green, Blue, Red for models
    
    # Panel specifications: (row, col), direction_attr, metric_attr, title, ylabel, is_hnm
    panels = [
        # Row 0: Faithfulness
        (0, 0, "positive_WU", "faithful_pct", "+Steer WU → F% (Good)", "Faithful %", False),
        (0, 1, "positive_WF", "unfaithful_pct", "+Steer WF → U% (Bad)", "Unfaithful %", False),
        # Row 1: Hint-Mentioning
        (1, 0, "positive_WU", "hint_mentioning_pct", "+Steer WU → Hm% (Good)", "Hint-Mentioning %", False),
        (1, 1, "positive_WF", "hint_mentioning_pct", "+Steer WF → Hnm% (Bad)", "Hint-Not-Mentioning %", True),
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
                    # Hnm% = 100 - Hm%
                    if is_hnm:
                        value = 100 - value
                    values.append(value)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.85)
        
        # Large font sizes for print readability
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=12)
    
    # Main title
    fig.suptitle("Variation 2: Good vs Bad Behavior of +Steering", 
                 fontsize=18, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
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
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    panels = [
        (0, "positive_WU", "faithful_pct", "+WU → F%", "Faithful %", False),
        (1, "negative_WF", "unfaithful_pct", "−WF → U%", "Unfaithful %", False),
        (2, "positive_WU", "hint_mentioning_pct", "+WU → Hm%", "Hint-Mentioning %", False),
        (3, "negative_WF", "hint_mentioning_pct", "−WF → Hnm%", "Hint-Not-Mentioning %", True),
    ]
    
    for col, direction, metric, title, ylabel, is_hnm in panels:
        ax = axes[col]
        
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
            ax.legend(loc="upper right", fontsize=10)
    
    fig.suptitle("Variation 3A: Both Steering Directions (Compressed)", 
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
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
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    panels = [
        (0, "positive_WU", "faithful_pct", "+WU → F% (Good)", "Faithful %", False),
        (1, "positive_WF", "unfaithful_pct", "+WF → U% (Bad)", "Unfaithful %", False),
        (2, "positive_WU", "hint_mentioning_pct", "+WU → Hm% (Good)", "Hint-Mentioning %", False),
        (3, "positive_WF", "hint_mentioning_pct", "+WF → Hnm% (Bad)", "Hint-Not-Mentioning %", True),
    ]
    
    for col, direction, metric, title, ylabel, is_hnm in panels:
        ax = axes[col]
        
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
            ax.legend(loc="upper right", fontsize=10)
    
    fig.suptitle("Variation 3B: +Steering Good vs Bad (Compressed)", 
                 fontsize=16, fontweight="bold", y=1.02)
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
    Variation 4: Three Rows — Add Correctness to V1
    
    | Row             | Left (9 bars)  | Right (9 bars) |
    |-----------------|----------------|----------------|
    | Faithfulness    | +steer WU→F%   | −steer WF→U%   |
    | Hint-Mentioning | +steer WU→Hm%  | −steer WF→Hnm% |
    | Correctness     | +steer I→C%    | −steer I→C%    |
    
    Note: I→C% pools WF and WU as "Incorrect" initial state.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
    approaches = ["off_policy", "linear", "mlp"]
    approach_labels = ["Off-Policy", "Linear", "MLP"]
    
    x = np.arange(len(approaches))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    # Panel specifications
    panels = [
        # Row 0: Faithfulness
        (0, 0, "positive_WU", "faithful_pct", "+Steer WU → F%", "Faithful %", False),
        (0, 1, "negative_WF", "unfaithful_pct", "−Steer WF → U%", "Unfaithful %", False),
        # Row 1: Hint-Mentioning
        (1, 0, "positive_WU", "hint_mentioning_pct", "+Steer WU → Hm%", "Hint-Mentioning %", False),
        (1, 1, "negative_WF", "hint_mentioning_pct", "−Steer WF → Hnm%", "Hint-Not-Mentioning %", True),
        # Row 2: Correctness (pool WF+WU as "Incorrect")
        (2, 0, "positive_WU", "correct_pct", "+Steer I → C%", "Correct %", False),
        (2, 1, "negative_WU", "correct_pct", "−Steer I → C%", "Correct %", False),
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
        
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(approach_labels, fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=12)
    
    fig.suptitle("Variation 4: Faithfulness + Hint-Mentioning + Correctness", 
                 fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    
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
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
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
    
    models = ["Qwen3-32B", "DeepSeek-R1-Llama-8B", "Qwen3-14B"]
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
    
    # Load real data for Qwen3-32B
    qwen_data = load_qwen3_32b(args.base_dir, args.hint_template)
    
    # Generate placeholders for other models
    random.seed(42)
    all_data = {
        "Qwen3-32B": qwen_data,
        "DeepSeek-R1-Llama-8B": generate_placeholder_model(),
        "Qwen3-14B": generate_placeholder_model()
    }
    
    # Print best configs
    print("\n=== Best Configurations (Qwen3-32B) ===")
    for approach in ["off_policy", "linear", "mlp"]:
        ad = getattr(qwen_data, approach)
        if ad.best_config:
            layer, coeff = ad.best_config
            r = ad.best_result
            print(f"{approach.upper()}: layer={layer}, coeff={coeff}")
            print(f"  +WU→F%: {r.positive_WU.faithful_pct:.1f}%")
            print(f"  −WF→U%: {r.negative_WF.unfaithful_pct:.1f}%")
            print(f"  +WU→Hm%: {r.positive_WU.hint_mentioning_pct:.1f}%")
    
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
