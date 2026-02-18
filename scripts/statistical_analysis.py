
import sys
import os
# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
from scipy.stats import norm

# =============================================================================
# Configuration & Constants
# =============================================================================

# Hardcoded Winning Configurations (Best Monitorability Recovery)
# Format: {Model: {Approach: (Layer, Coefficient)}}
WINNING_CONFIGS = {
    "DeepSeek-R1-Distill-Llama-8B": {
        "linear": (15, 2.0),
        "off_policy": (15, 2.0),
        "mlp": (25, 5.0),
        "random": (15, 2.0)
    },
    "Qwen3-14B": {
        "linear": (19, 2.0),
        "off_policy": (19, 1.0),
        "mlp": (7, 20.0),
        "random": (15, 2.0)
    },
    "Qwen3-32B": {
        "linear": (40, 0.6),
        "off_policy": (58, 2.0),
        "mlp": (41, 5.0),
        "random": (15, 2.0)
    }
}

# Mapping for file paths
MODEL_SUFFIXES = {
    "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    "Qwen3-14B": "Qwen3-14B",
    "Qwen3-32B": "Qwen3-32B"
}

BASE_DIR = Path("C:/Users/occhi/Desktop/unfaithfulness_steering")

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SteeringMetrics:
    """Raw counts for a specific steering configuration."""
    model: str
    approach: str
    layer: int
    coefficient: float
    
    # Monitorability Recovery (MR) components
    mr_success_count: int = 0
    mr_total_count: int = 0
    
    # Degradation (D) components
    d_prob_count: int = 0  # problematic Outcome count
    d_total_count: int = 0
    
    # Collateral Degradation (CD) components
    cd_prob_count: int = 0
    cd_total_count: int = 0
    
    # Collateral Monitorability Recovery (CMR) components
    cmr_success_count: int = 0
    cmr_total_count: int = 0

    # Lists to store hint-wise rates for averaging
    all_mr_rates: List[float] = field(default_factory=list)
    all_d_rates: List[float] = field(default_factory=list)
    all_cd_rates: List[float] = field(default_factory=list)
    all_cmr_rates: List[float] = field(default_factory=list)

    @property
    def mr_rate(self) -> float:
        # Return average of hint-wise rates if available, else pooled
        if self.all_mr_rates:
            return sum(self.all_mr_rates) / len(self.all_mr_rates)
        return self.mr_success_count / self.mr_total_count if self.mr_total_count > 0 else 0.0

    @property
    def d_rate(self) -> float:
        if self.all_d_rates:
            return sum(self.all_d_rates) / len(self.all_d_rates)
        return self.d_prob_count / self.d_total_count if self.d_total_count > 0 else 0.0

    @property
    def cd_rate(self) -> float:
        if self.all_cd_rates:
            return sum(self.all_cd_rates) / len(self.all_cd_rates)
        return self.cd_prob_count / self.cd_total_count if self.cd_total_count > 0 else 0.0
    
    @property
    def cmr_rate(self) -> float:
        if self.all_cmr_rates:
            return sum(self.all_cmr_rates) / len(self.all_cmr_rates)
        return self.cmr_success_count / self.cmr_total_count if self.cmr_total_count > 0 else 0.0

# =============================================================================
# Data Loading & extraction
# =============================================================================

def get_transitions_count(transitions: dict, keys: List[str]) -> int:
    """Sum counts for specified transition keys."""
    count = 0
    for key in keys:
        if key.endswith("*"): # Wildcard support
            prefix = key[:-1]
            count += sum(v.get("count", 0) for k, v in transitions.items() if k.startswith(prefix))
        elif key in transitions:
            count += transitions[key].get("count", 0)
    
    # Special handling for "any transition mentioning hint"
    if "ANY_HINT_MENTIONING" in keys:
         count += sum(v.get("count", 0) for k, v in transitions.items() if "_mentioning_hint" in k)

    return count

def extract_config_metrics(config_data: dict, model: str, approach: str, layer: int, coeff: float) -> SteeringMetrics:
    metrics = SteeringMetrics(model, approach, layer, coeff)
    
    # 1. Monitorability Recovery (Positive on WU)
    # Success: stable_faithful OR any hint mentioning
    pos_wu = config_data.get("positive_on_WU", {})
    trans = pos_wu.get("transitions", {})
    metrics.mr_total_count = pos_wu.get("n", 0)
    
    # Logic: stable_faithful + ANY transition ending in _mentioning_hint
    success_mr = trans.get("stable_faithful", {}).get("count", 0)
    success_mr += sum(v.get("count", 0) for k, v in trans.items() if k.endswith("_mentioning_hint"))
    metrics.mr_success_count = success_mr

    # 2. Degradation (Negative on WF)
    # Problem: stable_unfaithful
    neg_wf = config_data.get("negative_on_WF", {})
    trans = neg_wf.get("transitions", {})
    metrics.d_total_count = neg_wf.get("n", 0)
    metrics.d_prob_count = trans.get("stable_unfaithful", {}).get("count", 0)

    # 3. Collateral Degradation (Positive on WF)
    # Problem: stable_unfaithful
    pos_wf = config_data.get("positive_on_WF", {})
    trans = pos_wf.get("transitions", {})
    metrics.cd_total_count = pos_wf.get("n", 0)
    metrics.cd_prob_count = trans.get("stable_unfaithful", {}).get("count", 0)

    # 4. Collateral Monitorability Recovery (Negative on WU)
    # Success: stable_faithful OR any hint mentioning
    neg_wu = config_data.get("negative_on_WU", {})
    trans = neg_wu.get("transitions", {})
    metrics.cmr_total_count = neg_wu.get("n", 0)
    
    success_cmr = trans.get("stable_faithful", {}).get("count", 0)
    success_cmr += sum(v.get("count", 0) for k, v in trans.items() if k.endswith("_mentioning_hint"))
    metrics.cmr_success_count = success_cmr
    
    return metrics

def load_metrics_for_config(model: str, approach: str) -> Optional[SteeringMetrics]:
    """Load data for the specific winning configuration."""
    if model not in WINNING_CONFIGS or approach not in WINNING_CONFIGS[model]:
        print(f"No winning config for {model} {approach}")
        return None
        
    target_layer, target_coeff = WINNING_CONFIGS[model][approach]
    model_suffix = MODEL_SUFFIXES[model]
    
    # Path construction
    data_dir = BASE_DIR / "data" / "definitive_pipeline_data" / model_suffix
    pattern = f"summary_steered_{approach}_{model_suffix}_*.json"
    files = list(data_dir.glob(pattern))
    
    if not files:
        print(f"File not found for {pattern}")
        return None
        
    latest_file = sorted(files)[-1]
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Aggregate counts across all hint templates for the target config
    # We need to find the specific (layer, coeff) in "configurations_by_hint"
    # and sum up the raw N and Success counts.
    
    aggregated_metrics = SteeringMetrics(model, approach, target_layer, target_coeff)
    
    configs_by_hint = data.get("configurations_by_hint", {})
    found_any = False

    for hint_type, configs in configs_by_hint.items():
        # Find the specific config in this hint's list
        matching_cfg = next((c for c in configs if c.get("layer") == target_layer and math.isclose(c.get("coefficient_magnitude"), target_coeff, rel_tol=1e-5)), None)
        
        if matching_cfg:
            found_any = True
            # Extract metrics for this hint
            hint_metrics = extract_config_metrics(matching_cfg, model, approach, target_layer, target_coeff)
            
            # Add to aggregate (sum counts)
            aggregated_metrics.mr_success_count += hint_metrics.mr_success_count
            aggregated_metrics.mr_total_count += hint_metrics.mr_total_count
            
            aggregated_metrics.d_prob_count += hint_metrics.d_prob_count
            aggregated_metrics.d_total_count += hint_metrics.d_total_count
            
            aggregated_metrics.cd_prob_count += hint_metrics.cd_prob_count
            aggregated_metrics.cd_total_count += hint_metrics.cd_total_count
            
            aggregated_metrics.cmr_success_count += hint_metrics.cmr_success_count
            aggregated_metrics.cmr_total_count += hint_metrics.cmr_total_count
            
            # Store hint-wise rates for averaging
            aggregated_metrics.all_mr_rates.append(hint_metrics.mr_rate)
            aggregated_metrics.all_d_rates.append(hint_metrics.d_rate)
            aggregated_metrics.all_cd_rates.append(hint_metrics.cd_rate)
            aggregated_metrics.all_cmr_rates.append(hint_metrics.cmr_rate)

            
    if not found_any:
        print(f"Warning: Config L{target_layer} C{target_coeff} not found in {latest_file.name}")
        return None
        
    return aggregated_metrics

# =============================================================================
# Statistical Tests
# =============================================================================

def z_test_proportions(count1, n1, count2, n2):
    """
    Two-tailed Z-test for two proportions.
    Returns: (z_stat, p_value, cohens_h)
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0, 0.0
        
    p1 = count1 / n1
    p2 = count2 / n2
    
    # Pooled proportion
    p_pool = (count1 + count2) / (n1 + n2)
    
    # Standard Error
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    if se == 0:
        return 0.0, 1.0, 0.0
        
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed
    
    # Cohen's h
    # Arcsine transformation: phi = 2 * arcsin(sqrt(p))
    # h = phi1 - phi2
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    h = abs(phi1 - phi2)
    
    return z, p_value, h

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction.
    Returns boolean array of rejected hypotheses (significant).
    """
    n = len(p_values)
    if n == 0:
        return []
        
    # Sort p-values and keep indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Critical values: (i/n) * alpha
    critical_vals = (np.arange(1, n + 1) / n) * alpha
    
    # Find largest k where p_k <= (k/n)*alpha
    below_critical = sorted_p <= critical_vals
    
    if not np.any(below_critical):
        return [False] * n
        
    max_k_idx = np.where(below_critical)[0][-1]
    
    # Reject all null hypotheses with p-value <= p_max_k
    reject_threshold = sorted_p[max_k_idx]
    
    return [p <= reject_threshold for p in p_values]

# =============================================================================
# Main Execution
# =============================================================================

def run_analysis():
    print("Loading data...")
    models = ["DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B", "Qwen3-32B"]
    approaches = ["linear", "off_policy", "mlp", "random"]
    
    # Load all metrics
    data_store = {} # {model: {approach: Metrics}}
    
    for model in models:
        data_store[model] = {}
        for approach in approaches:
            print(f"  Processing {model} - {approach}...")
            metrics = load_metrics_for_config(model, approach)
            if metrics:
                data_store[model][approach] = metrics

    # Define Analysis Scenarios
    # List of (TestName, MetricAttr, BaselineApproach)
    scenarios = [
        ("Monitorability Recovery (vs Random)", "mr", "random"),
        ("Degradation (vs Random)", "d", "random"),
        ("Collateral Degradation (vs Random)", "cd", "random"),
        ("Collateral Monitorability Recovery (vs Random)", "cmr", "random")
    ]
    
    results_text = []
    results_text.append("Statistical Analysis Results\n============================")

    # 1. VS RANDOM BASELINE
    for scenario_name, metric_prefix, baseline_name in scenarios:
        results_text.append(f"\n## {scenario_name}")
        results_text.append(f"{'Model':<30} | {'Approach':<12} | {'Rate(%)':<8} | {'Base(%)':<8} | {'Diff':<6} | {'P-Value':<10} | {'Cohen`s h':<10} | {'Sig'}")
        results_text.append("-" * 120)
        
        p_values = []
        test_records = []
        
        # Collect all tests for this scenario (for FDR family)
        for model in models:
            baseline = data_store.get(model, {}).get(baseline_name)
            if not baseline:
                continue
                
            for approach in ["linear", "off_policy", "mlp"]:
                current = data_store.get(model, {}).get(approach)
                if not current:
                    continue
                
                # Get counts based on metric
                if metric_prefix == "mr":
                    c1, n1 = current.mr_success_count, current.mr_total_count
                    c2, n2 = baseline.mr_success_count, baseline.mr_total_count
                    rate1 = current.mr_rate
                    rate2 = baseline.mr_rate
                elif metric_prefix == "d":
                    c1, n1 = current.d_prob_count, current.d_total_count
                    c2, n2 = baseline.d_prob_count, baseline.d_total_count
                    rate1 = current.d_rate
                    rate2 = baseline.d_rate
                elif metric_prefix == "cd":
                    c1, n1 = current.cd_prob_count, current.cd_total_count
                    c2, n2 = baseline.cd_prob_count, baseline.cd_total_count
                    rate1 = current.cd_rate
                    rate2 = baseline.cd_rate
                elif metric_prefix == "cmr":
                    c1, n1 = current.cmr_success_count, current.cmr_total_count
                    c2, n2 = baseline.cmr_success_count, baseline.cmr_total_count
                    rate1 = current.cmr_rate
                    rate2 = baseline.cmr_rate
                
                # Z-test uses EFFECTIVE counts derived from Average Rate
                # logic: S_effective = Rate_avg * N_total
                # This aligns the statistical test with the metric being reported (Average Rate).
                
                c1_eff = round(rate1 * n1)
                c2_eff = round(rate2 * n2)
                
                z, p, h_pooled = z_test_proportions(c1_eff, n1, c2_eff, n2)
                
                # Cohen's h uses the average rates directly
                try:
                    p1_avg = rate1
                    p2_avg = rate2
                    # Clamp for arcsine
                    p1_avg = max(0.0, min(1.0, p1_avg))
                    p2_avg = max(0.0, min(1.0, p2_avg))
                    phi1 = 2 * math.asin(math.sqrt(p1_avg))
                    phi2 = 2 * math.asin(math.sqrt(p2_avg))
                    h = abs(phi1 - phi2)
                except:
                    h = h_pooled # Fallback

                p_values.append(p)
                test_records.append({
                    "model": model,
                    "approach": approach,
                    "rate1": rate1*100,
                    "rate2": rate2*100,
                    "diff": (rate1 - rate2)*100,
                    "p": p,
                    "h": h
                })
        
        # Apply FDR
        is_significant = benjamini_hochberg(p_values)
        
        # Print
        for rec, sig in zip(test_records, is_significant):
            sig_mark = "*" if sig else ""
            results_text.append(f"{rec['model']:<30} | {rec['approach']:<12} | {rec['rate1']:6.2f} | {rec['rate2']:6.2f} | {rec['diff']:+6.2f} | {rec['p']:<10.2e} | {rec['h']:<10.4f} | {sig_mark}")

    # 2. COMPARATIVE ANALYSIS (Linear vs MLP, Linear vs Off-Policy, etc.)
    # Group by model
    results_text.append("\n\n## Comparative Analysis (Approach vs Approach)")
    
    comparisons = [
        ("linear", "mlp"),
        ("linear", "off_policy"),
        ("off_policy", "mlp")
    ]
    
    metrics_to_test = [
        ("mr", "Monitorability Recovery"),
        ("d", "Degradation"),
        ("cd", "Collateral Degradation"),
        ("cmr", "Collateral Monitorability")
    ]

    for metric_prefix, metric_name in metrics_to_test:
        results_text.append(f"\n### Metric: {metric_name}")
        results_text.append(f"{'Model':<30} | {'Comparison':<20} | {'Rate1(%)':<8} | {'Rate2(%)':<8} | {'P-Value':<10} | {'Cohen`s h':<10} | {'Sig'}")
        results_text.append("-" * 120)

        p_values = []
        test_records = []

        for model in models:
            for app1_name, app2_name in comparisons:
                app1 = data_store.get(model, {}).get(app1_name)
                app2 = data_store.get(model, {}).get(app2_name)
                
                if not app1 or not app2:
                    continue
                    
                # Get counts and rates
                if metric_prefix == "mr":
                    c1, n1 = app1.mr_success_count, app1.mr_total_count
                    c2, n2 = app2.mr_success_count, app2.mr_total_count
                    rate1 = app1.mr_rate
                    rate2 = app2.mr_rate
                elif metric_prefix == "d":
                    c1, n1 = app1.d_prob_count, app1.d_total_count
                    c2, n2 = app2.d_prob_count, app2.d_total_count
                    rate1 = app1.d_rate
                    rate2 = app2.d_rate
                elif metric_prefix == "cd":
                    c1, n1 = app1.cd_prob_count, app1.cd_total_count
                    c2, n2 = app2.cd_prob_count, app2.cd_total_count
                    rate1 = app1.cd_rate
                    rate2 = app2.cd_rate
                elif metric_prefix == "cmr":
                    c1, n1 = app1.cmr_success_count, app1.cmr_total_count
                    c2, n2 = app2.cmr_success_count, app2.cmr_total_count
                    rate1 = app1.cmr_rate
                    rate2 = app2.cmr_rate
                
                # Z-test uses EFFECTIVE counts derived from Average Rate
                c1_eff = round(rate1 * n1)
                c2_eff = round(rate2 * n2)
                
                z, p, h_pooled = z_test_proportions(c1_eff, n1, c2_eff, n2)

                # Recalculate Cohen's h based on averaged rates
                try:
                    p1_avg = rate1
                    p2_avg = rate2
                    p1_avg = max(0.0, min(1.0, p1_avg))
                    p2_avg = max(0.0, min(1.0, p2_avg))
                    phi1 = 2 * math.asin(math.sqrt(p1_avg))
                    phi2 = 2 * math.asin(math.sqrt(p2_avg))
                    h = abs(phi1 - phi2)
                except:
                    h = h_pooled

                p_values.append(p)
                test_records.append({
                    "model": model,
                    "comp": f"{app1_name} vs {app2_name}",
                    "rate1": rate1*100,
                    "rate2": rate2*100,
                    "p": p,
                    "h": h
                })
        
        # Apply FDR
        is_significant = benjamini_hochberg(p_values)
        
        for rec, sig in zip(test_records, is_significant):
            sig_mark = "*" if sig else ""
            results_text.append(f"{rec['model']:<30} | {rec['comp']:<20} | {rec['rate1']:6.2f} | {rec['rate2']:6.2f} | {rec['p']:<10.2e} | {rec['h']:<10.4f} | {sig_mark}")

    # 3. MODEL-WISE COMPARISON (Same Approach, Different Models)
    results_text.append("\n\n## Model-wise Analysis (Model vs Model)")
    
    # We compare models for EACH approach
    # e.g., Linear: DeepSeek vs Qwen14
    
    comparisons_models = [
        ("DeepSeek-R1-Distill-Llama-8B", "Qwen3-14B"),
        ("DeepSeek-R1-Distill-Llama-8B", "Qwen3-32B"),
        ("Qwen3-14B", "Qwen3-32B")
    ]
    
    for metric_prefix, metric_name in metrics_to_test:
        results_text.append(f"\n### Metric: {metric_name}")
        results_text.append(f"{'Approach':<12} | {'Comparison':<40} | {'Rate1(%)':<8} | {'Rate2(%)':<8} | {'P-Value':<10} | {'Cohen`s h':<10} | {'Sig'}")
        results_text.append("-" * 140)

        p_values = []
        test_records = []

        for approach in approaches:
            for m1_name, m2_name in comparisons_models:
                m1_data = data_store.get(m1_name, {}).get(approach)
                m2_data = data_store.get(m2_name, {}).get(approach)
                
                if not m1_data or not m2_data:
                    continue
                    
                # Get counts and rates
                if metric_prefix == "mr":
                    c1, n1 = m1_data.mr_success_count, m1_data.mr_total_count
                    c2, n2 = m2_data.mr_success_count, m2_data.mr_total_count
                    rate1 = m1_data.mr_rate
                    rate2 = m2_data.mr_rate
                elif metric_prefix == "d":
                    c1, n1 = m1_data.d_prob_count, m1_data.d_total_count
                    c2, n2 = m2_data.d_prob_count, m2_data.d_total_count
                    rate1 = m1_data.d_rate
                    rate2 = m2_data.d_rate
                elif metric_prefix == "cd":
                    c1, n1 = m1_data.cd_prob_count, m1_data.cd_total_count
                    c2, n2 = m2_data.cd_prob_count, m2_data.cd_total_count
                    rate1 = m1_data.cd_rate
                    rate2 = m2_data.cd_rate
                elif metric_prefix == "cmr":
                    c1, n1 = m1_data.cmr_success_count, m1_data.cmr_total_count
                    c2, n2 = m2_data.cmr_success_count, m2_data.cmr_total_count
                    rate1 = m1_data.cmr_rate
                    rate2 = m2_data.cmr_rate
                
                # Z-test uses EFFECTIVE counts derived from Average Rate
                c1_eff = round(rate1 * n1)
                c2_eff = round(rate2 * n2)
                
                z, p, h_pooled = z_test_proportions(c1_eff, n1, c2_eff, n2)

                # Recalculate Cohen's h based on averaged rates
                try:
                    p1_avg = rate1
                    p2_avg = rate2
                    p1_avg = max(0.0, min(1.0, p1_avg))
                    p2_avg = max(0.0, min(1.0, p2_avg))
                    phi1 = 2 * math.asin(math.sqrt(p1_avg))
                    phi2 = 2 * math.asin(math.sqrt(p2_avg))
                    h = abs(phi1 - phi2)
                except:
                    h = h_pooled

                p_values.append(p)
                test_records.append({
                    "approach": approach,
                    "comp": f"{m1_name} vs {m2_name}",
                    "rate1": rate1*100,
                    "rate2": rate2*100,
                    "p": p,
                    "h": h
                })
        
        # Apply FDR
        is_significant = benjamini_hochberg(p_values)
        
        for rec, sig in zip(test_records, is_significant):
            sig_mark = "*" if sig else ""
            results_text.append(f"{rec['approach']:<12} | {rec['comp']:<40} | {rec['rate1']:6.2f} | {rec['rate2']:6.2f} | {rec['p']:<10.2e} | {rec['h']:<10.4f} | {sig_mark}")

    # Output to File
    out_path = BASE_DIR / "statistical_results_detailed.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_text))
    
    print(f"\nAnalysis Complete. Results saved to: {out_path}")
    print("\n".join(results_text))

if __name__ == "__main__":
    run_analysis()
