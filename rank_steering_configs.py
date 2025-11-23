import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import math

# ==============================================================================
# STATISTICAL CORRECTOR
# ==============================================================================

class StatisticalCorrector:
    """
    Handles global statistical correction (Benjamini-Hochberg) and continuous
    confidence penalties (Sigmoid).
    """
    def __init__(self):
        self.p_values = []  # List of (id, p_value) tuples
        self.adjusted_p_values = {}  # Map id -> adjusted_p_value
        self.is_collecting = True
        
    def register(self, unique_id, p_value):
        """
        Pass 1: Collect p-values.
        """
        if self.is_collecting:
            # We only care about valid p-values [0, 1]
            if p_value is not None and not np.isnan(p_value):
                self.p_values.append((unique_id, p_value))
        
    def correct(self):
        """
        Apply Benjamini-Yekutieli (FDR) correction.
        More conservative than BH, works with arbitrary dependence.
        """
        self.is_collecting = False
        
        if not self.p_values:
            return

        # Sort by p-value
        sorted_p = sorted(self.p_values, key=lambda x: x[1])
        m = len(sorted_p)
        
        # Compute BY correction constant: c(m) = sum(1/k for k=1..m)
        c_m = sum(1.0 / k for k in range(1, m + 1))
        
        current_min_adjusted = 1.0
        
        # Iterate backwards to enforce monotonicity
        for i in range(m - 1, -1, -1):
            unique_id, p_raw = sorted_p[i]
            rank = i + 1
            
            # Benjamini-Yekutieli: p_adj = p * m * c(m) / rank
            adjusted = p_raw * m * c_m / rank
            
            # Cap at 1.0
            adjusted = min(1.0, adjusted)
            
            # Enforce monotonicity
            adjusted = min(adjusted, current_min_adjusted)
            current_min_adjusted = adjusted
            
            self.adjusted_p_values[unique_id] = adjusted

    def get_penalty_weight(self, unique_id, p_raw, is_desirable):
        """
        Pass 2: Get the weight based on the adjusted p-value using a Sigmoid function.
        """
        if self.is_collecting:
            return 1.0  # No penalty during collection phase
            
        # If we didn't track this ID (e.g. p was NaN), return 0 weight (safe fallback)
        if unique_id not in self.adjusted_p_values:
            return 0.0
            
        p_adj = self.adjusted_p_values[unique_id]
        
        # Sigmoid Function
        if is_desirable:
            # SKEPTICAL: Center at 0.05
            k = 50 
            threshold = 0.05
            weight = 1 / (1 + math.exp(k * (p_adj - threshold)))
        else:
            # PARANOID: Center at 0.15
            k = 30
            threshold = 0.15
            weight = 1 / (1 + math.exp(k * (p_adj - threshold)))

        return weight

# ==============================================================================
# METRIC EXTRACTION
# ==============================================================================

def calculate_p_value(k, n, p_null=0.5):
    """
    Calculates the p-value for a binomial test.
    """
    if n == 0:
        return 1.0
    res = stats.binomtest(k, n, p_null, alternative='two-sided')
    return res.pvalue

def get_unique_id(hint_name, config_key, group_name, metric_name):
    """Generates a unique ID for statistical correction."""
    return f"{hint_name}::{config_key}::{group_name}::{metric_name}"

def extract_metrics_wrong_only(config, hint_name, config_key, corrector):
    """
    Extracts metrics for initially WRONG groups (WU, WF).
    Returns raw rates without statistical weighting.
    """
    # 1. Effectiveness: Positive Effectiveness on WF
    pos_eff_rate = 0.0
    
    if 'positive_on_WF' in config:
        trans = config['positive_on_WF'].get('transitions', {})
        k = trans.get('to_same_answer_unfaithful', {}).get('count', 0)
        n = config['positive_on_WF'].get('n', 0)
        pos_eff_rate = k / n if n > 0 else 0

    # Negative Effectiveness on WF
    neg_eff_rate = 0.0
    
    if 'negative_on_WF' in config:
        trans = config['negative_on_WF'].get('transitions', {})
        k = trans.get('to_same_answer_unfaithful', {}).get('count', 0)
        n = config['negative_on_WF'].get('n', 0)
        neg_eff_rate = k / n if n > 0 else 0

    # Side Effects
    pos_unwanted_rate = 0.0
    
    if 'positive_on_WU' in config:
        trans = config['positive_on_WU'].get('transitions', {})
        k = trans.get('to_same_answer_faithful', {}).get('count', 0)
        n = config['positive_on_WU'].get('n', 0)
        pos_unwanted_rate = k / n if n > 0 else 0

    neg_unwanted_rate = 0.0
    
    if 'negative_on_WU' in config:
        trans = config['negative_on_WU'].get('transitions', {})
        k = trans.get('to_same_answer_faithful', {}).get('count', 0)
        n = config['negative_on_WU'].get('n', 0)
        neg_unwanted_rate = k / n if n > 0 else 0

    # Hint errors, incompleteness, and to_correct
    hint_errors = []
    incompleteness = []
    to_corrects = []
    w_groups = ['positive_on_WU', 'positive_on_WF', 'negative_on_WU', 'negative_on_WF']

    for group_name in w_groups:
        if group_name in config:
            trans = config[group_name].get('transitions', {})
            n = config[group_name].get('n', 0)
            
            k = trans.get('to_hint_error', {}).get('count', 0)
            hint_errors.append(k / n if n > 0 else 0)

            k = trans.get('to_incomplete', {}).get('count', 0)
            incompleteness.append(k / n if n > 0 else 0)

            k = trans.get('to_correct', {}).get('count', 0)
            to_corrects.append(k / n if n > 0 else 0)

    hint_error_rate = np.mean(hint_errors) if hint_errors else 0.0
    incomplete_rate = np.mean(incompleteness) if incompleteness else 0.0
    to_correct_rate = np.mean(to_corrects) if to_corrects else 0.0

    return {
        'pos_effectiveness': pos_eff_rate,
        'neg_effectiveness': neg_eff_rate,
        'pos_unwanted_faithful': pos_unwanted_rate,
        'neg_unwanted_faithful': neg_unwanted_rate,
        'hint_error': hint_error_rate,
        'incomplete': incomplete_rate,
        'to_correct': to_correct_rate
    }


# ==============================================================================
# RANKING LOGIC
# ==============================================================================

def calculate_linear_score(metrics, weights=None):
    """
    Calculates a score based on Weighted Linear Sum.
    Score = (Weight * Benefit) - (Weight * Penalty)
    """
    if weights is None:
        weights = {
            'pos_effectiveness': 3.0,
            'neg_effectiveness': 3.0,
            'to_correct': 2.0,
            'pos_unwanted_faithful': 10.0,
            'neg_unwanted_faithful': 10.0,
            'hint_error': 5.0,
            'incomplete': 5.0
        }
    
    score = 0.0
    
    # Benefits (Add to score)
    score += metrics.get('pos_effectiveness', 0.0) * weights.get('pos_effectiveness', 1.0)
    score += metrics.get('neg_effectiveness', 0.0) * weights.get('neg_effectiveness', 1.0)
    score += metrics.get('to_correct', 0.0) * weights.get('to_correct', 1.0)
    
    # Penalties (Subtract from score)
    score -= metrics.get('pos_unwanted_faithful', 0.0) * weights.get('pos_unwanted_faithful', 1.0)
    score -= metrics.get('neg_unwanted_faithful', 0.0) * weights.get('neg_unwanted_faithful', 1.0)
    score -= metrics.get('hint_error', 0.0) * weights.get('hint_error', 1.0)
    score -= metrics.get('incomplete', 0.0) * weights.get('incomplete', 1.0)
    
    return score

def rank_by_score(configs, weights=None):
    for config in configs:
        config['score'] = calculate_linear_score(config['metrics'], weights)
    return sorted(configs, key=lambda x: x['score'], reverse=True)

# ==============================================================================
# MAIN ANALYSIS FLOW
# ==============================================================================

def analyze_data(data, research_weights=None):
    if research_weights is None:
        research_weights = {
            'pos_effectiveness': 3.0,
            'neg_effectiveness': 3.0,
            'to_correct': 1.0,
            'pos_unwanted_faithful': 10.0,
            'neg_unwanted_faithful': 10.0,
            'hint_error': 5.0,
            'incomplete': 5.0
        }

    hint_templates = list(data.get('configurations_by_hint', {}).keys())
    
    # Extract metrics and calculate scores
    print("Calculating scores...")
    results_by_hint = {}
    all_configs_by_hint = {} # Store all for aggregation
    
    for hint in hint_templates:
        configs_list = data['configurations_by_hint'][hint]
        configs_with_metrics = []
        
        # First pass: extract raw metrics
        for config_data in configs_list:
            layer = config_data.get('layer')
            coeff = config_data.get('coefficient_magnitude')
            if layer is None or coeff is None: continue
            config_key = f"L{layer}_C{coeff}"
            
            w_metrics = extract_metrics_wrong_only(config_data, hint, config_key, None)
            
            has_data = 'positive_on_WF' in config_data or 'negative_on_WF' in config_data or 'positive_on_WU' in config_data or 'negative_on_WU' in config_data
            
            if has_data:
                configs_with_metrics.append({
                    'layer': layer,
                    'coefficient': coeff,
                    'id': config_key,
                    'raw_metrics': w_metrics
                })

        
        # Normalize metrics across all configs for this hint
        metric_keys = ['pos_effectiveness', 'neg_effectiveness', 'pos_unwanted_faithful', 
                       'neg_unwanted_faithful', 'hint_error', 'incomplete', 'to_correct']
        
        # Collect all values per metric
        metric_values = {key: [] for key in metric_keys}
        for cfg in configs_with_metrics:
            for key in metric_keys:
                metric_values[key].append(cfg['raw_metrics'].get(key, 0.0))
        
        # Compute mean and std for each metric
        metric_stats = {}
        for key in metric_keys:
            values = metric_values[key]
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                metric_stats[key] = (mean_val, std_val if std_val > 0 else 1.0)
            else:
                metric_stats[key] = (0.0, 1.0)
        
        # Normalize metrics and apply weights
        w_configs_list = []
        for cfg in configs_with_metrics:
            normalized_metrics = {}
            for key in metric_keys:
                raw_val = cfg['raw_metrics'].get(key, 0.0)
                mean_val, std_val = metric_stats[key]
                # Z-score normalize the metric
                normalized_metrics[key] = (raw_val - mean_val) / std_val if std_val > 0 else 0.0
            
            # Now apply weights to normalized metrics
            score = calculate_linear_score(normalized_metrics, research_weights)
            
            w_configs_list.append({
                'layer': cfg['layer'],
                'coefficient': cfg['coefficient'],
                'metrics': cfg['raw_metrics'],  # Keep raw for output
                'id': cfg['id'],
                'score': score
            })
        
        # Sort by score
        w_ranked = sorted(w_configs_list, key=lambda x: x['score'], reverse=True)
        all_configs_by_hint[hint] = w_ranked
        
        results_by_hint[hint] = {
            'rankings': w_ranked
        }


    # ==========================================================================
    # AGGREGATION (Z-Score Normalization + Power Mean)
    # ==========================================================================
    print("Aggregating scores across hints...")
    
    # 1. Collect all scores: config_id -> {hint: score}
    config_scores = {}
    hint_stats = {} # hint -> (mean, std)
    
    for hint, configs in all_configs_by_hint.items():
        scores = [c['score'] for c in configs]
        if not scores:
            continue
            
        mean_s = np.mean(scores)
        std_s = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        hint_stats[hint] = (mean_s, std_s)
        
        for c in configs:
            cid = c['id']
            if cid not in config_scores:
                config_scores[cid] = {}
            config_scores[cid][hint] = c['score']
            
    # 2. Z-Score Normalization and Power Mean Aggregation
    global_rankings = []
    POWER_MEAN_P = -0.5  # Risk-averse parameter
    
    for cid, scores_map in config_scores.items():
        z_scores = []
        z_scores_map = {}
        
        for hint in hint_templates:
            if hint not in scores_map:
                continue
            
            raw = scores_map[hint]
            mean_s, std_s = hint_stats.get(hint, (0, 1))
            
            # Z-score normalization
            if std_s > 0:
                z = (raw - mean_s) / std_s
            else:
                z = 0.0  # All scores identical for this hint
                
            z_scores.append(z)
            z_scores_map[hint] = z
            
        if z_scores:
            # Power Mean: (mean(z^p))^(1/p)
            # For p < 0, need to shift scores to be positive
            min_z = min(z_scores)
            if min_z <= 0:
                shift = abs(min_z) + 0.1
                shifted_scores = [z + shift for z in z_scores]
            else:
                shift = 0
                shifted_scores = z_scores
            
            # Compute power mean
            power_mean_val = (sum(z**POWER_MEAN_P for z in shifted_scores) / len(shifted_scores)) ** (1/POWER_MEAN_P)
            global_score = power_mean_val - shift
            
            # Parse ID for metadata
            try:
                parts = cid.replace('L', '').split('_C')
                layer = int(parts[0])
                coeff = float(parts[1])
            except:
                layer, coeff = 0, 0.0

            global_rankings.append({
                'id': cid,
                'layer': layer,
                'coefficient': coeff,
                'global_score': global_score,
                'raw_scores': scores_map,
                'z_scores': z_scores_map
            })
            
    # Sort by Global Score
    global_rankings.sort(key=lambda x: x['global_score'], reverse=True)
    
    return {
        'by_hint': results_by_hint,
        'global_ranking': global_rankings
    }

def main():
    parser = argparse.ArgumentParser(description='Rank steering configurations with Global Statistical Correction.')
    parser.add_argument('input_file', nargs='?', default='data/sprint_5_2025-11-15/summaries/steered_faithfulness/summary_steered_val_mean_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-11-21.jsonl')
    args = parser.parse_args()

    print(f"Loading summary from: {args.input_file}")
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}")
        return

    results = analyze_data(data)

    output_dir = Path('data/sprint_5_2025-11-15/analysis_ranking_configs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"rankings_steered_val_mean_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'source_file': args.input_file,
                'timestamp': timestamp,
                'method': 'Global Benjamini-Hochberg + Sigmoid Penalty (Wrong Groups Only)'
            },
            'results': results
        }, f, indent=2)
        
    print(f"\nAnalysis saved to: {output_file}")

if __name__ == "__main__":
    main()
