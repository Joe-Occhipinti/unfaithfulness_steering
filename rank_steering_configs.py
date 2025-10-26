"""
rank_steering_configs.py

Complete ranking system for steering configurations with:
- Separate analysis for Correct groups (CU/CF) and Wrong groups (WU/WF)
- Combined analysis with proper sample-size weighting
- Unified analysis (averaging CU+WU and CF+WF)
- All ranking functionality in one place
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT_FILE = "data/sprint4_2025-10-21/summaries/steered_faithfulness/summary_faithfulness_steered_neg_bas_psyXprof_2025-10-19.json"

# =============================================================================
# METRICS EXTRACTION FOR SEPARATED GROUPS
# =============================================================================

def extract_metrics_correct_only(config: Dict[str, Any], apply_confidence_penalty: bool = True) -> Dict[str, float]:
    """
    Extract metrics only for initially CORRECT groups (CU, CF).

    Returns:
        Dictionary with metrics for C groups only
    """
    from scipy import stats

    def group_exists(group_name: str) -> bool:
        return group_name in config and config[group_name].get('n', 0) > 0

    def calculate_confidence_weight(group_name: str, transition_type: str,
                                   is_desirable: bool = True) -> float:
        if not apply_confidence_penalty:
            return 1.0

        if not group_exists(group_name):
            return 0.0

        group_data = config[group_name]
        n = group_data['n']
        transitions = group_data.get('transitions', {})
        transition_data = transitions.get(transition_type, {})
        count = transition_data.get('count', 0)

        if n <= 0:
            return 0.0

        if is_desirable:
            # Test if effectiveness is significantly > 0%
            p_value = stats.binomtest(count, n, p=0.0, alternative='greater').pvalue
        else:
            # Test if side effects are significantly > 0% (same as desirable)
            p_value = stats.binomtest(count, n, p=0.0, alternative='greater').pvalue

        if p_value < 0.05:
            final_weight = 1.0  # Significant - full weight
        else:
            final_weight = 0.5  # Not significant - half weight

        if not is_desirable:
            final_weight = max(0.5, final_weight)  # Side effects always get at least 0.5

        return final_weight

    # POSITIVE STEERING on CU only (CU -> faithful)
    pos_effectiveness = 0.0
    if group_exists('positive_on_CU'):
        transitions = config['positive_on_CU'].get('transitions', {})
        faithful_rate = transitions.get('to_same_answer_faithful', {}).get('rate', 0)
        confidence = calculate_confidence_weight('positive_on_CU', 'to_same_answer_faithful', is_desirable=True)
        pos_effectiveness = faithful_rate * confidence

    # NEGATIVE STEERING on CF only (CF -> unfaithful)
    neg_effectiveness = 0.0
    if group_exists('negative_on_CF'):
        transitions = config['negative_on_CF'].get('transitions', {})
        unfaithful_rate = transitions.get('to_same_answer_unfaithful', {}).get('rate', 0)
        confidence = calculate_confidence_weight('negative_on_CF', 'to_same_answer_unfaithful', is_desirable=True)
        neg_effectiveness = unfaithful_rate * confidence

    # Side effects for C groups
    pos_unwanted = 0.0
    if group_exists('positive_on_CF'):
        rate = config['positive_on_CF'].get('transitions', {}).get('to_same_answer_unfaithful', {}).get('rate', 0)
        confidence = calculate_confidence_weight('positive_on_CF', 'to_same_answer_unfaithful', is_desirable=False)
        pos_unwanted = rate * confidence

    neg_unwanted = 0.0
    if group_exists('negative_on_CU'):
        rate = config['negative_on_CU'].get('transitions', {}).get('to_same_answer_faithful', {}).get('rate', 0)
        confidence = calculate_confidence_weight('negative_on_CU', 'to_same_answer_faithful', is_desirable=False)
        neg_unwanted = rate * confidence

    # Hint errors and incompleteness for C groups only
    hint_errors = []
    incompleteness = []
    c_groups = ['positive_on_CU', 'positive_on_CF', 'negative_on_CU', 'negative_on_CF']

    for group_name in c_groups:
        if group_exists(group_name):
            # Hint errors
            rate = config[group_name].get('transitions', {}).get('to_hint_error', {}).get('rate', 0)
            confidence = calculate_confidence_weight(group_name, 'to_hint_error', is_desirable=False)
            hint_errors.append(rate * confidence)

            # Incomplete
            rate = config[group_name].get('transitions', {}).get('to_incomplete', {}).get('rate', 0)
            confidence = calculate_confidence_weight(group_name, 'to_incomplete', is_desirable=False)
            incompleteness.append(rate * confidence)

    hint_error_rate = np.mean(hint_errors) if hint_errors else 0.0
    incomplete_rate = np.mean(incompleteness) if incompleteness else 0.0

    # Check if we have any C groups at all
    has_c_groups = (group_exists('positive_on_CU') or group_exists('negative_on_CF') or
                    group_exists('positive_on_CF') or group_exists('negative_on_CU'))

    return {
        'pos_effectiveness': pos_effectiveness,
        'neg_effectiveness': neg_effectiveness,
        'pos_unwanted_faithful': pos_unwanted,
        'neg_unwanted_faithful': neg_unwanted,
        'hint_error': hint_error_rate,
        'incomplete': incomplete_rate,
        'layer': config.get('layer'),
        'coefficient': config.get('coefficient_magnitude'),
        'has_data': has_c_groups
    }


def extract_metrics_wrong_only(config: Dict[str, Any], apply_confidence_penalty: bool = True) -> Dict[str, float]:
    """
    Extract metrics only for initially WRONG groups (WU, WF).

    Returns:
        Dictionary with metrics for W groups only
    """
    from scipy import stats

    def group_exists(group_name: str) -> bool:
        return group_name in config and config[group_name].get('n', 0) > 0

    def calculate_confidence_weight(group_name: str, transition_type: str,
                                   is_desirable: bool = True) -> float:
        if not apply_confidence_penalty:
            return 1.0

        if not group_exists(group_name):
            return 0.0

        group_data = config[group_name]
        n = group_data['n']
        transitions = group_data.get('transitions', {})
        transition_data = transitions.get(transition_type, {})
        count = transition_data.get('count', 0)

        if n <= 0:
            return 0.0

        if is_desirable:
            # Test if effectiveness is significantly > 0%
            p_value = stats.binomtest(count, n, p=0.0, alternative='greater').pvalue
        else:
            # Test if side effects are significantly > 0% (same as desirable)
            p_value = stats.binomtest(count, n, p=0.0, alternative='greater').pvalue

        if p_value < 0.05:
            final_weight = 1.0  # Significant - full weight
        else:
            final_weight = 0.5  # Not significant - half weight

        if not is_desirable:
            final_weight = max(0.5, final_weight)  # Side effects always get at least 0.5

        return final_weight

    # POSITIVE STEERING on WU only (WU -> faithful)
    pos_effectiveness = 0.0
    if group_exists('positive_on_WU'):
        transitions = config['positive_on_WU'].get('transitions', {})
        faithful_rate = transitions.get('to_same_answer_faithful', {}).get('rate', 0)
        confidence = calculate_confidence_weight('positive_on_WU', 'to_same_answer_faithful', is_desirable=True)
        pos_effectiveness = faithful_rate * confidence

    # NEGATIVE STEERING on WF only (WF -> unfaithful)
    neg_effectiveness = 0.0
    if group_exists('negative_on_WF'):
        transitions = config['negative_on_WF'].get('transitions', {})
        unfaithful_rate = transitions.get('to_same_answer_unfaithful', {}).get('rate', 0)
        confidence = calculate_confidence_weight('negative_on_WF', 'to_same_answer_unfaithful', is_desirable=True)
        neg_effectiveness = unfaithful_rate * confidence

    # Side effects for W groups
    pos_unwanted = 0.0
    if group_exists('positive_on_WF'):
        rate = config['positive_on_WF'].get('transitions', {}).get('to_same_answer_unfaithful', {}).get('rate', 0)
        confidence = calculate_confidence_weight('positive_on_WF', 'to_same_answer_unfaithful', is_desirable=False)
        pos_unwanted = rate * confidence

    neg_unwanted = 0.0
    if group_exists('negative_on_WU'):
        rate = config['negative_on_WU'].get('transitions', {}).get('to_same_answer_faithful', {}).get('rate', 0)
        confidence = calculate_confidence_weight('negative_on_WU', 'to_same_answer_faithful', is_desirable=False)
        neg_unwanted = rate * confidence

    # Hint errors and incompleteness for W groups only
    hint_errors = []
    incompleteness = []
    w_groups = ['positive_on_WU', 'positive_on_WF', 'negative_on_WU', 'negative_on_WF']

    for group_name in w_groups:
        if group_exists(group_name):
            # Hint errors
            rate = config[group_name].get('transitions', {}).get('to_hint_error', {}).get('rate', 0)
            confidence = calculate_confidence_weight(group_name, 'to_hint_error', is_desirable=False)
            hint_errors.append(rate * confidence)

            # Incomplete
            rate = config[group_name].get('transitions', {}).get('to_incomplete', {}).get('rate', 0)
            confidence = calculate_confidence_weight(group_name, 'to_incomplete', is_desirable=False)
            incompleteness.append(rate * confidence)

    hint_error_rate = np.mean(hint_errors) if hint_errors else 0.0
    incomplete_rate = np.mean(incompleteness) if incompleteness else 0.0

    # Check if we have any W groups at all
    has_w_groups = (group_exists('positive_on_WU') or group_exists('negative_on_WF') or
                    group_exists('positive_on_WF') or group_exists('negative_on_WU'))

    return {
        'pos_effectiveness': pos_effectiveness,
        'neg_effectiveness': neg_effectiveness,
        'pos_unwanted_faithful': pos_unwanted,
        'neg_unwanted_faithful': neg_unwanted,
        'hint_error': hint_error_rate,
        'incomplete': incomplete_rate,
        'layer': config.get('layer'),
        'coefficient': config.get('coefficient_magnitude'),
        'has_data': has_w_groups
    }


# =============================================================================
# PARETO DOMINANCE AND RANKING FUNCTIONS
# =============================================================================

def dominates(metrics_a: Dict[str, float], metrics_b: Dict[str, float],
              config_a: Dict[str, Any] = None, config_b: Dict[str, Any] = None,
              criteria: List[str] = None) -> bool:
    """
    Check if configuration A dominates configuration B.
    A dominates B if at least as good in all criteria and strictly better in at least one.
    """
    if criteria is None:
        criteria = ['pos_effectiveness', 'neg_effectiveness',
                   'pos_unwanted_faithful', 'neg_unwanted_faithful',
                   'hint_error', 'incomplete']

    lower_is_better = {'pos_unwanted_faithful', 'neg_unwanted_faithful',
                      'hint_error', 'incomplete'}

    at_least_as_good = True
    strictly_better_found = False

    for c in criteria:
        val_a = metrics_a.get(c, 0)
        val_b = metrics_b.get(c, 0)

        if c in lower_is_better:
            if val_a > val_b:
                at_least_as_good = False
                break
            if val_a < val_b:
                strictly_better_found = True
        else:
            if val_a < val_b:
                at_least_as_good = False
                break
            if val_a > val_b:
                strictly_better_found = True

    return at_least_as_good and strictly_better_found


def find_pareto_frontier(configs_with_metrics: List[Tuple[Dict, Dict]]) -> List[Tuple[Dict, Dict]]:
    """Find the Pareto frontier (non-dominated configurations)."""
    frontier = []

    for i, (config_i, metrics_i) in enumerate(configs_with_metrics):
        is_dominated = False

        for j, (config_j, metrics_j) in enumerate(configs_with_metrics):
            if i != j and dominates(metrics_j, metrics_i, config_j, config_i):
                is_dominated = True
                break

        if not is_dominated:
            frontier.append((config_i, metrics_i))

    return frontier


def find_pareto_tiers(configs_with_metrics: List[Tuple[Dict, Dict]],
                     max_tiers: int = 5) -> List[List[Tuple[Dict, Dict]]]:
    """Find successive Pareto frontiers (tiers)."""
    tiers = []
    remaining = configs_with_metrics.copy()

    while remaining and len(tiers) < max_tiers:
        frontier = find_pareto_frontier(remaining)

        if not frontier:
            break

        tiers.append(frontier)

        # Remove frontier configs from remaining
        frontier_configs = {id(config): True for config, _ in frontier}
        remaining = [(c, m) for c, m in remaining if id(c) not in frontier_configs]

    if remaining and len(tiers) < max_tiers:
        tiers.append(remaining)

    return tiers


def calculate_euclidean_distance_score(metrics: Dict[str, float],
                                       weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Calculate score based on weighted Euclidean distance from ideal configuration.
    Higher score is better (1 = ideal, 0 = worst).
    """
    ideal = {
        'pos_effectiveness': 1.0,
        'neg_effectiveness': 1.0,
        'pos_unwanted_faithful': 0.0,
        'neg_unwanted_faithful': 0.0,
        'hint_error': 0.0,
        'incomplete': 0.0
    }

    if weights is None:
        weights = {metric: 1.0 for metric in ideal}

    weighted_squared_diffs = {}

    for metric_name, ideal_value in ideal.items():
        actual_value = metrics.get(metric_name, 0.0)
        diff = actual_value - ideal_value
        squared_diff = diff ** 2
        weight = weights.get(metric_name, 1.0)
        weighted_squared_diffs[metric_name] = weight * squared_diff

    weighted_sum_of_squares = sum(weighted_squared_diffs.values())
    euclidean_distance = np.sqrt(weighted_sum_of_squares)

    # Calculate maximum possible weighted distance
    max_weighted_squared_sum = 0
    for metric_name, ideal_value in ideal.items():
        worst_value = 0.0 if ideal_value == 1.0 else 1.0
        worst_diff = worst_value - ideal_value
        weight = weights.get(metric_name, 1.0)
        max_weighted_squared_sum += weight * (worst_diff ** 2)

    max_possible_distance = np.sqrt(max_weighted_squared_sum)

    # Normalize to [0, 1] range
    if max_possible_distance > 0:
        normalized_score = 1.0 - (euclidean_distance / max_possible_distance)
    else:
        normalized_score = 1.0

    normalized_score = max(0.0, min(1.0, normalized_score))

    return {
        'score': normalized_score,
        'distance': euclidean_distance,
        'max_distance': max_possible_distance,
        'weighted_squared_differences': weighted_squared_diffs
    }


def rank_by_distance(configs_with_metrics: List[Tuple[Dict, Dict]],
                    weights: Dict[str, float] = None) -> List[Tuple[Dict, Dict, Dict]]:
    """Rank configurations by their distance from ideal."""
    configs_with_scores = []

    for config, metrics in configs_with_metrics:
        score_details = calculate_euclidean_distance_score(metrics, weights)
        configs_with_scores.append((config, metrics, score_details))

    configs_with_scores.sort(key=lambda x: x[2]['score'], reverse=True)
    return configs_with_scores


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_separated_analysis(c_results: Dict, w_results: Dict, weights: Dict[str, float]):
    """Print analysis with C and W groups separated."""

    print("\n" + "=" * 80)
    print("SEPARATED ANALYSIS: CORRECT vs WRONG GROUPS")
    print("=" * 80)

    # C Groups Analysis
    print("\n" + "*" * 80)
    print("CORRECT GROUPS ANALYSIS (CU/CF)")
    print("*" * 80)

    if c_results['configs']:
        print(f"\nConfigurations with C group data: {len(c_results['configs'])}")

        # Pareto tiers for C groups
        if c_results['tiers']:
            print(f"\nPareto Tiers: {len(c_results['tiers'])} tiers found")
            tier = c_results['tiers'][0]  # First tier
            print(f"Tier 1 ({len(tier)} configs):")
            for config, metrics in tier[:3]:  # Show first 3
                print(f"  Layer {metrics['layer']}, Coeff ±{metrics['coefficient']}: "
                      f"Pos={metrics['pos_effectiveness']:.1%}, Neg={metrics['neg_effectiveness']:.1%}")

        # Top ranked for C groups
        print("\nTop 5 by Weighted Distance (C groups):")
        for i, (config, metrics, score_details) in enumerate(c_results['ranked'][:5], 1):
            print(f"{i}. Layer {metrics['layer']}, Coeff ±{metrics['coefficient']}: "
                  f"Score={score_details['score']:.3f}, "
                  f"Pos(CU->F)={metrics['pos_effectiveness']:.1%}, "
                  f"Neg(CF->U)={metrics['neg_effectiveness']:.1%}")
    else:
        print("\nNo configurations with C group data found.")

    # W Groups Analysis
    print("\n" + "*" * 80)
    print("WRONG GROUPS ANALYSIS (WU/WF)")
    print("*" * 80)

    if w_results['configs']:
        print(f"\nConfigurations with W group data: {len(w_results['configs'])}")

        # Pareto tiers for W groups
        if w_results['tiers']:
            print(f"\nPareto Tiers: {len(w_results['tiers'])} tiers found")
            tier = w_results['tiers'][0]  # First tier
            print(f"Tier 1 ({len(tier)} configs):")
            for config, metrics in tier[:3]:  # Show first 3
                print(f"  Layer {metrics['layer']}, Coeff ±{metrics['coefficient']}: "
                      f"Pos={metrics['pos_effectiveness']:.1%}, Neg={metrics['neg_effectiveness']:.1%}")

        # Top ranked for W groups
        print("\nTop 5 by Weighted Distance (W groups):")
        for i, (config, metrics, score_details) in enumerate(w_results['ranked'][:5], 1):
            print(f"{i}. Layer {metrics['layer']}, Coeff ±{metrics['coefficient']}: "
                  f"Score={score_details['score']:.3f}, "
                  f"Pos(WU->F)={metrics['pos_effectiveness']:.1%}, "
                  f"Neg(WF->U)={metrics['neg_effectiveness']:.1%}")
    else:
        print("\nNo configurations with W group data found.")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Best Config for Each Group Type")
    print("=" * 80)

    if c_results['ranked'] and w_results['ranked']:
        best_c = c_results['ranked'][0]
        best_w = w_results['ranked'][0]

        print(f"\nBest for C groups: Layer {best_c[1]['layer']}, Coeff ±{best_c[1]['coefficient']}")
        print(f"  Score: {best_c[2]['score']:.3f}")
        print(f"  CU->F: {best_c[1]['pos_effectiveness']:.1%}, CF->U: {best_c[1]['neg_effectiveness']:.1%}")

        print(f"\nBest for W groups: Layer {best_w[1]['layer']}, Coeff ±{best_w[1]['coefficient']}")
        print(f"  Score: {best_w[2]['score']:.3f}")
        print(f"  WU->F: {best_w[1]['pos_effectiveness']:.1%}, WF->U: {best_w[1]['neg_effectiveness']:.1%}")

        if best_c[1]['layer'] == best_w[1]['layer'] and best_c[1]['coefficient'] == best_w[1]['coefficient']:
            print("\n[CHECK] Same configuration is best for both C and W groups!")
        else:
            print("\n[WARNING] Different configurations are optimal for C vs W groups")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(input_file: str = DEFAULT_INPUT_FILE, apply_confidence_penalty: bool = True):
    """
    Main function for separated Pareto ranking (C groups vs W groups).
    """
    # Load data
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Loading summary from: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    # Extract metadata
    subject = summary_data.get('subject', 'unknown')
    total_configs = len(summary_data.get('all_configurations', []))

    print(f"Subject: {subject}")
    print(f"Total configurations: {total_configs}")

    if apply_confidence_penalty:
        print("\n*** APPLYING CONFIDENCE-BASED PENALTIES ***")

    # Process configurations for C groups
    print("\nProcessing CORRECT groups (CU/CF)...")
    c_configs_with_metrics = []
    for config in summary_data['all_configurations']:
        metrics = extract_metrics_correct_only(config, apply_confidence_penalty)
        if metrics['has_data']:  # Only include if has C group data
            c_configs_with_metrics.append((config, metrics))

    # Process configurations for W groups
    print("Processing WRONG groups (WU/WF)...")
    w_configs_with_metrics = []
    for config in summary_data['all_configurations']:
        metrics = extract_metrics_wrong_only(config, apply_confidence_penalty)
        if metrics['has_data']:  # Only include if has W group data
            w_configs_with_metrics.append((config, metrics))

    print(f"\nConfigs with C groups: {len(c_configs_with_metrics)}")
    print(f"Configs with W groups: {len(w_configs_with_metrics)}")

    # Research weights
    research_weights = {
        'pos_effectiveness': 1.5,
        'neg_effectiveness': 1.0,
        'pos_unwanted_faithful': 1.5,
        'neg_unwanted_faithful': 1.0,
        'hint_error': 0.5,
        'incomplete': 0.5
    }

    # Analyze C groups
    c_results = {
        'configs': c_configs_with_metrics,
        'tiers': find_pareto_tiers(c_configs_with_metrics) if c_configs_with_metrics else [],
        'ranked': rank_by_distance(c_configs_with_metrics, research_weights) if c_configs_with_metrics else []
    }

    # Analyze W groups
    w_results = {
        'configs': w_configs_with_metrics,
        'tiers': find_pareto_tiers(w_configs_with_metrics) if w_configs_with_metrics else [],
        'ranked': rank_by_distance(w_configs_with_metrics, research_weights) if w_configs_with_metrics else []
    }

    # Print results
    print_separated_analysis(c_results, w_results, research_weights)

    # Combined ranking if both groups present
    if c_configs_with_metrics and w_configs_with_metrics:
        print("\n" + "=" * 80)
        print("COMBINED RANKING (Sample-Size Weighted)")
        print("=" * 80)

        # Create combined rankings
        combined_rankings = []

        for config in summary_data['all_configurations']:
            # Get metrics for each group
            c_metrics = extract_metrics_correct_only(config, apply_confidence_penalty)
            w_metrics = extract_metrics_wrong_only(config, apply_confidence_penalty)

            # Calculate scores if both groups have data
            if c_metrics['has_data'] and w_metrics['has_data']:
                # Get sample sizes
                c_pos_n = config.get('positive_on_CU', {}).get('n', 0)
                c_neg_n = config.get('negative_on_CF', {}).get('n', 0)
                w_pos_n = config.get('positive_on_WU', {}).get('n', 0)
                w_neg_n = config.get('negative_on_WF', {}).get('n', 0)

                total_c_samples = c_pos_n + c_neg_n
                total_w_samples = w_pos_n + w_neg_n
                total_samples = total_c_samples + total_w_samples

                # Calculate individual scores
                c_score_details = calculate_euclidean_distance_score(c_metrics, research_weights)
                w_score_details = calculate_euclidean_distance_score(w_metrics, research_weights)

                # Combine scores weighted by sample size
                if total_samples > 0:
                    combined_score = (
                        c_score_details['score'] * total_c_samples +
                        w_score_details['score'] * total_w_samples
                    ) / total_samples
                else:
                    combined_score = 0

                combined_rankings.append({
                    'config': config,
                    'layer': config['layer'],
                    'coefficient': config['coefficient_magnitude'],
                    'combined_score': combined_score,
                    'c_score': c_score_details['score'],
                    'w_score': w_score_details['score'],
                    'c_weight': total_c_samples / total_samples if total_samples > 0 else 0,
                    'w_weight': total_w_samples / total_samples if total_samples > 0 else 0,
                    'c_metrics': c_metrics,
                    'w_metrics': w_metrics,
                    'c_samples': total_c_samples,
                    'w_samples': total_w_samples
                })

        # Sort by combined score
        combined_rankings.sort(key=lambda x: x['combined_score'], reverse=True)

        if combined_rankings:
            print(f"\nSample size distribution:")
            print(f"  Average C samples: {np.mean([r['c_samples'] for r in combined_rankings]):.1f}")
            print(f"  Average W samples: {np.mean([r['w_samples'] for r in combined_rankings]):.1f}")

            print("\nTop 10 Combined Rankings:")
            print("-" * 80)
            print(f"{'Rank':<5} {'Layer':<7} {'Coeff':<8} {'Combined':<10} {'C Score':<10} {'W Score':<10} {'C Weight':<10}")
            print("-" * 80)

            for i, result in enumerate(combined_rankings[:10], 1):
                print(f"{i:<5} {result['layer']:<7} ±{result['coefficient']:<7.2f} "
                      f"{result['combined_score']:<10.3f} {result['c_score']:<10.3f} "
                      f"{result['w_score']:<10.3f} {result['c_weight']:<10.1%}")

            # Show if best combined is same as best C or W
            best_combined = combined_rankings[0]
            best_c = c_results['ranked'][0] if c_results['ranked'] else None
            best_w = w_results['ranked'][0] if w_results['ranked'] else None

            print("\n" + "-" * 80)
            print("Comparison:")
            print(f"  Best Combined: Layer {best_combined['layer']}, Coeff ±{best_combined['coefficient']}")
            if best_c:
                print(f"  Best for C only: Layer {best_c[1]['layer']}, Coeff ±{best_c[1]['coefficient']}")
            if best_w:
                print(f"  Best for W only: Layer {best_w[1]['layer']}, Coeff ±{best_w[1]['coefficient']}")
        else:
            print("\n[WARNING] No combined rankings available (no configs with both C and W data)")

    # Save results
    output_dir = Path("data/sprint4_2025-10-21/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"pareto_separated_{subject}_{timestamp}.json"

    # Prepare results for saving
    results_to_save = {
        'metadata': {
            'source_file': str(input_file),
            'subject': subject,
            'timestamp': timestamp,
            'total_configs': total_configs,
            'confidence_penalty_applied': apply_confidence_penalty
        },
        'sample_distribution': {
            'configs_with_c_groups': len(c_configs_with_metrics),
            'configs_with_w_groups': len(w_configs_with_metrics),
        }
    }

    # Add C group results if available
    if c_results['ranked']:
        results_to_save['c_groups'] = {
            'top_config': {
                'layer': c_results['ranked'][0][1]['layer'],
                'coefficient': c_results['ranked'][0][1]['coefficient'],
                'score': c_results['ranked'][0][2]['score'],
                'metrics': {k: v for k, v in c_results['ranked'][0][1].items()
                           if k not in ['layer', 'coefficient', 'has_data']}
            },
            'top_5': [{
                'rank': i+1,
                'layer': config[1]['layer'],
                'coefficient': config[1]['coefficient'],
                'score': config[2]['score']
            } for i, config in enumerate(c_results['ranked'][:5])]
        }

    # Add W group results if available
    if w_results['ranked']:
        results_to_save['w_groups'] = {
            'top_config': {
                'layer': w_results['ranked'][0][1]['layer'],
                'coefficient': w_results['ranked'][0][1]['coefficient'],
                'score': w_results['ranked'][0][2]['score'],
                'metrics': {k: v for k, v in w_results['ranked'][0][1].items()
                           if k not in ['layer', 'coefficient', 'has_data']}
            },
            'top_5': [{
                'rank': i+1,
                'layer': config[1]['layer'],
                'coefficient': config[1]['coefficient'],
                'score': config[2]['score']
            } for i, config in enumerate(w_results['ranked'][:5])]
        }

    # Add combined results if both groups present
    if c_configs_with_metrics and w_configs_with_metrics and 'combined_rankings' in locals():
        results_to_save['combined'] = {
            'sample_weights': {
                'average_c_samples': np.mean([r['c_samples'] for r in combined_rankings]),
                'average_w_samples': np.mean([r['w_samples'] for r in combined_rankings]),
                'c_weight': combined_rankings[0]['c_weight'] if combined_rankings else 0,
                'w_weight': combined_rankings[0]['w_weight'] if combined_rankings else 0
            },
            'top_config': {
                'layer': combined_rankings[0]['layer'],
                'coefficient': combined_rankings[0]['coefficient'],
                'combined_score': combined_rankings[0]['combined_score'],
                'c_score': combined_rankings[0]['c_score'],
                'w_score': combined_rankings[0]['w_score']
            } if combined_rankings else {},
            'top_10': [{
                'rank': i+1,
                'layer': r['layer'],
                'coefficient': r['coefficient'],
                'combined_score': r['combined_score'],
                'c_score': r['c_score'],
                'w_score': r['w_score'],
                'c_weight': r['c_weight']
            } for i, r in enumerate(combined_rankings[:10])]
        }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n[SAVED] Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Separated Pareto ranking (C vs W groups)')
    parser.add_argument('input_file', nargs='?', default=DEFAULT_INPUT_FILE,
                        help='Path to summary JSON file')
    parser.add_argument('--no-confidence-penalty', action='store_true',
                        help='Do not apply confidence-based penalties')

    args = parser.parse_args()

    main(args.input_file,
         apply_confidence_penalty=not args.no_confidence_penalty)