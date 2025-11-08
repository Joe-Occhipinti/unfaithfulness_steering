"""
generate_ranking_tables.py

Generates detailed tables from pareto ranking results, showing:
- Top 5 configs per ranking type (C-only, W-only, Combined, Overall)
- Raw transition rates with statistical significance markers
- Detailed breakdowns with weighted rates and p-values

Usage:
    python generate_ranking_tables.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_PARETO_FILE = "data/sprint4_2025-10-21/analysis/pareto_rankings_2025-10-27.json"
DEFAULT_SUMMARY_FILE = "data/sprint4_2025-10-21/summaries/steered_faithfulness/summary_faithfulness_steered_sprint4_2025-10-27.json"
OUTPUT_FILE = "data/sprint4_2025-10-21/analysis/ranking_tables_2025-10-27.txt"


def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_config_in_summary(summary: Dict, hint_template: str, layer: int,
                           coefficient: float) -> Dict:
    """
    Find configuration data in summary file.

    Returns the full config dict with all group data.
    """
    configs = summary['configurations_by_hint'].get(hint_template, [])

    for config in configs:
        if config['layer'] == layer and abs(config['coefficient_magnitude'] - coefficient) < 0.001:
            return config

    return None


def compute_pvalue(n: int, count: int) -> float:
    """
    Compute p-value using binomial test.
    H0: p = 0.0 (no transitions without steering, temp=0)
    """
    if n <= 0:
        return 1.0
    return stats.binomtest(count, n, p=0.0, alternative='greater').pvalue


def get_significance_marker(p_value: float) -> str:
    """Convert p-value to significance marker."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def extract_c_metrics(config: Dict) -> Dict[str, Dict[str, float]]:
    """
    Extract C-group metrics with raw rates and p-values.

    Returns:
        {
            'pos_effectiveness': {'rate': 0.456, 'count': 26, 'n': 57, 'p_value': 0.0003},
            ...
        }
    """
    metrics = {}

    # pos_effectiveness: positive on CU -> faithful
    if 'positive_on_CU' in config:
        group = config['positive_on_CU']
        trans = group.get('transitions', {}).get('to_same_answer_faithful', {})
        metrics['pos_effectiveness'] = {
            'rate': trans.get('rate', 0),
            'count': trans.get('count', 0),
            'n': group.get('n', 0),
            'p_value': compute_pvalue(group.get('n', 0), trans.get('count', 0))
        }

    # neg_effectiveness: negative on CF -> unfaithful
    if 'negative_on_CF' in config:
        group = config['negative_on_CF']
        trans = group.get('transitions', {}).get('to_same_answer_unfaithful', {})
        metrics['neg_effectiveness'] = {
            'rate': trans.get('rate', 0),
            'count': trans.get('count', 0),
            'n': group.get('n', 0),
            'p_value': compute_pvalue(group.get('n', 0), trans.get('count', 0))
        }

    # pos_unwanted: positive on CF -> unfaithful (side effect)
    if 'positive_on_CF' in config:
        group = config['positive_on_CF']
        trans = group.get('transitions', {}).get('to_same_answer_unfaithful', {})
        metrics['pos_unwanted'] = {
            'rate': trans.get('rate', 0),
            'count': trans.get('count', 0),
            'n': group.get('n', 0),
            'p_value': compute_pvalue(group.get('n', 0), trans.get('count', 0))
        }

    # neg_unwanted: negative on CU -> faithful (side effect)
    if 'negative_on_CU' in config:
        group = config['negative_on_CU']
        trans = group.get('transitions', {}).get('to_same_answer_faithful', {})
        metrics['neg_unwanted'] = {
            'rate': trans.get('rate', 0),
            'count': trans.get('count', 0),
            'n': group.get('n', 0),
            'p_value': compute_pvalue(group.get('n', 0), trans.get('count', 0))
        }

    # hint_error and incomplete: averaged across all C groups
    c_groups = ['positive_on_CU', 'negative_on_CU', 'positive_on_CF', 'negative_on_CF']

    hint_errors = []
    hint_error_counts = []
    hint_error_ns = []

    incompleteness = []
    incomplete_counts = []
    incomplete_ns = []

    for group_name in c_groups:
        if group_name in config:
            group = config[group_name]
            n = group.get('n', 0)

            # Hint errors
            trans = group.get('transitions', {}).get('to_hint_error', {})
            hint_errors.append(trans.get('rate', 0))
            hint_error_counts.append(trans.get('count', 0))
            hint_error_ns.append(n)

            # Incomplete
            trans = group.get('transitions', {}).get('to_incomplete', {})
            incompleteness.append(trans.get('rate', 0))
            incomplete_counts.append(trans.get('count', 0))
            incomplete_ns.append(n)

    # Average metrics
    total_hint_n = sum(hint_error_ns)
    total_hint_count = sum(hint_error_counts)
    metrics['hint_error'] = {
        'rate': np.mean(hint_errors) if hint_errors else 0,
        'count': total_hint_count,
        'n': total_hint_n,
        'p_value': compute_pvalue(total_hint_n, total_hint_count)
    }

    total_incomplete_n = sum(incomplete_ns)
    total_incomplete_count = sum(incomplete_counts)
    metrics['incomplete'] = {
        'rate': np.mean(incompleteness) if incompleteness else 0,
        'count': total_incomplete_count,
        'n': total_incomplete_n,
        'p_value': compute_pvalue(total_incomplete_n, total_incomplete_count)
    }

    return metrics


def extract_w_metrics(config: Dict) -> Dict[str, Dict[str, float]]:
    """
    Extract W-group metrics with raw rates and p-values.
    Similar to extract_c_metrics but for W groups, including to_correct.
    """
    metrics = {}

    # pos_effectiveness: positive on WU -> faithful
    if 'positive_on_WU' in config:
        group = config['positive_on_WU']
        trans = group.get('transitions', {}).get('to_same_answer_faithful', {})
        metrics['pos_effectiveness'] = {
            'rate': trans.get('rate', 0),
            'count': trans.get('count', 0),
            'n': group.get('n', 0),
            'p_value': compute_pvalue(group.get('n', 0), trans.get('count', 0))
        }

    # neg_effectiveness: negative on WF -> unfaithful
    if 'negative_on_WF' in config:
        group = config['negative_on_WF']
        trans = group.get('transitions', {}).get('to_same_answer_unfaithful', {})
        metrics['neg_effectiveness'] = {
            'rate': trans.get('rate', 0),
            'count': trans.get('count', 0),
            'n': group.get('n', 0),
            'p_value': compute_pvalue(group.get('n', 0), trans.get('count', 0))
        }

    # pos_unwanted: positive on WF -> unfaithful (side effect)
    if 'positive_on_WF' in config:
        group = config['positive_on_WF']
        trans = group.get('transitions', {}).get('to_same_answer_unfaithful', {})
        metrics['pos_unwanted'] = {
            'rate': trans.get('rate', 0),
            'count': trans.get('count', 0),
            'n': group.get('n', 0),
            'p_value': compute_pvalue(group.get('n', 0), trans.get('count', 0))
        }

    # neg_unwanted: negative on WU -> faithful (side effect)
    if 'negative_on_WU' in config:
        group = config['negative_on_WU']
        trans = group.get('transitions', {}).get('to_same_answer_faithful', {})
        metrics['neg_unwanted'] = {
            'rate': trans.get('rate', 0),
            'count': trans.get('count', 0),
            'n': group.get('n', 0),
            'p_value': compute_pvalue(group.get('n', 0), trans.get('count', 0))
        }

    # hint_error, incomplete, to_correct: averaged across all W groups
    w_groups = ['positive_on_WU', 'negative_on_WU', 'positive_on_WF', 'negative_on_WF']

    hint_errors = []
    hint_error_counts = []
    hint_error_ns = []

    incompleteness = []
    incomplete_counts = []
    incomplete_ns = []

    to_corrects = []
    to_correct_counts = []
    to_correct_ns = []

    for group_name in w_groups:
        if group_name in config:
            group = config[group_name]
            n = group.get('n', 0)

            # Hint errors
            trans = group.get('transitions', {}).get('to_hint_error', {})
            hint_errors.append(trans.get('rate', 0))
            hint_error_counts.append(trans.get('count', 0))
            hint_error_ns.append(n)

            # Incomplete
            trans = group.get('transitions', {}).get('to_incomplete', {})
            incompleteness.append(trans.get('rate', 0))
            incomplete_counts.append(trans.get('count', 0))
            incomplete_ns.append(n)

            # To correct
            trans = group.get('transitions', {}).get('to_correct', {})
            to_corrects.append(trans.get('rate', 0))
            to_correct_counts.append(trans.get('count', 0))
            to_correct_ns.append(n)

    # Average metrics
    total_hint_n = sum(hint_error_ns)
    total_hint_count = sum(hint_error_counts)
    metrics['hint_error'] = {
        'rate': np.mean(hint_errors) if hint_errors else 0,
        'count': total_hint_count,
        'n': total_hint_n,
        'p_value': compute_pvalue(total_hint_n, total_hint_count)
    }

    total_incomplete_n = sum(incomplete_ns)
    total_incomplete_count = sum(incomplete_counts)
    metrics['incomplete'] = {
        'rate': np.mean(incompleteness) if incompleteness else 0,
        'count': total_incomplete_count,
        'n': total_incomplete_n,
        'p_value': compute_pvalue(total_incomplete_n, total_incomplete_count)
    }

    total_correct_n = sum(to_correct_ns)
    total_correct_count = sum(to_correct_counts)
    metrics['to_correct'] = {
        'rate': np.mean(to_corrects) if to_corrects else 0,
        'count': total_correct_count,
        'n': total_correct_n,
        'p_value': compute_pvalue(total_correct_n, total_correct_count)
    }

    return metrics


def print_c_ranking_table(hint: str, rankings: Dict, summary: Dict):
    """Print C-only ranking table for a hint."""
    print("\n" + "=" * 80)
    print(f"C-ONLY RANKING: {hint}")
    print("=" * 80)
    print("\nTop 5 Configurations for Initially Correct Groups (CU/CF)")
    print("-" * 80)
    print(f"{'Rank':<5} {'Layer':<7} {'Coeff':<8} {'Score':<8} {'pos_eff':<12} {'neg_eff':<12} {'pos_unwant':<12} {'neg_unwant':<12} {'hint_err':<11} {'incomp':<11}")
    print("-" * 80)

    top_5 = rankings['results_by_hint'][hint]['c_groups']['top_5']

    for rank_info in top_5:
        rank = rank_info['rank']
        layer = rank_info['layer']
        coeff = rank_info['coefficient']
        score = rank_info['score']

        # Find config in summary
        config = find_config_in_summary(summary, hint, layer, coeff)
        if not config:
            print(f"{rank:<5} {layer:<7} ±{coeff:<7.2f} {score:<8.3f} [DATA NOT FOUND]")
            continue

        # Extract metrics
        metrics = extract_c_metrics(config)

        # Format row
        def fmt(m):
            if m not in metrics:
                return "-"
            rate = metrics[m]['rate']
            sig = get_significance_marker(metrics[m]['p_value'])
            return f"{rate:.3f}{sig}"

        print(f"{rank:<5} {layer:<7} ±{coeff:<7.2f} {score:<8.3f} "
              f"{fmt('pos_effectiveness'):<12} {fmt('neg_effectiveness'):<12} "
              f"{fmt('pos_unwanted'):<12} {fmt('neg_unwanted'):<12} "
              f"{fmt('hint_error'):<11} {fmt('incomplete'):<11}")

    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    print("Values shown: raw_rate + significance marker")

    # Detailed breakdown for rank 1
    if top_5:
        rank1 = top_5[0]
        config = find_config_in_summary(summary, hint, rank1['layer'], rank1['coefficient'])
        if config:
            print(f"\nDETAILED METRICS FOR RANK 1 (Layer {rank1['layer']}, Coeff {rank1['coefficient']}):")
            print("-" * 80)
            print(f"{'Metric':<20} {'Raw Rate':<12} {'Count/N':<15} {'P-value':<12} {'Sig':<8}")
            print("-" * 80)

            metrics = extract_c_metrics(config)
            metric_names = ['pos_effectiveness', 'neg_effectiveness', 'pos_unwanted',
                          'neg_unwanted', 'hint_error', 'incomplete']

            for name in metric_names:
                if name in metrics:
                    m = metrics[name]
                    print(f"{name:<20} {m['rate']:<12.3f} {m['count']}/{m['n']:<12} "
                          f"{m['p_value']:<12.4f} {get_significance_marker(m['p_value']):<8}")


def print_w_ranking_table(hint: str, rankings: Dict, summary: Dict):
    """Print W-only ranking table for a hint."""
    print("\n" + "=" * 80)
    print(f"W-ONLY RANKING: {hint}")
    print("=" * 80)
    print("\nTop 5 Configurations for Initially Wrong Groups (WU/WF)")
    print("-" * 80)
    print(f"{'Rank':<5} {'Layer':<7} {'Coeff':<8} {'Score':<8} {'pos_eff':<12} {'neg_eff':<12} {'pos_unwant':<12} {'neg_unwant':<12} {'hint_err':<11} {'incomp':<11} {'to_correct':<11}")
    print("-" * 80)

    top_5 = rankings['results_by_hint'][hint]['w_groups']['top_5']

    for rank_info in top_5:
        rank = rank_info['rank']
        layer = rank_info['layer']
        coeff = rank_info['coefficient']
        score = rank_info['score']

        config = find_config_in_summary(summary, hint, layer, coeff)
        if not config:
            print(f"{rank:<5} {layer:<7} ±{coeff:<7.2f} {score:<8.3f} [DATA NOT FOUND]")
            continue

        metrics = extract_w_metrics(config)

        def fmt(m):
            if m not in metrics:
                return "-"
            rate = metrics[m]['rate']
            sig = get_significance_marker(metrics[m]['p_value'])
            return f"{rate:.3f}{sig}"

        print(f"{rank:<5} {layer:<7} ±{coeff:<7.2f} {score:<8.3f} "
              f"{fmt('pos_effectiveness'):<12} {fmt('neg_effectiveness'):<12} "
              f"{fmt('pos_unwanted'):<12} {fmt('neg_unwanted'):<12} "
              f"{fmt('hint_error'):<11} {fmt('incomplete'):<11} {fmt('to_correct'):<11}")

    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    print("Values shown: raw_rate + significance marker")

    # Detailed breakdown for rank 1
    if top_5:
        rank1 = top_5[0]
        config = find_config_in_summary(summary, hint, rank1['layer'], rank1['coefficient'])
        if config:
            print(f"\nDETAILED METRICS FOR RANK 1 (Layer {rank1['layer']}, Coeff {rank1['coefficient']}):")
            print("-" * 80)
            print(f"{'Metric':<20} {'Raw Rate':<12} {'Count/N':<15} {'P-value':<12} {'Sig':<8}")
            print("-" * 80)

            metrics = extract_w_metrics(config)
            metric_names = ['pos_effectiveness', 'neg_effectiveness', 'pos_unwanted',
                          'neg_unwanted', 'hint_error', 'incomplete', 'to_correct']

            for name in metric_names:
                if name in metrics:
                    m = metrics[name]
                    print(f"{name:<20} {m['rate']:<12.3f} {m['count']}/{m['n']:<12} "
                          f"{m['p_value']:<12.4f} {get_significance_marker(m['p_value']):<8}")


def print_combined_ranking_table(hint: str, rankings: Dict, summary: Dict):
    """Print combined C+W ranking table for a hint."""
    print("\n" + "=" * 80)
    print(f"COMBINED (C+W) RANKING: {hint}")
    print("=" * 80)
    print("\nTop 5 Configurations (Good at Both C and W Groups)")
    print("-" * 80)
    print(f"{'Rank':<5} {'Layer':<7} {'Coeff':<8} {'Combined':<10} {'C_Score':<10} {'W_Score':<10}")
    print("-" * 80)

    top_10 = rankings['results_by_hint'][hint]['combined']['top_10']

    for rank_info in top_10[:5]:  # Only show top 5
        rank = rank_info['rank']
        layer = rank_info['layer']
        coeff = rank_info['coefficient']
        combined = rank_info['combined_score']
        c_score = rank_info['c_score']
        w_score = rank_info['w_score']

        print(f"{rank:<5} {layer:<7} ±{coeff:<7.2f} {combined:<10.3f} {c_score:<10.3f} {w_score:<10.3f}")

    # Detailed breakdown for rank 1
    if top_10:
        rank1 = top_10[0]
        config = find_config_in_summary(summary, hint, rank1['layer'], rank1['coefficient'])
        if config:
            print(f"\nDETAILED BREAKDOWN FOR RANK 1 (Layer {rank1['layer']}, Coeff {rank1['coefficient']}):")
            print("-" * 80)

            print("\nC-GROUP METRICS:")
            print(f"{'Metric':<20} {'Raw Rate':<12} {'Count/N':<15} {'P-value':<12} {'Sig':<8}")
            print("-" * 40)

            c_metrics = extract_c_metrics(config)
            for name in ['pos_effectiveness', 'neg_effectiveness', 'pos_unwanted',
                        'neg_unwanted', 'hint_error', 'incomplete']:
                if name in c_metrics:
                    m = c_metrics[name]
                    print(f"{name:<20} {m['rate']:<12.3f} {m['count']}/{m['n']:<12} "
                          f"{m['p_value']:<12.4f} {get_significance_marker(m['p_value']):<8}")

            print("\nW-GROUP METRICS:")
            print(f"{'Metric':<20} {'Raw Rate':<12} {'Count/N':<15} {'P-value':<12} {'Sig':<8}")
            print("-" * 40)

            w_metrics = extract_w_metrics(config)
            for name in ['pos_effectiveness', 'neg_effectiveness', 'pos_unwanted',
                        'neg_unwanted', 'hint_error', 'incomplete', 'to_correct']:
                if name in w_metrics:
                    m = w_metrics[name]
                    print(f"{name:<20} {m['rate']:<12.3f} {m['count']}/{m['n']:<12} "
                          f"{m['p_value']:<12.4f} {get_significance_marker(m['p_value']):<8}")


def print_overall_ranking_table(rankings: Dict, summary: Dict):
    """Print overall ranking table across all hints."""
    if 'overall_ranking' not in rankings:
        return

    print("\n" + "=" * 80)
    print("OVERALL RANKING (ACROSS ALL HINTS)")
    print("=" * 80)
    print("\nTop 5 Configurations (Best Average Performance Across All Hints)")
    print("-" * 80)

    hint_templates = rankings['metadata']['hint_templates']
    header = f"{'Rank':<5} {'Layer':<7} {'Coeff':<8} {'Overall':<10}"
    for h in hint_templates:
        header += f" {h[:12]:<12}"
    print(header)
    print("-" * 80)

    top_10 = rankings['overall_ranking']['top_10']

    for rank_info in top_10[:5]:  # Only show top 5
        rank = rank_info['rank']
        layer = rank_info['layer']
        coeff = rank_info['coefficient']
        overall = rank_info['overall_score']

        row = f"{rank:<5} {layer:<7} ±{coeff:<7.2f} {overall:<10.3f}"
        for h in hint_templates:
            score = rank_info['scores_by_hint'].get(h, 0)
            row += f" {score:<12.3f}"
        print(row)

    # Detailed breakdown for rank 1
    if top_10:
        rank1 = top_10[0]
        print(f"\nDETAILED BREAKDOWN FOR RANK 1 (Layer {rank1['layer']}, Coeff {rank1['coefficient']}):")
        print("-" * 80)

        for hint in hint_templates:
            combined_score = rank1['scores_by_hint'][hint]
            print(f"\nHINT: {hint} (combined_score: {combined_score:.3f})")

            # Get C and W scores from per-hint results
            hint_results = rankings['results_by_hint'][hint]

            # Find this config in the combined rankings for this hint
            combined_rankings = hint_results['combined']['top_10']
            matching = [r for r in combined_rankings if r['layer'] == rank1['layer']
                       and abs(r['coefficient'] - rank1['coefficient']) < 0.001]

            if matching:
                c_score = matching[0]['c_score']
                w_score = matching[0]['w_score']
                print(f"  C-Score: {c_score:.3f}  |  W-Score: {w_score:.3f}")

                # Get key metrics
                config = find_config_in_summary(summary, hint, rank1['layer'], rank1['coefficient'])
                if config:
                    c_metrics = extract_c_metrics(config)
                    w_metrics = extract_w_metrics(config)

                    # Show key C metrics
                    c_parts = []
                    for m in ['pos_effectiveness', 'neg_effectiveness', 'hint_error']:
                        if m in c_metrics:
                            rate = c_metrics[m]['rate']
                            sig = get_significance_marker(c_metrics[m]['p_value'])
                            c_parts.append(f"{m.replace('_', '')}={rate:.3f}{sig}")
                    print(f"  Key C Metrics: {', '.join(c_parts)}")

                    # Show key W metrics
                    w_parts = []
                    for m in ['pos_effectiveness', 'neg_effectiveness', 'to_correct']:
                        if m in w_metrics:
                            rate = w_metrics[m]['rate']
                            sig = get_significance_marker(w_metrics[m]['p_value'])
                            w_parts.append(f"{m.replace('_', '')}={rate:.3f}{sig}")
                    print(f"  Key W Metrics: {', '.join(w_parts)}")


def main(pareto_file: str, summary_file: str):
    """Main function."""
    import sys
    from io import StringIO

    print("=" * 80)
    print("GENERATING DETAILED RANKING TABLES")
    print("=" * 80)
    print(f"\nLoading pareto rankings from: {pareto_file}")
    print(f"Loading summary data from: {summary_file}")

    rankings = load_json(pareto_file)
    summary = load_json(summary_file)

    # Redirect stdout to capture all print statements
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    # Generate tables for each hint
    for hint in rankings['metadata']['hint_templates']:
        if hint in rankings['results_by_hint']:
            hint_results = rankings['results_by_hint'][hint]

            # C-only ranking
            if 'c_groups' in hint_results:
                print_c_ranking_table(hint, rankings, summary)

            # W-only ranking
            if 'w_groups' in hint_results:
                print_w_ranking_table(hint, rankings, summary)

            # Combined ranking
            if 'combined' in hint_results:
                print_combined_ranking_table(hint, rankings, summary)

    # Overall ranking
    print_overall_ranking_table(rankings, summary)

    # Restore stdout
    sys.stdout = old_stdout
    output_text = captured_output.getvalue()

    # Print to console
    print(output_text)

    # Save to file
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED RANKING TABLES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated from:\n")
        f.write(f"  Pareto rankings: {pareto_file}\n")
        f.write(f"  Summary data: {summary_file}\n")
        f.write(f"  Date: {Path(OUTPUT_FILE).stem.split('_')[-1]}\n")
        f.write("=" * 80 + "\n\n")
        f.write(output_text)

    print("\n" + "=" * 80)
    print("TABLE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTables saved to: {output_path}")


if __name__ == "__main__":
    main(DEFAULT_PARETO_FILE, DEFAULT_SUMMARY_FILE)
