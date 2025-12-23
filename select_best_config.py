"""
Select best steering configuration per hint template PER DATASET.

A configuration is (layer, |strength|) with bidirectional steering on UNFAITHFUL answers:
- Positive steering (+strength): should INCREASE correct + hint mentioning
- Negative steering (-strength): should DECREASE hint mentioning

Scoring formula:
    score = unfaithful_pos.correct_mentioning_hint_pct / (
        unfaithful_neg.correct_mentioning_hint_pct +
        sum(bad_metrics_from_both_records) + epsilon
    )

Best config = argmax(score) per hint_template per dataset.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Summary files to analyze
SUMMARY_FILES = [
    "data/definitive_pipeline_data/mentioned_annotated_steered_val_off_policy_2nd_2025-12-20_summary.json",
    "data/definitive_pipeline_data/mentioned_annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_summary.json",
    "data/definitive_pipeline_data/mentioned_annotated_steered_val_gradient_2hidden8_2025-12-06_summary.json",
    "data/definitive_pipeline_data/mentioned_annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25_summary.json"
]

# Small epsilon to avoid division by zero
EPSILON = 1e-6


def load_summary(filepath: str) -> Dict[str, Dict]:
    """Load a single summary file."""
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️  Skipping {filepath} (not found)")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_config_pairs(summary: Dict[str, Dict]) -> Dict[Tuple, Dict]:
    """
    Group records by bidirectional config: (layer, |coefficient|, hint_template).
    
    Returns dict mapping config_key to:
    {
        'unfaithful_pos': record with +coeff, unfaithful
        'unfaithful_neg': record with -coeff, unfaithful  
        'faithful_pos': record with +coeff, faithful
        'faithful_neg': record with -coeff, faithful
    }
    """
    configs = defaultdict(dict)
    
    for key, record in summary.items():
        layer = record['layer']
        coeff = record['coefficient']
        hint = record['hint_template']
        faith = record['original_faithfulness']
        
        # Config key is (layer, |coeff|, hint)
        strength = abs(coeff)
        config_key = (layer, strength, hint)
        
        # Classify this record
        if coeff > 0 and faith == 'unfaithful':
            configs[config_key]['unfaithful_pos'] = record
        elif coeff > 0 and faith == 'faithful':
            configs[config_key]['faithful_pos'] = record
        elif coeff < 0 and faith == 'unfaithful':
            configs[config_key]['unfaithful_neg'] = record
        elif coeff < 0 and faith == 'faithful':
            configs[config_key]['faithful_neg'] = record
    
    return dict(configs)


def compute_score(config_data: Dict) -> float:
    """
    Compute score for a bidirectional config on UNFAITHFUL answers.
    
    score = unfaithful_pos.correct_mentioning_hint / (
        unfaithful_neg.correct_mentioning_hint +
        unfaithful_pos.(incomplete + hint_induced_errors) +
        unfaithful_neg.(incomplete_mentioning + hint_induced_errors_mentioning) +
        epsilon
    )
    
    Rationale:
    - Numerator: Positive steering makes unfaithful → correct + mentions hint (GOOD)
    - Denominator part 1: Negative steering fails to suppress hint mention (BAD)
    - Denominator part 2: Positive steering causes failures (BAD, even if no mention)
    - Denominator part 3: Negative steering causes failures WHILE mentioning hint (DOUBLY BAD)
    """
    unfaithful_pos = config_data.get('unfaithful_pos', {})
    unfaithful_neg = config_data.get('unfaithful_neg', {})
    
    if not unfaithful_pos or not unfaithful_neg:
        return 0  # Incomplete pair
    
    # Numerator: good outcome from positive steering
    numerator = unfaithful_pos.get('correct_mentioning_hint_pct', 0)
    
    # Denominator components:
    # 1. Negative steering fails to suppress hint mentioning (correct but still mentions)
    neg_still_mentions = unfaithful_neg.get('correct_mentioning_hint_pct', 0)
    
    # 2. Positive steering causes failures (incomplete or errors, regardless of mention)
    pos_failures = (
        unfaithful_pos.get('incomplete_pct', 0) +
        unfaithful_pos.get('hint_induced_errors_pct', 0)
    )
    
    # 3. Negative steering causes failures WHILE mentioning hint
    neg_failures_mentioning = (
        unfaithful_neg.get('incomplete_mentioning_hint_pct', 0) +
        unfaithful_neg.get('hint_induced_errors_mentioning_hint_pct', 0)
    )
    
    denominator = neg_still_mentions + pos_failures + neg_failures_mentioning + EPSILON
    
    return numerator / denominator


def find_best_per_hint(configs: Dict[Tuple, Dict]) -> Dict[str, Dict]:
    """Find best config per hint_template."""
    # Group by hint_template
    by_hint = defaultdict(list)
    
    for (layer, strength, hint), data in configs.items():
        score = compute_score(data)
        by_hint[hint].append({
            'layer': layer,
            'strength': strength,
            'score': score,
            'data': data
        })
    
    # Find best per hint
    best = {}
    for hint, candidates in by_hint.items():
        if not candidates:
            continue
        
        # Sort by score descending
        sorted_cands = sorted(candidates, key=lambda x: x['score'], reverse=True)
        winner = sorted_cands[0]
        
        unfaithful_pos = winner['data'].get('unfaithful_pos', {})
        unfaithful_neg = winner['data'].get('unfaithful_neg', {})
        
        best[hint] = {
            'layer': winner['layer'],
            'strength': winner['strength'],
            'score': round(winner['score'], 4),
            'unfaithful_pos_correct_mentioning': unfaithful_pos.get('correct_mentioning_hint_pct', 0),
            'unfaithful_neg_correct_mentioning': unfaithful_neg.get('correct_mentioning_hint_pct', 0),
            'unfaithful_pos_correct': unfaithful_pos.get('correct_pct', 0),
            'unfaithful_neg_correct': unfaithful_neg.get('correct_pct', 0),
            'sample_size_unfaithful_pos': unfaithful_pos.get('total_records', 0),
            'sample_size_unfaithful_neg': unfaithful_neg.get('total_records', 0),
            'top_3': [
                {'layer': c['layer'], 'strength': c['strength'], 'score': round(c['score'], 4)}
                for c in sorted_cands[:3]
            ]
        }
    
    return best


def process_dataset(filepath: str) -> Tuple[str, Dict[str, Dict]]:
    """Process a single dataset and return best configs."""
    dataset_name = Path(filepath).stem.replace("mentioned_", "").replace("_summary", "")
    
    summary = load_summary(filepath)
    if not summary:
        return dataset_name, {}
    
    configs = build_config_pairs(summary)
    best = find_best_per_hint(configs)
    
    return dataset_name, best


def print_results(all_results: Dict[str, Dict[str, Dict]]):
    """Print results for all datasets."""
    print("\n" + "="*90)
    print("BEST STEERING CONFIGS PER HINT TEMPLATE PER DATASET")
    print("="*90)
    print("\nScoring: correct_mentioning_hint(unfaithful+pos) / (correct_mentioning_hint(unfaithful-neg) + bad_metrics)")
    
    for dataset_name, best_configs in all_results.items():
        print(f"\n{'─'*90}")
        print(f"📊 DATASET: {dataset_name}")
        print(f"{'─'*90}")
        
        if not best_configs:
            print("   No valid configs found")
            continue
        
        for hint, config in sorted(best_configs.items()):
            print(f"\n  📌 {hint}")
            print(f"     Best: Layer={config['layer']}, Strength=±{config['strength']}")
            print(f"     Score: {config['score']}")
            print(f"     Unfaithful+pos → correct_mentioning: {config['unfaithful_pos_correct_mentioning']:.1f}%")
            print(f"     Unfaithful-neg → correct_mentioning: {config['unfaithful_neg_correct_mentioning']:.1f}% (should be low)")
            print(f"     Sample sizes: {config['sample_size_unfaithful_pos']} / {config['sample_size_unfaithful_neg']}")
            print(f"     Top 3: {config['top_3']}")


def main():
    print("Processing datasets...")
    
    all_results = {}
    for filepath in SUMMARY_FILES:
        dataset_name, best = process_dataset(filepath)
        all_results[dataset_name] = best
        print(f"  ✓ {dataset_name}: {len(best)} hint templates")
    
    print_results(all_results)
    
    # Save results
    output_path = Path("data/definitive_pipeline_data/best_configs_per_dataset.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
