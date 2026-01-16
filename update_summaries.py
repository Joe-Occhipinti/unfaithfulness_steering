
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import asyncio

# Fix path to include src
sys.path.append(os.getcwd())

from src.data import load_jsonl
from src.steered_global_faithfulness import (
    group_records_by_config,
    compute_config_metrics
)
from src.config import ModelConfig

# Determine input relationships
# We need to find the annotated jsonl files and regenerate their corresponding summary json files.

def regenerate_summary_for_file(annotated_path: Path):
    print(f"Processing {annotated_path.name}...")
    
    # 1. Load records
    records = load_jsonl(annotated_path)
    
    if not records:
        print(f"  No records found in {annotated_path}")
        return

    # 2. Group by configuration
    # Note: group_records_by_config uses 'steering_layer' etc.
    grouped = group_records_by_config(records)
    
    # 3. Compute metrics
    # We need to reconstruct the "all_configs" list that the summary generator expects.
    # The compute_config_metrics function in src.steered_global_faithfulness does this.
    # It usually runs classification, BUT here we want to use the EXISTING classifications
    # in the annotated_records.
    
    # PROBLEM: compute_config_metrics calls compute_group_metrics_async which RUNS CLASSIFICATION.
    # We do NOT want to re-run LLM classification. We want to aggregate existing results.
    
    # We need a custom aggregator that just reads 'rule_classification', 'faithfulness', 'hint_mentioned'
    # from the records and computes transitions.
    
    all_configs = []
    
    # Sort keys for deterministic output
    sorted_keys = sorted(grouped.keys())
    
    for (hint_template, layer, coeff_mag) in sorted_keys:
        config_groups = grouped[(hint_template, layer, coeff_mag)]
        
        config_result = {
            'hint_template': hint_template,
            'layer': layer,
            'coefficient_magnitude': coeff_mag
        }
        
        # Process each group (positive_on_CF, etc.)
        for group_name in ['positive_on_CF', 'positive_on_CU', 'positive_on_WF', 'positive_on_WU',
                           'negative_on_CF', 'negative_on_CU', 'negative_on_WF', 'negative_on_WU']:
            
            group_records = config_groups.get(group_name, [])
            n = len(group_records)
            
            if n == 0:
                config_result[group_name] = {'n': 0, 'transitions': {}, 'classifications': {}}
                continue
                
            # Compute transitions cheaply from EXISTING fields
            # We need to reconstruct the "rule_classifications", "faithfulness_results", "hint_mention_results" maps
            # that compute_transitions expects.
            
            rule_map = {}
            faith_map = {}
            hint_map = {}
            
            parts = group_name.split('_on_')
            initial_state = parts[1]
            
            from src.steered_global_faithfulness import get_record_id, compute_transitions
            
            for r in group_records:
                qid = get_record_id(r)
                rule_map[qid] = r.get('rule_classification', 'error')
                faith_map[qid] = r.get('faithfulness')
                hint_map[qid] = r.get('hint_mentioned')
            
            transitions = compute_transitions(
                group_records,
                rule_map,
                faith_map,
                hint_map,
                initial_state
            )
            
            config_result[group_name] = {
                'n': n,
                'transitions': transitions,
                # 'classifications': ... (we don't strictly need this in summary, but nice to have)
            }
            
        all_configs.append(config_result)

    # 4. Create summary object
    # We try to infer metadata from filename or first record
    subject = annotated_path.parent.name # e.g. Qwen3-32B
    
    # Hint template is mixed, so we use 'mixed' or pass None
    # effectively create_summary just structures it.
    
    summary = {
        'evaluation_date': 'REGENERATED',
        'method': 'global_llm_judge_steered_with_stratification (regenerated)',
        'judge_model': 'gemini-2.5-flash', # Assumption
        'source_file': str(annotated_path),
        'subject': subject,
        'total_records': len(records),
        
        'dataset_info': {
            'total_configurations': len(all_configs),
            'layers': sorted(set(c['layer'] for c in all_configs)),
            'coefficient_magnitudes': sorted(set(c['coefficient_magnitude'] for c in all_configs)),
            'note': 'Regenerated from annotated records after hint mention fix'
        },

        'configurations_by_hint': {
            ht: [c for c in all_configs if c['hint_template'] == ht]
            for ht in sorted(set(c['hint_template'] for c in all_configs))
        }
    }
    
    # 5. Save summary
    # Determine summary path. Usually "summary_{stem}.json" where stem is from Input,
    # but here input is "annotated_steered_...".
    # Original summary was "summary_steered_...".
    # So we remove "annotated_" prefix.
    
    stem = annotated_path.stem
    if stem.startswith("annotated_"):
        original_stem = stem[len("annotated_"):]
    else:
        original_stem = stem
        
    summary_path = annotated_path.parent / f"summary_{original_stem}.json"
    
    print(f"Saving summary to {summary_path}...")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    # Base paths
    base_dirs = [
        Path("data/definitive_pipeline_data/Qwen3-32B"),
        Path("data/definitive_pipeline_data/DeepSeek-R1-Distill-Llama-8B"),
        Path("data/definitive_pipeline_data/Qwen3-14B")
    ]
    
    # We want to process annotated files that correspond to "steered_linear", "steered_mlp", "steered_off_policy"
    # And potentially random.
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
            
        # Find annotated files
        files = list(base_dir.glob("annotated_steered_*.jsonl"))
        
        for f in files:
            # We assume these are the ones we updated/care about
            regenerate_summary_for_file(f)

if __name__ == "__main__":
    main()
