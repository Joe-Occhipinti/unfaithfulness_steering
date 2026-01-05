"""
eval_steered_global_faithfulness.py

Steered global faithfulness evaluation script with unified async classification.

This script:
1. Auto-discovers steered evaluation dataset based on model and steering mode
2. Groups records by (hint_template, layer, coefficient)
3. For each configuration:
   a. Rule-based classification (stable, changed, incomplete)
   b. Phase A: Faithfulness judgment for stable answers
   c. Phase B: Hint mention detection for changed/incomplete
   d. Compute transition rates
4. Save annotated dataset + summary
5. Print detailed metrics

Usage:
    python eval_steered_global_faithfulness.py --model Qwen3-32B --steering-mode linear
    python eval_steered_global_faithfulness.py --model Qwen3-8B --steering-mode off_policy
"""

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from src.data import load_jsonl, save_jsonl
from src.config import TODAY, ModelConfig

# Import core classification functions
from src.steered_global_faithfulness import (
    group_records_by_config,
    compute_config_metrics
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data directory structure
DATA_DIR = Path("data/definitive_pipeline_data")

# Model configuration
JUDGE_MODEL = ModelConfig.ANNOTATION_MODELS.get("gemini-2.5-flash", "google/gemini-2.5-flash")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness and hint mentions for steered model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval_steered_global_faithfulness.py --model Qwen3-32B --steering-mode linear
    python eval_steered_global_faithfulness.py --model Qwen3-8B --steering-mode off_policy
    python eval_steered_global_faithfulness.py --input-file path/to/custom.jsonl
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-32B",
        help="Model name (e.g., Qwen3-32B, Qwen3-8B). Used for file discovery."
    )
    
    # Steering mode / dataset type
    parser.add_argument(
        "--steering-mode",
        type=str,
        default="linear",
        choices=["linear", "off_policy", "mlp"],
        help="Type of steering to evaluate: linear, off_policy, or mlp (default: linear)"
    )
    
    # Direct input file (overrides auto-discovery)
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Direct path to input JSONL file (overrides --model and --steering-mode)"
    )
    
    # Output directory (optional override)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file directory)"
    )
    
    return parser.parse_args()


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_steered_file(model_name: str, steering_mode: str) -> Path:
    """
    Find the most recent steered results file for a given model and mode.
    
    Searches in data/definitive_pipeline_data/{model_name}/ for files matching:
    steered_{mode}_{model}_{date}.jsonl
    
    Args:
        model_name: Model name (e.g., "Qwen3-32B")
        steering_mode: Steering type ("linear", "off_policy", "mlp")
    
    Returns:
        Path to the most recent matching file
    
    Raises:
        FileNotFoundError: If no matching file found
    """
    model_dir = DATA_DIR / model_name
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Available models: {[d.name for d in DATA_DIR.iterdir() if d.is_dir()]}"
        )
    
    # Pattern: steered_{mode}_{model}_{date}.jsonl
    pattern = f"steered_{steering_mode}_{model_name}_*.jsonl"
    matching_files = list(model_dir.glob(pattern))
    
    if not matching_files:
        # List available files for helpful error message
        all_steered = list(model_dir.glob("steered_*.jsonl"))
        available_modes = set()
        for f in all_steered:
            parts = f.stem.split('_')
            if len(parts) >= 2:
                available_modes.add(parts[1])
        
        raise FileNotFoundError(
            f"No steered files found for mode '{steering_mode}' in {model_dir}\n"
            f"Pattern searched: {pattern}\n"
            f"Available modes: {available_modes if available_modes else 'None'}\n"
            f"Steered files found: {[f.name for f in all_steered]}"
        )
    
    # Sort by date (YYYY-MM-DD format at end of filename) and return most recent
    def extract_date(path: Path) -> str:
        """Extract date from filename for sorting."""
        stem = path.stem  # steered_linear_Qwen3-32B_2026-01-04
        # Date is the last part after underscore
        match = re.search(r'(\d{4}-\d{2}-\d{2})$', stem)
        return match.group(1) if match else ""
    
    most_recent = sorted(matching_files, key=extract_date)[-1]
    return most_recent


def get_output_paths(input_file: Path, output_dir: Path = None) -> dict:
    """
    Generate output file paths based on input file.
    
    Args:
        input_file: Path to input file
        output_dir: Optional output directory override
    
    Returns:
        Dictionary with 'annotated' and 'summary' paths
    """
    if output_dir is None:
        output_dir = input_file.parent
    
    # Generate output names based on input
    stem = input_file.stem  # e.g., steered_linear_Qwen3-32B_2026-01-04
    
    return {
        'annotated': output_dir / f"annotated_{stem}.jsonl",
        'summary': output_dir / f"summary_{stem}.json"
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_annotated_records(all_configs, original_records):
    """
    Create annotated dataset by merging classifications back into original records.

    Args:
        all_configs: List of all configuration results with classifications
        original_records: Original steered records

    Returns:
        List of annotated records with classification fields
    """
    # Build classification lookup: (question_id, layer, coeff) -> classification dict
    classification_lookup = {}

    for config in all_configs:
        layer = config['layer']
        coeff_mag = config['coefficient_magnitude']

        for group_name, group_data in config.items():
            if group_name in ['positive_on_CF', 'positive_on_CU', 'positive_on_WF', 'positive_on_WU',
                            'negative_on_CF', 'negative_on_CU', 'negative_on_WF', 'negative_on_WU']:
                classifications = group_data.get('classifications', {})
                for qid, class_data in classifications.items():
                    # Determine coefficient sign from group name
                    if 'positive' in group_name:
                        coeff = coeff_mag
                    else:
                        coeff = -coeff_mag

                    key = (qid, layer, coeff)
                    classification_lookup[key] = class_data

    # Annotate original records
    annotated = []
    for record in original_records:
        qid = record.get('question_id', record.get('prompt_index', record.get('hinted_id')))
        layer = record['steering_layer']
        coeff = record.get('steering_coefficient', 0)

        key = (qid, layer, coeff)
        class_data = classification_lookup.get(key, {})

        # Create annotated record
        annotated_record = record.copy()
        
        # Add new classification fields
        if isinstance(class_data, dict):
            annotated_record['rule_classification'] = class_data.get('rule', 'error')
            annotated_record['faithfulness'] = class_data.get('faithfulness')
            annotated_record['hint_mentioned'] = class_data.get('hint_mentioned')
        else:
            # Legacy format fallback
            annotated_record['rule_classification'] = class_data if isinstance(class_data, str) else 'error'
            annotated_record['faithfulness'] = None
            annotated_record['hint_mentioned'] = None

        annotated.append(annotated_record)

    return annotated


def create_summary(all_configs, subject, hint_template, input_file):
    """
    Create summary JSON with all configuration metrics.

    Args:
        all_configs: List of all configuration results
        subject: Subject name
        hint_template: Hint template name
        input_file: Input file path

    Returns:
        Summary dictionary
    """
    # Count total examples (sum across all 8 groups)
    total_examples = 0
    if all_configs:
        sample_config = all_configs[0]
        for group_name in ['positive_on_CF', 'positive_on_CU', 'positive_on_WF', 'positive_on_WU',
                          'negative_on_CF', 'negative_on_CU', 'negative_on_WF', 'negative_on_WU']:
            if group_name in sample_config:
                total_examples += sample_config[group_name]['n']
        # Divide by 2 (positive + negative steering on same records)
        total_examples = total_examples // 2

    # Create summary
    summary = {
        'evaluation_date': TODAY,
        'method': 'global_llm_judge_steered_with_stratification',
        'judge_model': JUDGE_MODEL,
        'source_file': input_file,
        'subject': subject,
        'hint_template': hint_template,
        'total_examples': total_examples,

        'dataset_info': {
            'total_configurations': len(all_configs),
            'layers': sorted(set(c['layer'] for c in all_configs)),
            'coefficient_magnitudes': sorted(set(c['coefficient_magnitude'] for c in all_configs)),
            'note': 'Stratified by initial state (CF/CU/WF/WU) - 8 groups per configuration'
        },

        'all_configurations': all_configs
    }

    return summary


def print_configs_summary(all_configs, hint_template: str):
    """
    Print detailed summary of all configurations with their transition metrics.

    Args:
        all_configs: List of all configuration results
        hint_template: The hint template being summarized
    """
    print("\n" + "=" * 80)
    print(f"CONFIGURATIONS SUMMARY - {hint_template.upper()}")
    print("=" * 80)
    print(f"Total configurations processed: {len(all_configs)}")
    print(f"Initial states tracked: CF, CU, WF, WU")
    print(f"Steering directions: positive (+), negative (-)")
    print()

    # Print each configuration's metrics
    for config in all_configs:
        layer = config['layer']
        coeff_mag = config['coefficient_magnitude']
        print(f"\n  Layer {layer}, Coeff ±{coeff_mag}:")
        print(f"  {'-' * 60}")

        # Print positive steering results
        print(f"    POSITIVE STEERING (+{coeff_mag}):")
        for initial_state in ['CF', 'CU', 'WF', 'WU']:
            group_name = f'positive_on_{initial_state}'
            group_data = config.get(group_name, {})
            n = group_data.get('n', 0)
            if n > 0:
                t = group_data.get('transitions', {})
                f_rate = t.get('stable_faithful', {}).get('rate', 0)
                u_rate = t.get('stable_unfaithful', {}).get('rate', 0)
                w2c_rate = t.get('wrong_to_correct', {}).get('rate', 0)
                err_rate = t.get('hint_error', {}).get('rate', 0)
                inc_rate = t.get('incomplete', {}).get('rate', 0)
                print(f"      {initial_state}: n={n:3d} | F:{f_rate:.0%} U:{u_rate:.0%} | W→C:{w2c_rate:.0%} Err:{err_rate:.0%} Inc:{inc_rate:.0%}")

        # Print negative steering results
        print(f"    NEGATIVE STEERING (-{coeff_mag}):")
        for initial_state in ['CF', 'CU', 'WF', 'WU']:
            group_name = f'negative_on_{initial_state}'
            group_data = config.get(group_name, {})
            n = group_data.get('n', 0)
            if n > 0:
                t = group_data.get('transitions', {})
                f_rate = t.get('stable_faithful', {}).get('rate', 0)
                u_rate = t.get('stable_unfaithful', {}).get('rate', 0)
                w2c_rate = t.get('wrong_to_correct', {}).get('rate', 0)
                err_rate = t.get('hint_error', {}).get('rate', 0)
                inc_rate = t.get('incomplete', {}).get('rate', 0)
                print(f"      {initial_state}: n={n:3d} | F:{f_rate:.0%} U:{u_rate:.0%} | W→C:{w2c_rate:.0%} Err:{err_rate:.0%} Inc:{inc_rate:.0%}")

    print("\n" + "=" * 80)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for steered global faithfulness evaluation."""
    args = parse_args()
    
    print(f"\n{'=' * 80}")
    print("STEP 1: Initialization")
    print(f"{'=' * 80}")
    
    # Determine input file
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return
        print(f"✓ Using provided input file: {input_path}")
        subject = args.model  # Best guess if provided
    else:
        try:
            input_path = find_steered_file(args.model, args.steering_mode)
            print(f"✓ Auto-discovered input file: {input_path}")
            subject = args.model
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    # Determine output paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
        
    output_paths = get_output_paths(input_path, output_dir)
    print(f"  Annotated output: {output_paths['annotated']}")
    print(f"  Summary output: {output_paths['summary']}")

    print(f"=== STEERED GLOBAL FAITHFULNESS EVALUATION - {TODAY} ===")
    print(f"Subject: {subject}")

    # 1. Load data
    print(f"\n{'=' * 80}")
    print("STEP 1: Loading Data")
    print(f"{'=' * 80}")
    print(f"Loading steered dataset from: {input_path}")

    all_records = load_jsonl(input_path)
    print(f"✓ Loaded {len(all_records)} records")

    # Detect hint templates in dataset
    hint_templates = sorted(set(r.get('hint_template', 'unknown') for r in all_records))
    print(f"✓ Detected hint templates: {hint_templates}")

    # 2. Group records
    print(f"\n{'=' * 80}")
    print("STEP 2: Grouping Records by Configuration")
    print(f"{'=' * 80}")

    grouped = group_records_by_config(all_records)
    print(f"✓ Found {len(grouped)} unique (hint_template, layer, coefficient) configurations")

    # Print configuration details
    hint_templates_in_grouped = sorted(set(k[0] for k in grouped.keys()))
    layers = sorted(set(k[1] for k in grouped.keys()))
    coeffs = sorted(set(k[2] for k in grouped.keys()))
    print(f"  Hint templates: {hint_templates_in_grouped}")
    print(f"  Layers: {layers}")
    print(f"  Coefficient magnitudes: {coeffs}")

    # 3. Process each hint template separately
    print(f"\n{'=' * 80}")
    print("STEP 3: Processing Each Hint Template")
    print(f"{'=' * 80}")
    print(f"Model: {JUDGE_MODEL}")

    all_outputs = {}  # Store outputs per hint template
    all_annotated_records = []  # Collect all annotated records across hints
    all_configs_combined = []  # Collect all configs across hints

    for hint_template in hint_templates_in_grouped:
        print(f"\n{'*' * 80}")
        print(f"PROCESSING HINT TEMPLATE: {hint_template}")
        print(f"{'*' * 80}")

        # Filter configs for this hint template
        template_configs = [(k, v) for k, v in grouped.items() if k[0] == hint_template]
        print(f"  Found {len(template_configs)} configurations for '{hint_template}'")

        all_configs = []

        for (ht, layer, coeff_mag), config_groups in template_configs:
            config_result = compute_config_metrics(
                config_groups,
                hint_template=ht,
                layer=layer,
                coeff_mag=coeff_mag,
                model=JUDGE_MODEL,
                verbose=True
            )
            all_configs.append(config_result)

        print(f"\n✓ Processed all {len(all_configs)} configurations for '{hint_template}'")

        # 5. Print configuration summary
        print_configs_summary(all_configs, hint_template)

        # 6. Collect annotated records (don't save yet - will combine all hints)
        print(f"\n  Collecting annotated records for '{hint_template}'...")
        template_records = [r for r in all_records if r.get('hint_template', 'unknown') == hint_template]
        annotated_records = create_annotated_records(all_configs, template_records)
        all_annotated_records.extend(annotated_records)
        all_configs_combined.extend(all_configs)
        print(f"  ✓ Collected {len(annotated_records)} annotated records")

        # Store outputs
        all_outputs[hint_template] = {
            'configs': all_configs,
            'n_records': len(annotated_records)
        }

    # 4. Save combined outputs (all hint templates together)
    print(f"\n{'=' * 80}")
    print("STEP 4: Saving Combined Outputs")
    print(f"{'=' * 80}")

    # 4a. Save combined annotated dataset
    print(f"\nSaving combined annotated dataset...")
    os.makedirs(os.path.dirname(output_paths['annotated']), exist_ok=True)
    save_jsonl(all_annotated_records, output_paths['annotated'])
    print(f"✓ Saved {len(all_annotated_records)} annotated records: {output_paths['annotated']}")

    # 4b. Save combined summary
    print(f"\nSaving combined summary...")
    combined_summary = {
        'evaluation_date': TODAY,
        'method': 'global_llm_judge_steered_with_stratification',
        'judge_model': JUDGE_MODEL,
        'source_file': str(input_path),
        'subject': subject,
        'total_records': len(all_annotated_records),
        'hint_templates': hint_templates_in_grouped,

        'dataset_info': {
            'total_configurations': len(all_configs_combined),
            'layers': sorted(set(c['layer'] for c in all_configs_combined)),
            'coefficient_magnitudes': sorted(set(c['coefficient_magnitude'] for c in all_configs_combined)),
            'note': 'Stratified by initial state (CF/CU/WF/WU) - Two-phase classification: faithfulness for stable, hint mentions for changed'
        },

        # Store configs grouped by hint template
        'configurations_by_hint': {
            ht: [c for c in all_configs_combined if c['hint_template'] == ht]
            for ht in hint_templates_in_grouped
        }
    }

    os.makedirs(os.path.dirname(output_paths['summary']), exist_ok=True)
    with open(output_paths['summary'], 'w', encoding='utf-8') as f:
        json.dump(combined_summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved combined summary: {output_paths['summary']}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"✓ Processed {len(all_records)} steered responses")
    print(f"✓ Processed {len(hint_templates_in_grouped)} hint template(s): {hint_templates_in_grouped}")

    print(f"\n=== OUTPUT FILES ===")
    print(f"  - Annotated dataset: {output_paths['annotated']} ({len(all_annotated_records)} records)")
    print(f"  - Summary JSON: {output_paths['summary']}")

    print(f"\n=== RESULTS BY HINT TEMPLATE ===")
    for hint_template, outputs in all_outputs.items():
        print(f"\n  [{hint_template}]")
        print(f"    - Configurations analyzed: {len(outputs['configs'])}")
        print(f"    - Records processed: {outputs['n_records']}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
