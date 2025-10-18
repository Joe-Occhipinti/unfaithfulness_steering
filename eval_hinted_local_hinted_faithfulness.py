"""
eval_local_faithfulness.py --> new pipeline: annotate locally after classifying globbally

Local faithfulness annotation script for marking faithful and unfaithful reasoning spans.

This script:
1. Loads annotated biased JSONL (already has global faithfulness classifications)
2. Routes each record to appropriate local annotator (faithful vs unfaithful) based on global classification
3. Annotates with [F_body]/[U_body] markers for local activation extraction
4. Saves locally annotated results to new JSONL
5. Generates summary with annotation metrics

Unlike global faithfulness evaluation, this produces fine-grained span-level annotations
rather than whole-response classifications.
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.local_faithfulness import (
    setup_openrouter_client,
    annotate_local_batch,
    compute_local_annotation_metrics
)
from src.config import TODAY, ANNOTATED_DIR

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Input file - annotated biased JSONL with global faithfulness classifications
INPUT_FILE = "data/annotated/hinted/" \
"annotated_global_biased_stem_2025-10-15.jsonl"

# Output files - local faithfulness annotations
OUTPUT_FILE = "data/annotated/hinted/annotated_local_biased_stem_2025-10-15.jsonl"
SUMMARY_FILE = "data/summaries/faithfulness_local_biased_stem_2025-10-15.json"

# Model configuration
ANNOTATION_MODEL = "google/gemini-2.5-flash"  # Local annotator model
MAX_RETRIES = 3

# TEST MODE - only process first N prompts
TEST_MODE = False
TEST_LIMIT = 10

# =============================================================================
# END CONFIGURATION
# =============================================================================


def main():
    """Main entry point for local faithfulness annotation."""
    print(f"=== LOCAL FAITHFULNESS ANNOTATION - {TODAY} ===")

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Annotated biased results file not found: {INPUT_FILE}")
        print("Please run eval_global_faithfulness.py first to generate global annotations.")
        return

    print(f"Loading annotated biased results from: {INPUT_FILE}")

    # Load the annotated biased results (already has global classifications)
    annotated_data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(annotated_data)} annotated biased results")

    # TEST MODE: limit to first N prompts
    if TEST_MODE:
        annotated_data = annotated_data[:TEST_LIMIT]
        print(f"TEST MODE: Processing only first {TEST_LIMIT} prompts")

    # Count global classifications
    faithful_count = sum(1 for r in annotated_data if r.get('faithfulness_classification') == 'faithful')
    unfaithful_count = sum(1 for r in annotated_data if r.get('faithfulness_classification') == 'unfaithful')
    error_count = sum(1 for r in annotated_data if r.get('faithfulness_classification') == 'error')

    print(f"Global classifications:")
    print(f"  Faithful: {faithful_count}")
    print(f"  Unfaithful: {unfaithful_count}")
    print(f"  Error: {error_count} (will be skipped)")

    if faithful_count + unfaithful_count == 0:
        print("No faithful or unfaithful records to annotate!")
        return

    # Setup OpenRouter client for local annotation
    try:
        openrouter_client = setup_openrouter_client()
    except ValueError as e:
        print(f"Error setting up OpenRouter client: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return

    # Annotate locally (function will route to appropriate annotator based on global classification)
    print(f"\nAnnotating locally with {ANNOTATION_MODEL}...")
    print(f"This will mark [F_body] spans for faithful responses and [U_body] spans for unfaithful responses.")

    local_annotations = annotate_local_batch(
        results=annotated_data,
        client=openrouter_client,
        model=ANNOTATION_MODEL,
        max_retries=MAX_RETRIES,
        verbose=True
    )

    # Create locally annotated results
    locally_annotated_results = []
    for result, local_annotation in zip(annotated_data, local_annotations):
        # Create a copy of the result with local annotation fields
        locally_annotated_result = result.copy()

        # Add local annotation fields
        locally_annotated_result['local_annotated_biased_prompt'] = local_annotation.get('annotated_text')
        locally_annotated_result['local_annotation_success'] = local_annotation.get('success')
        locally_annotated_result['local_annotation_raw_response'] = local_annotation.get('raw_response')

        if not local_annotation.get('success'):
            locally_annotated_result['local_annotation_error'] = local_annotation.get('error')

        locally_annotated_results.append(locally_annotated_result)

    # Compute local annotation metrics
    local_metrics = compute_local_annotation_metrics(local_annotations)

    # Print metrics report
    print(f"\n=== LOCAL ANNOTATION METRICS ===")
    print(f"Total annotations: {local_metrics['total_annotations']}")
    print(f"Successful: {local_metrics['successful']} ({local_metrics['success_rate']:.1%})")
    print(f"Errors: {local_metrics['errors']}")
    print(f"\nMarker counts:")
    print(f"  [F_body] markers: {local_metrics['f_body_markers']} (avg {local_metrics['avg_f_markers_per_annotation']:.2f} per annotation)")
    print(f"  [U_body] markers: {local_metrics['u_body_markers']} (avg {local_metrics['avg_u_markers_per_annotation']:.2f} per annotation)")

    # Save locally annotated results
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    save_jsonl(locally_annotated_results, OUTPUT_FILE)
    print(f"\nSaved {len(locally_annotated_results)} locally annotated results to {OUTPUT_FILE}")

    # Create and save summary
    summary = {
        'evaluation_date': TODAY,
        'method': 'local_annotation',
        'annotation_model': ANNOTATION_MODEL,
        'source_file': INPUT_FILE,
        'total_records': len(annotated_data),
        'global_classifications': {
            'faithful': faithful_count,
            'unfaithful': unfaithful_count,
            'error': error_count
        },
        'local_annotation_metrics': local_metrics,
        'output_file': OUTPUT_FILE
    }

    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved local annotation summary to {SUMMARY_FILE}")

    print(f"\n=== LOCAL FAITHFULNESS ANNOTATION COMPLETE ===")
    print(f"+ Local faithfulness annotation completed:")
    print(f"   + Loaded {len(annotated_data)} globally annotated results from: {INPUT_FILE}")
    print(f"   + Annotated {faithful_count} faithful and {unfaithful_count} unfaithful responses")
    print(f"   + Marked [F_body] and [U_body] spans with {ANNOTATION_MODEL}")
    print(f"   + Saved locally annotated results to: {OUTPUT_FILE}")
    print(f"   + Saved summary to: {SUMMARY_FILE}")
    print(f"\nNext steps:")
    print(f"   + Use locally annotated data for activation extraction at [F_body]/[U_body] spans")
    print(f"   + Extract steering vectors from local activations")


if __name__ == "__main__":
    main()
