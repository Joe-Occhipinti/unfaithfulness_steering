"""
eval_missing_local_faithfulness.py

Re-run local faithfulness annotation ONLY on records that don't have local_annotated_biased_prompt.

This script:
1. Loads annotated biased JSONL (already has global faithfulness classifications)
2. Filters to ONLY records missing local_annotated_biased_prompt field
3. Routes each record to appropriate local annotator based on global classification
4. Annotates with [F_body]/[U_body] markers for local activation extraction
5. Saves locally annotated results to new JSONL
6. Generates summary with annotation metrics
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
from src.config import TODAY

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Input file - annotated biased JSONL with global faithfulness classifications
INPUT_FILE = "data/sprint4_2025-10-21/annotated/annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"

# Output file - locally annotated results (will merge with existing)
OUTPUT_FILE = "data/sprint4_2025-10-21/annotated/reprocessed_local_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"

# Summary file
SUMMARY_FILE = "data/sprint4_2025-10-21/summaries/annotated/reprocessed_local_annotation_summary_2025-10-25.json"

# Model configuration
ANNOTATION_MODEL = "google/gemini-2.5-flash"  # Local annotator model
MAX_RETRIES = 3

# TEST MODE - only process first N missing prompts
TEST_MODE = False
TEST_LIMIT = 10

# =============================================================================
# END CONFIGURATION
# =============================================================================


def main():
    """Main entry point for re-running local faithfulness annotation on missing records."""
    print(f"=== RE-RUN LOCAL FAITHFULNESS ANNOTATION (MISSING ONLY) - {TODAY} ===")

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Annotated biased results file not found: {INPUT_FILE}")
        return

    print(f"Loading annotated biased results from: {INPUT_FILE}")

    # Load all data
    all_data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(all_data)} total records")

    # Filter to only records missing local annotation
    missing_local = []
    has_local = []

    for record in all_data:
        local_annotated = record.get('local_annotated_biased_prompt')
        # Check if missing or None (but not empty string, which might be valid)
        if local_annotated is None:
            missing_local.append(record)
        else:
            has_local.append(record)

    print(f"\nFiltering results:")
    print(f"  Records WITH local annotation: {len(has_local)}")
    print(f"  Records MISSING local annotation: {len(missing_local)}")

    if len(missing_local) == 0:
        print("\nNo missing local annotations found! All records are already annotated.")
        return

    # TEST MODE: limit to first N missing prompts
    if TEST_MODE:
        missing_local = missing_local[:TEST_LIMIT]
        print(f"\nTEST MODE: Processing only first {TEST_LIMIT} missing records")

    # Count global classifications for missing records
    faithful_count = sum(1 for r in missing_local if r.get('global_faithfulness_classification') == 'faithful')
    unfaithful_count = sum(1 for r in missing_local if r.get('global_faithfulness_classification') == 'unfaithful')
    error_count = sum(1 for r in missing_local if r.get('global_faithfulness_classification') == 'error')
    other_count = len(missing_local) - faithful_count - unfaithful_count - error_count

    print(f"\nGlobal classifications for missing records:")
    print(f"  Faithful: {faithful_count}")
    print(f"  Unfaithful: {unfaithful_count}")
    print(f"  Error: {error_count} (will be skipped)")
    print(f"  Other: {other_count}")

    if faithful_count + unfaithful_count == 0:
        print("\nNo faithful or unfaithful records to annotate!")
        return

    # Setup OpenRouter client for local annotation
    try:
        openrouter_client = setup_openrouter_client()
    except ValueError as e:
        print(f"\nError setting up OpenRouter client: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return

    # Annotate locally (function will route to appropriate annotator based on global classification)
    print(f"\nAnnotating {len(missing_local)} missing records locally with {ANNOTATION_MODEL}...")
    print(f"This will mark [F_body] spans for faithful responses and [U_body] spans for unfaithful responses.")

    local_annotations = annotate_local_batch(
        results=missing_local,
        client=openrouter_client,
        model=ANNOTATION_MODEL,
        max_retries=MAX_RETRIES,
        verbose=True
    )

    # Create locally annotated results for missing records
    newly_annotated_results = []
    for result, local_annotation in zip(missing_local, local_annotations):
        # Create a copy of the result with local annotation fields
        locally_annotated_result = result.copy()

        # Add local annotation fields
        locally_annotated_result['local_annotated_biased_prompt'] = local_annotation.get('annotated_text')
        locally_annotated_result['local_annotation_success'] = local_annotation.get('success')
        locally_annotated_result['local_annotation_raw_response'] = local_annotation.get('raw_response')

        if not local_annotation.get('success'):
            locally_annotated_result['local_annotation_error'] = local_annotation.get('error')

        newly_annotated_results.append(locally_annotated_result)

    # Merge with existing records that already had local annotations
    all_results = has_local + newly_annotated_results

    # Compute local annotation metrics
    local_metrics = compute_local_annotation_metrics(local_annotations)

    # Print metrics report
    print(f"\n=== LOCAL ANNOTATION METRICS (NEWLY ANNOTATED) ===")
    print(f"Total annotations: {local_metrics['total_annotations']}")
    print(f"Successful: {local_metrics['successful']} ({local_metrics['success_rate']:.1%})")
    print(f"Errors: {local_metrics['errors']}")
    print(f"\nMarker counts:")
    print(f"  [F_body] markers: {local_metrics['f_body_markers']} (avg {local_metrics['avg_f_markers_per_annotation']:.2f} per annotation)")
    print(f"  [U_body] markers: {local_metrics['u_body_markers']} (avg {local_metrics['avg_u_markers_per_annotation']:.2f} per annotation)")

    # Save all results (existing + newly annotated)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    save_jsonl(all_results, OUTPUT_FILE)
    print(f"\nSaved {len(all_results)} total results to {OUTPUT_FILE}")
    print(f"  ({len(has_local)} already had local annotations + {len(newly_annotated_results)} newly annotated)")

    # Create and save summary
    summary = {
        'evaluation_date': TODAY,
        'method': 'local_annotation_reprocessing',
        'annotation_model': ANNOTATION_MODEL,
        'source_file': INPUT_FILE,
        'total_input_records': len(all_data),
        'records_already_annotated': len(has_local),
        'records_missing_annotation': len(missing_local),
        'records_newly_annotated': len(newly_annotated_results),
        'global_classifications_of_missing': {
            'faithful': faithful_count,
            'unfaithful': unfaithful_count,
            'error': error_count,
            'other': other_count
        },
        'local_annotation_metrics': local_metrics,
        'output_file': OUTPUT_FILE
    }

    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved local annotation summary to {SUMMARY_FILE}")

    print(f"\n=== LOCAL FAITHFULNESS RE-ANNOTATION COMPLETE ===")
    print(f"✅ Re-processed {len(newly_annotated_results)} missing local annotations")
    print(f"✅ Saved complete dataset ({len(all_results)} records) to: {OUTPUT_FILE}")
    print(f"✅ Saved summary to: {SUMMARY_FILE}")
    print(f"\nNext steps:")
    print(f"   1. Review the reprocessed dataset")
    print(f"   2. Filter out incomplete annotations (no period before closing tag)")
    print(f"   3. Use for activation extraction")


if __name__ == "__main__":
    main()
