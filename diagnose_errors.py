"""
diagnose_errors.py

Analyze error classifications to understand their nature and identify prevention strategies.

Errors can occur at two stages:
1. Rule-based classification (in steered_global_faithfulness.py)
2. LLM judge (in global_faithfulness.py)

This script examines actual data to categorize error sources.
"""

import json
from collections import Counter
from src.data import load_jsonl


def analyze_errors(input_file: str):
    """
    Analyze error classifications in annotated data.

    Args:
        input_file: Path to annotated JSONL file
    """
    print(f"=== ERROR ANALYSIS ===")
    print(f"Input: {input_file}\n")

    # Load data
    records = load_jsonl(input_file)
    print(f"Total records: {len(records)}")

    # Find records with error classification
    errors = [r for r in records if r.get('steered_global_faithfulness_classification') == 'error']
    print(f"Records with 'error' classification: {len(errors)}")

    if len(errors) == 0:
        print("\nNo errors found!")
        return

    print(f"Error rate: {len(errors) / len(records) * 100:.1f}%\n")

    # Categorize error sources
    print("=" * 80)
    print("ERROR CATEGORIZATION")
    print("=" * 80)

    error_categories = {
        'llm_judge_json_parse_error': 0,
        'llm_judge_api_error': 0,
        'llm_judge_invalid_classification': 0,
        'llm_judge_timeout': 0,
        'missing_data': 0,
        'unknown': 0
    }

    error_details = []

    for record in errors:
        qid = record.get('question_id', record.get('prompt_index'))

        # Check if we have raw response (indicates LLM was called)
        # Note: This requires checking original judgment data if available

        # Check for missing required fields
        if not record.get('steered_answer_letter'):
            error_categories['missing_data'] += 1
            error_details.append({
                'qid': qid,
                'category': 'missing_data',
                'detail': 'Missing steered_answer_letter',
                'question': record.get('question', '')[:100]
            })
        elif not record.get('ground_truth_letter'):
            error_categories['missing_data'] += 1
            error_details.append({
                'qid': qid,
                'category': 'missing_data',
                'detail': 'Missing ground_truth_letter',
                'question': record.get('question', '')[:100]
            })
        elif not record.get('hint_letter'):
            error_categories['missing_data'] += 1
            error_details.append({
                'qid': qid,
                'category': 'missing_data',
                'detail': 'Missing hint_letter',
                'question': record.get('question', '')[:100]
            })
        else:
            # Likely LLM judge error
            error_categories['unknown'] += 1
            error_details.append({
                'qid': qid,
                'category': 'unknown',
                'detail': 'Possible LLM judge error (needs raw response to confirm)',
                'completeness': record.get('completeness'),
                'steered_answer': record.get('steered_answer_letter'),
                'ground_truth': record.get('ground_truth_letter'),
                'hint': record.get('hint_letter')
            })

    # Print summary
    print("\nError Categories:")
    for category, count in error_categories.items():
        if count > 0:
            print(f"  {category}: {count} ({count/len(errors)*100:.1f}%)")

    # Print detailed examples
    print("\n" + "=" * 80)
    print("ERROR EXAMPLES (first 5 of each type)")
    print("=" * 80)

    categories_shown = set()
    for detail in error_details:
        category = detail['category']
        if category not in categories_shown:
            print(f"\n{category.upper()}:")
            categories_shown.add(category)

        # Show first 5 examples of each category
        examples_of_category = [d for d in error_details if d['category'] == category]
        if examples_of_category.index(detail) < 5:
            print(f"\n  Question ID: {detail['qid']}")
            for key, value in detail.items():
                if key != 'qid' and key != 'category':
                    print(f"    {key}: {value}")

    # Recommendations
    print("\n" + "=" * 80)
    print("PREVENTION RECOMMENDATIONS")
    print("=" * 80)

    if error_categories['missing_data'] > 0:
        print("\n1. MISSING DATA:")
        print("   - Add validation checks before classification")
        print("   - Ensure all required fields are present in input data")
        print("   - Consider adding a separate 'invalid_data' category")

    if error_categories['unknown'] > 0:
        print("\n2. LLM JUDGE ERRORS (suspected):")
        print("   - Store raw LLM responses in annotated data for debugging")
        print("   - Implement more robust JSON parsing (try multiple patterns)")
        print("   - Add response format validation to the judge prompt")
        print("   - Consider increasing max_retries for API calls")
        print("   - Monitor specific error messages from API")

    print("\n3. GENERAL RECOMMENDATIONS:")
    print("   - Log all errors to a separate file with full context")
    print("   - Add error classification subtypes (json_error, api_error, etc.)")
    print("   - Create a manual review queue for high-value errors")
    print("   - Track error rates over time to identify patterns")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diagnose_errors.py <path_to_annotated_file.jsonl>")
        print("\nExample:")
        print("  python diagnose_errors.py data/annotated/annotated_steered_local_high_school_psychology_professor_2025-10-06.jsonl")
        return

    input_file = sys.argv[1]
    analyze_errors(input_file)


if __name__ == "__main__":
    main()
