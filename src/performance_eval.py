"""
performance_eval.py

Performance evaluation utilities for faithfulness steering workflow.
Includes format validation, answer extraction, accuracy computation, and visualization.
Reusable across baseline, hinted, and steering evaluation scripts.
"""

import json
import os
from typing import Dict, Any, List
from google import genai
from google.genai import types
from tqdm import tqdm

def load_validation_prompt() -> str:
    """
    Load Gemini validation prompt from prompts folder.
    Reusable across all evaluation scripts.

    Returns:
        Validation prompt template string
    """
    prompt_path = os.path.join("prompts", "gemini_validation_prompt.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def setup_gemini_client() -> genai.Client:
    """
    Setup Gemini client for format validation.
    Reusable across all evaluation scripts.

    Returns:
        Configured Gemini client
    """
    print(f"\n--- Setting up Gemini validation ---")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    print("Gemini 2.5 Flash-Lite client ready")
    return client

def validate_with_gemini(response: str, client: genai.Client) -> Dict[str, Any]:
    """
    Use Gemini 2.5 Flash-Lite to validate format and extract final answer.
    Reusable across all evaluation scripts.

    Args:
        response: Model response to validate
        client: Gemini client

    Returns:
        Dictionary with format_followed, response_complete, final_answer
    """

    # Load validation prompt from prompts folder
    validation_prompt_template = load_validation_prompt()
    validation_prompt = validation_prompt_template.format(response=response)

    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=validation_prompt)],
            ),
        ]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0  # Deterministic for validation
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=config,
        )

        # Parse JSON response
        result = json.loads(response.text.strip())
        return result

    except Exception as e:
        print(f"Gemini validation error: {e}")
        # Fallback - assume format not followed if validation fails
        return {
            "format_followed": False,
            "response_complete": True,
            "final_answer": None
        }

def compute_accuracy_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute detailed accuracy metrics from evaluation results.
    Reusable across all evaluation scripts.

    Args:
        results: List of evaluation result dictionaries

    Returns:
        Dictionary with comprehensive accuracy metrics
    """
    total = len(results)
    correct = sum(1 for r in results if r['accuracy_label'] == 'correct')

    # Subject breakdown
    subject_stats = {}
    for result in results:
        subject = result['subject']
        if subject not in subject_stats:
            subject_stats[subject] = {
                'correct': 0, 'total': 0, 'extraction_failed': 0,
                'format_violations': 0, 'incomplete_responses': 0
            }

        subject_stats[subject]['total'] += 1
        if result['accuracy_label'] == 'correct':
            subject_stats[subject]['correct'] += 1
        if result.get('answer_letter') is None:  # Changed from 'final_answer'
            subject_stats[subject]['extraction_failed'] += 1
        if not result.get('format_followed', True):
            subject_stats[subject]['format_violations'] += 1
        if not result.get('response_complete', True):
            subject_stats[subject]['incomplete_responses'] += 1

    # Calculate subject accuracies
    for subject in subject_stats:
        stats = subject_stats[subject]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    return {
        'overall_accuracy': correct / total if total > 0 else 0,
        'total_questions': total,
        'correct_answers': correct,
        'wrong_answers': total - correct,
        'extraction_failures': sum(1 for r in results if r.get('answer_letter') is None),
        'format_violations': sum(1 for r in results if not r.get('format_followed', True)),
        'incomplete_responses': sum(1 for r in results if not r.get('response_complete', True)),
        'subject_breakdown': subject_stats
    }

def print_accuracy_report(metrics: Dict[str, Any]) -> None:
    """
    Print formatted accuracy report.
    Reusable across all evaluation scripts.

    Args:
        metrics: Accuracy metrics dictionary
    """
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Correct: {metrics['correct_answers']}, Wrong: {metrics['wrong_answers']}")
    print(f"Extraction Failures: {metrics['extraction_failures']}")
    print(f"Format Violations: {metrics['format_violations']}")
    print(f"Incomplete Responses: {metrics['incomplete_responses']}")

    print(f"\nSubject Breakdown:")
    for subject, stats in metrics['subject_breakdown'].items():
        print(f"  {subject}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")

def validate_responses_batch(responses: List[str], client: genai.Client) -> List[Dict[str, Any]]:
    """
    Validate multiple responses with progress tracking.
    Reusable across all evaluation scripts.

    Args:
        responses: List of responses to validate
        client: Gemini client

    Returns:
        List of validation results
    """
    print(f"\n--- Validating {len(responses)} responses with Gemini ---")

    validations = []
    for response in tqdm(responses, desc="Validating"):
        validation = validate_with_gemini(response, client)
        validations.append(validation)

    return validations