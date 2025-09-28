"""
performance_eval.py

Performance evaluation utilities for faithfulness steering workflow.
Includes format validation, answer extraction, accuracy computation, and visualization.
Reusable across baseline, hinted, and steering evaluation scripts.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
from tqdm import tqdm

from .config import ModelConfig

def load_validation_prompt() -> str:
    """
    Load validation prompt from prompts folder.
    Reusable across all evaluation scripts.

    Returns:
        Validation prompt template string
    """
    prompt_path = os.path.join("prompts", "validation_prompt.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def setup_openrouter_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Setup OpenRouter client for all model interactions.
    Reusable across all evaluation scripts.

    Args:
        api_key: Optional API key, otherwise uses environment variable

    Returns:
        Configured OpenAI client for OpenRouter
    """
    print(f"\n--- Setting up OpenRouter client ---")
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    print("OpenRouter client ready")
    return client

# Backward compatibility aliases (can be removed later)
setup_gemini_client = setup_openrouter_client
setup_deepseek_client = setup_openrouter_client

def validate_response(response: str, client: OpenAI) -> Dict[str, Any]:
    """
    Validate format and extract final answer using the default validation model.
    Only passes the last sentence for efficiency.
    Reusable across all evaluation scripts.

    Args:
        response: Model response to validate
        client: OpenAI client configured for OpenRouter

    Returns:
        Dictionary with format_followed, response_complete, final_answer
    """

    # Extract only the last 200 characters for validation
    last_segment = response.strip()[-200:] if len(response.strip()) > 200 else response.strip()

    # Load validation prompt from prompts folder
    validation_prompt_template = load_validation_prompt()
    validation_prompt = validation_prompt_template.format(response=last_segment)

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": os.environ.get("SITE_URL", "https://github.com"),
                "X-Title": os.environ.get("SITE_NAME", "Faithfulness Steering")
            },
            model=ModelConfig.DEFAULT_VALIDATION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            temperature=0  # Deterministic for validation
        )

        # Parse JSON response
        result = json.loads(completion.choices[0].message.content.strip())
        return result

    except Exception as e:
        print(f"Validation error: {e}")
        # Fallback - assume format not followed if validation fails
        return {
            "format_followed": False,
            "response_complete": True,
            "final_answer": None
        }



def extract_validation_data(validation: Dict[str, Any]) -> tuple:
    """
    Extract validation data with safe defaults.
    Reusable across all evaluation scripts.

    Args:
        validation: Validation dictionary from DeepSeek/Gemini

    Returns:
        Tuple of (format_followed, response_complete, answer_letter)
    """
    format_followed = validation.get('format_followed', False)
    response_complete = validation.get('response_complete', False)
    answer_letter = validation.get('final_answer', None)
    return format_followed, response_complete, answer_letter


def label_accuracy(answer_letter: str, ground_truth_letter: str) -> tuple:
    """
    Determine correctness and accuracy label.
    Reusable across all evaluation scripts.

    Args:
        answer_letter: Model's answer letter (A/B/C/D or None)
        ground_truth_letter: Correct answer letter

    Returns:
        Tuple of (is_correct, accuracy_label)
    """
    is_correct = (answer_letter == ground_truth_letter) if answer_letter is not None else False
    accuracy_label = 'correct' if is_correct else 'wrong'
    return is_correct, accuracy_label

def compute_bias_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute bias-specific metrics for hinted evaluation.
    Used in hinted_eval.py for bias analysis.

    Args:
        results: List of hinted evaluation results with bias_label field

    Returns:
        Dictionary with bias metrics
    """
    total = len(results)
    biased_count = sum(1 for r in results if r.get('bias_label') == 'biased')
    not_biased_count = sum(1 for r in results if r.get('bias_label') == 'not-biased')

    return {
        'total_hinted_questions': total,
        'biased_answers': biased_count,
        'not_biased_answers': not_biased_count,
        'bias_rate': biased_count / total if total > 0 else 0,
        'hint_resistance_rate': not_biased_count / total if total > 0 else 0
    }

def compute_completeness_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute response completeness metrics from evaluation results.
    Reusable across all evaluation scripts.

    Args:
        results: List of evaluation result dictionaries with 'response_complete' field

    Returns:
        Dictionary with completeness metrics
    """
    total = len(results)
    complete = sum(1 for r in results if r.get('response_complete', False))
    incomplete = total - complete

    return {
        'completeness_rate': complete / total if total > 0 else 0.0,
        'complete_responses': complete,
        'incomplete_responses': incomplete,
        'total_responses': total
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
        # Check for answer_letter (baseline), hinted_answer_letter (hinted eval), or steered_answer_letter (steered eval)
        extracted_answer = (result.get('answer_letter') or
                          result.get('hinted_answer_letter') or
                          result.get('steered_answer_letter'))
        if extracted_answer is None:
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
        'extraction_failures': sum(1 for r in results
                                  if not any([r.get('answer_letter'),
                                             r.get('hinted_answer_letter'),
                                             r.get('steered_answer_letter')])),
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

def validate_responses(responses: List[str], client: OpenAI) -> List[Dict[str, Any]]:
    """
    Validate multiple responses with rate limiting and progress tracking.
    Uses the default validation model from config.
    Reusable across all evaluation scripts.

    Args:
        responses: List of responses to validate
        client: OpenAI client configured for OpenRouter

    Returns:
        List of validation results
    """
    model = ModelConfig.DEFAULT_VALIDATION_MODEL
    min_delay = ModelConfig.get_min_delay(model)

    print(f"\n--- Validating {len(responses)} responses with {model} ---")
    print(f"Rate limit: {min_delay:.1f}s delays between requests")
    print(f"Estimated time: {len(responses) * min_delay / 60:.1f} minutes")

    validations = []
    start_time = time.time()

    for i, response in enumerate(tqdm(responses, desc="Validating")):
        # Rate limiting: ensure minimum delay between requests
        if i > 0:
            elapsed = time.time() - request_start_time
            if elapsed < min_delay:
                sleep_time = min_delay - elapsed
                time.sleep(sleep_time)

        # Validate single response
        request_start_time = time.time()
        validation = validate_response(response, client)
        validations.append(validation)

    return validations