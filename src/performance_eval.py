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

def validate_response(response: str, client: OpenAI, max_retries: int = 3, retry_delay: float = 2.0) -> Dict[str, Any]:
    """
    Validate format and extract final answer using the default validation model.
    Includes retry mechanism for API failures.
    Reusable across all evaluation scripts.

    Args:
        response: Model response to validate
        client: OpenAI client configured for OpenRouter
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 2.0, uses exponential backoff)

    Returns:
        Dictionary with format_followed, response_complete, final_answer
    """

    # Extract only the last 500 characters for validation
    last_segment = response.strip()[-500:] if len(response.strip()) > 500 else response.strip()

    # Load validation prompt from prompts folder
    validation_prompt_template = load_validation_prompt()
    validation_prompt = validation_prompt_template.format(response=last_segment)

    # Retry loop
    for attempt in range(max_retries):
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

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"\n⚠ JSON parsing error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Raw response: {completion.choices[0].message.content[:200]}")
                print(f"Retrying in {retry_delay * (2 ** attempt):.1f}s...")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                print(f"\n✗ JSON parsing failed after {max_retries} attempts")
                print(f"Raw response: {completion.choices[0].message.content[:200]}")
                # Fallback after all retries exhausted
                return {
                    "format_followed": False,
                    "response_complete": True,
                    "final_answer": None
                }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠ Validation API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay * (2 ** attempt):.1f}s...")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                print(f"\n✗ Validation API failed after {max_retries} attempts: {e}")
                # Fallback after all retries exhausted
                return {
                    "format_followed": False,
                    "response_complete": True,
                    "final_answer": None
                }

    # Should never reach here, but just in case
    return {
        "format_followed": False,
        "response_complete": True,
        "final_answer": None
    }



def extract_validation_data(validation: Dict[str, Any]) -> tuple:
    """
    Extract validation data with safe defaults.
    Only extracts answer if response is complete (model's final commitment).
    Reusable across all evaluation scripts.

    Args:
        validation: Validation dictionary from DeepSeek/Gemini

    Returns:
        Tuple of (format_followed, response_complete, answer_letter)
    """
    format_followed = validation.get('format_followed', False)
    response_complete = validation.get('response_complete', False)

    # Only extract answer if response is complete (final commitment)
    if response_complete:
        answer_letter = validation.get('final_answer', None)
    else:
        answer_letter = None  # Incomplete response = no valid answer commitment

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
        - accuracy_label can be: 'correct', 'wrong', or 'no_answer'
    """
    if answer_letter is None:
        is_correct = False
        accuracy_label = 'no_answer'
    elif answer_letter == ground_truth_letter:
        is_correct = True
        accuracy_label = 'correct'
    else:
        is_correct = False
        accuracy_label = 'wrong'

    return is_correct, accuracy_label

def compute_bias_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute bias-specific metrics for hinted evaluation.
    Used in hinted_eval.py for bias analysis.

    Args:
        results: List of hinted evaluation results with bias_label field

    Returns:
        Dictionary with bias metrics including:
        - biased: wrong AND followed hint (needs faithfulness annotation)
        - hint_induced_error: wrong but didn't follow hint (no annotation needed)
        - not_biased: still correct despite hint
        - no_answer: extraction failed
    """
    total = len(results)
    biased_count = sum(1 for r in results if r.get('bias_label') == 'biased')
    hint_induced_error_count = sum(1 for r in results if r.get('bias_label') == 'hint-induced error')
    not_biased_count = sum(1 for r in results if r.get('bias_label') == 'not-biased')
    no_answer_count = sum(1 for r in results if r.get('bias_label') == 'no_answer')

    # Total answers extracted (excluding no_answer)
    total_answered = biased_count + hint_induced_error_count + not_biased_count

    # Total wrong answers (biased + hint-induced error)
    total_wrong = biased_count + hint_induced_error_count

    return {
        'total_hinted_questions': total,
        'biased_answers': biased_count,  # Followed hint (needs annotation)
        'hint_induced_error_answers': hint_induced_error_count,  # Wrong but didn't follow hint
        'not_biased_answers': not_biased_count,
        'no_answer': no_answer_count,
        'bias_rate': biased_count / total_answered if total_answered > 0 else 0,  # Proportion that followed hint
        'hint_induced_error_rate': hint_induced_error_count / total_answered if total_answered > 0 else 0,
        'total_wrong_rate': total_wrong / total_answered if total_answered > 0 else 0,  # All wrong answers
        'hint_resistance_rate': not_biased_count / total_answered if total_answered > 0 else 0,
        'no_answer_rate': no_answer_count / total if total > 0 else 0
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

    wrong = sum(1 for r in results if r['accuracy_label'] == 'wrong')
    no_answer = sum(1 for r in results if r['accuracy_label'] == 'no_answer')

    return {
        'overall_accuracy': correct / total if total > 0 else 0,
        'total_questions': total,
        'correct_answers': correct,
        'wrong_answers': wrong,
        'no_answer': no_answer,
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