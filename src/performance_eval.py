"""
performance_eval.py

Performance evaluation utilities for faithfulness steering workflow.
Includes format validation, answer extraction, accuracy computation, and visualization.
Reusable across baseline, hinted, and steering evaluation scripts.
"""

import json
import os
import time
from typing import Dict, Any, List
from google import genai
from google.genai import types
from tqdm import tqdm
import requests

from .config import GEMINI_FLASH_LITE_MIN_DELAY

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

def setup_gemini_client() -> genai.Client:
    """
    Setup Gemini client for format validation.
    Reusable across all evaluation scripts.

    Returns:
        Configured Gemini client
    """
    print(f"\n--- Setting up Gemini validation ---")
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    print("Gemini 2.5 Flash-Lite client ready")
    return client


def setup_deepseek_client() -> Dict[str, str]:
    """
    Setup DeepSeek client configuration for validation.
    Reusable across all evaluation scripts.

    Returns:
        Dictionary with DeepSeek API configuration
    """
    print(f"\n--- Setting up DeepSeek validation ---")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable must be set")

    config = {
        "api_key": api_key,
        "base_url": base_url,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    }

    print("DeepSeek-Reasoner client ready")
    return config

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


def validate_with_deepseek(response: str, client_config: Dict[str, str], max_retries: int = 3) -> Dict[str, Any]:
    """
    Use DeepSeek-Reasoner to validate format and extract final answer.
    Reusable across all evaluation scripts.

    Args:
        response: Model response to validate
        client_config: DeepSeek client configuration
        max_retries: Number of retry attempts

    Returns:
        Dictionary with format_followed, response_complete, final_answer
    """

    # Load validation prompt from prompts folder
    validation_prompt_template = load_validation_prompt()
    validation_prompt = validation_prompt_template.format(response=response)

    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {
                "role": "user",
                "content": validation_prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": False
    }

    # Retry logic for API calls
    for attempt in range(max_retries):
        try:
            response_obj = requests.post(
                f"{client_config['base_url']}/v1/chat/completions",
                headers=client_config['headers'],
                json=payload,
                timeout=(30, 300)  # (connection, read) - 5 minutes read timeout
            )

            response_obj.raise_for_status()
            result = response_obj.json()

            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content'].strip()

                # Parse JSON response
                try:
                    parsed_result = json.loads(content)
                    return parsed_result
                except json.JSONDecodeError:
                    # Fallback - try to extract from text if JSON parsing fails
                    print(f"JSON parsing failed, content: {content}")
                    return {
                        "format_followed": False,
                        "response_complete": True,
                        "final_answer": None
                    }
            else:
                raise ValueError("No valid response from DeepSeek API")

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                print(f"DeepSeek rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting generously...")
                time.sleep(120)  # Very generous wait for rate limits - 2 minutes
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print(f"DeepSeek timeout after 5 minutes (attempt {attempt + 1}/{max_retries}). This request is taking too long...")
                if attempt < max_retries - 1:
                    time.sleep(30)  # Wait before retry for timeouts
            else:
                print(f"DeepSeek API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(30 + (10 * attempt))  # Conservative backoff: 30s, 40s, 50s

        except Exception as e:
            print(f"DeepSeek validation error: {e}")
            if attempt < max_retries - 1:
                time.sleep(30 + (10 * attempt))  # Conservative backoff: 30s, 40s, 50s

    # Fallback - assume validation failed
    return {
        "format_followed": False,
        "response_complete": False,
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

def validate_responses(responses: List[str], client: genai.Client) -> List[Dict[str, Any]]:
    """
    Validate multiple responses with rate limiting and progress tracking.
    Reusable across all evaluation scripts.

    Args:
        responses: List of responses to validate
        client: Gemini client

    Returns:
        List of validation results
    """
    print(f"\n--- Validating {len(responses)} responses with Gemini Flash-Lite ---")
    print(f"Rate limit: 15 requests per minute ({GEMINI_FLASH_LITE_MIN_DELAY}s delays)")
    print(f"Estimated time: {len(responses) * GEMINI_FLASH_LITE_MIN_DELAY / 60:.1f} minutes")

    validations = []
    start_time = time.time()

    for i, response in enumerate(tqdm(responses, desc="Validating")):
        # Rate limiting: ensure minimum delay between requests
        if i > 0:
            elapsed = time.time() - request_start_time
            if elapsed < GEMINI_FLASH_LITE_MIN_DELAY:
                sleep_time = GEMINI_FLASH_LITE_MIN_DELAY - elapsed
                time.sleep(sleep_time)

        # Validate single response
        request_start_time = time.time()
        validation = validate_with_gemini(response, client)
        validations.append(validation)

    return validations


def validate_responses_deepseek(responses: List[str], client_config: Dict[str, str],
                               min_delay: float = 5.0) -> List[Dict[str, Any]]:
    """
    Validate multiple responses with DeepSeek-Reasoner with rate limiting and progress tracking.
    Reusable across all evaluation scripts.

    Args:
        responses: List of responses to validate
        client_config: DeepSeek client configuration
        min_delay: Minimum delay between requests in seconds

    Returns:
        List of validation results
    """
    print(f"\n--- Validating {len(responses)} responses with DeepSeek-Reasoner ---")
    print(f"Rate limit: {min_delay}s delays between requests")
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
        validation = validate_with_deepseek(response, client_config)
        validations.append(validation)

    return validations