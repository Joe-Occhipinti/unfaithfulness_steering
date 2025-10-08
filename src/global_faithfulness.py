"""
global_faithfulness_judge.py

LLM-based faithfulness judge using a global annotation approach.
Uses OpenRouter API with gemini-2.5-flash to directly classify faithfulness
based on the faithfulness_global_annotation_professor.txt system prompt.

This is an alternative to the tag-based annotation + rule-based classification
approach in faithfulness_eval.py.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from openai import OpenAI
from tqdm import tqdm

# Import model configuration
try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig


def load_global_annotation_prompt(hint_template: str = "professor") -> str:
    """
    Load the global faithfulness annotation prompt based on hint template.

    Args:
        hint_template: Type of hint used (e.g., "metadata", "professor", "black_square")

    Returns:
        Global annotation prompt string
    """
    # Map hint templates to global annotation prompt files
    prompt_files = {
        "professor": "faithfulness_global_annotation_professor.txt",
        "metadata": "faithfulness_global_annotation_metadata.txt",
        "black_square": "faithfulness_global_annotation_fewshot_marker.txt",
        "white_square": "faithfulness_global_annotation_fewshot_marker.txt",
    }

    prompt_file = prompt_files.get(hint_template, "faithfulness_global_annotation_professor.txt")
    prompt_path = Path("prompts") / prompt_file

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Global annotation prompt not found at {prompt_path} for hint template '{hint_template}'"
        )


def setup_openrouter_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Setup OpenRouter client for LLM judge.

    Args:
        api_key: Optional API key, otherwise uses environment variable

    Returns:
        Configured OpenAI client for OpenRouter
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def judge_faithfulness(
    biased_prompt: str,
    client: OpenAI,
    hint_template: str = "professor",
    model: str = "google/gemini-2.5-flash",
    max_retries: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Judge faithfulness of a single biased prompt using LLM.

    Args:
        biased_prompt: The full biased prompt (hinted input + model response)
        client: OpenRouter client
        hint_template: Type of hint used (e.g., "metadata", "professor")
        model: Model to use for judgment (default: gemini-2.5-flash)
        max_retries: Number of retry attempts
        verbose: Whether to print progress/debug information

    Returns:
        Dictionary with:
            - classification: "faithful", "unfaithful", or "error"
            - success: Boolean indicating if judgment succeeded
            - raw_response: Raw LLM response (for debugging)
            - api_usage: Token usage statistics (if available)
            - error: Error message (if failed)
    """
    # Load system prompt
    try:
        system_prompt = load_global_annotation_prompt(hint_template)
    except FileNotFoundError as e:
        if verbose:
            print(f"Error loading annotation prompt: {e}")
        return {
            "classification": "error",
            "success": False,
            "raw_response": None,
            "error": str(e)
        }

    # Construct user prompt with just the text to classify
    user_prompt = biased_prompt

    # Retry logic for API calls
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": os.environ.get("SITE_URL", "https://github.com"),
                    "X-Title": os.environ.get("SITE_NAME", "Faithfulness Steering")
                },
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.0,
                max_tokens=4000,
                top_p=0.1
            )

            raw_response = completion.choices[0].message.content.strip()

            # Extract token usage
            usage = completion.usage if hasattr(completion, 'usage') else None

            # Parse JSON response to extract classification
            import json
            try:
                response_json = json.loads(raw_response)
                classification = response_json.get("classification", None)

                # Validate classification
                if classification not in ["faithful", "unfaithful"]:
                    if verbose:
                        print(f"Warning: Unexpected classification '{classification}', treating as error")
                        print(f"Raw response: {raw_response}")
                    classification = "error"

            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Failed to parse JSON response: {e}")
                    print(f"Raw response: {raw_response}")
                classification = "error"

            # Mark as failed if classification is error
            success = (classification != "error")

            return {
                "classification": classification,
                "success": success,
                "raw_response": raw_response,
                "api_usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                }
            }

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                if verbose:
                    print(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Waiting 60s...")
                time.sleep(60)
            else:
                if verbose:
                    print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {
                        "classification": "error",
                        "success": False,
                        "raw_response": None,
                        "error": str(e)
                    }

    # If we exhausted all retries
    return {
        "classification": "error",
        "success": False,
        "raw_response": None,
        "error": "Max retries exceeded"
    }


def judge_batch(
    results: List[Dict[str, Any]],
    client: OpenAI,
    model: str = "google/gemini-2.5-flash",
    max_retries: int = 3,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Judge faithfulness for multiple biased prompts with rate limiting and progress tracking.

    Args:
        results: List of result dictionaries with biased prompts and metadata
        client: OpenRouter client
        model: Model to use for judgment (default: gemini-2.5-flash)
        max_retries: Maximum retry attempts per judgment
        verbose: Whether to print progress information

    Returns:
        List of judgment results, each containing:
            - classification: "faithful", "unfaithful", or "error"
            - success: Boolean
            - raw_response: Raw LLM response
            - api_usage: Token usage statistics
    """
    min_delay = ModelConfig.get_min_delay(model)

    if verbose:
        print(f"\n--- Judging {len(results)} prompts for faithfulness with {model} ---")
        print(f"Rate limit: {min_delay:.1f}s between requests")
        print(f"Estimated time: {len(results) * min_delay / 60:.1f} minutes")

    judgments = []
    start_time = time.time()
    total_tokens = 0
    error_count = 0

    for i, result in enumerate(tqdm(results, desc="Judging", disable=not verbose)):
        # Rate limiting
        if i > 0:
            elapsed = time.time() - request_start_time
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)

        request_start_time = time.time()

        # Extract hint template for THIS prompt
        hint_template = result.get('hint_template', 'professor')

        # Extract biased prompt (hinted input + generated text)
        # Priority: steered_prompt > hinted_prompt > biased_prompt > concatenated fallback
        if 'steered_prompt' in result:
            biased_prompt = result['steered_prompt']
        elif 'hinted_prompt' in result:
            biased_prompt = result['hinted_prompt']
        elif 'biased_prompt' in result:
            biased_prompt = result['biased_prompt']
        else:
            input_prompt = result.get('hinted_input_prompt', result.get('biased_input_prompt', ''))
            generated_text = result.get('hinted_generated_text', result.get('biased_generated_text', ''))
            biased_prompt = input_prompt + generated_text

        # Judge faithfulness
        judgment = judge_faithfulness(
            biased_prompt=biased_prompt,
            client=client,
            hint_template=hint_template,
            model=model,
            max_retries=max_retries,
            verbose=False  # Disable per-request verbose to avoid clutter
        )

        if judgment['success'] and judgment['classification'] != "error":
            total_tokens += judgment.get('api_usage', {}).get('total_tokens', 0)
        else:
            error_count += 1

        judgments.append(judgment)

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"\nJudgment complete in {elapsed_time/60:.1f} minutes")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Errors: {error_count}/{len(results)}")

    return judgments
