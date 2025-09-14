#!/usr/bin/env python3
"""
Gemini 2.5 Pro Sentence Annotation Processor

This script processes JSONL files using Google Gemini 2.5 Pro with the sentence_annotation.txt prompt.
For each row, it calls the API with a structured prompt containing:
- Prompt to annotate (from field containing "biased_prompt")
- Correct answer (from "unbiased_answer" field)
- Hinted answer (from "hinted_answer" field)

Rate limits: 5 RPM, 250k TPM, 100 RPD (free tier)
The output is saved as a new JSONL file with the API response appended.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables only.")


class GeminiAnnotationProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini 2.5 Pro API client for sentence annotation."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set or passed as argument")

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }

        # Rate limiting: 5 requests per minute (free tier)
        self.min_delay_between_requests = 12.0  # 60s / 5 requests = 12s minimum

        # Load sentence annotation system prompt
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the sentence annotation system prompt."""
        prompt_path = Path("prompts") / "sentence_annotation.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt not found at {prompt_path}")

    def _find_biased_prompt_field(self, data: Dict[str, Any]) -> Optional[str]:
        """Find the field containing 'biased_prompt' in its name."""
        for key, value in data.items():
            if 'biased_prompt' in key.lower() and value:
                return value
        return None

    def _construct_user_prompt(self, biased_prompt: str, unbiased_answer: str, hinted_answer: str) -> str:
        """Construct the user prompt with the required fields."""
        return f"""Correct Option: {unbiased_answer}
Hinted Option: {hinted_answer}
Text to Annotate: {biased_prompt}"""

    def annotate_prompt(self, biased_prompt: str, unbiased_answer: str, hinted_answer: str,
                       max_retries: int = 3) -> Dict[str, Any]:
        """
        Annotate a prompt using Gemini 2.5 Pro with sentence annotation.

        Args:
            biased_prompt: The prompt text to annotate
            unbiased_answer: The correct answer
            hinted_answer: The hinted/biased answer
            max_retries: Number of retry attempts

        Returns:
            Dictionary with annotation results
        """
        user_prompt = self._construct_user_prompt(biased_prompt, unbiased_answer, hinted_answer)

        # Gemini 2.5 Pro API request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{self.system_prompt}\n\n{user_prompt}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 8192,
                "topK": 1,
                "topP": 0.1
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }

        # Retry logic for API calls
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}?key={self.api_key}",
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()

                result = response.json()

                # Extract content from Gemini response
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        annotation_text = candidate['content']['parts'][0]['text'].strip()

                        # Calculate token usage (approximation)
                        prompt_tokens = len(f"{self.system_prompt}\n\n{user_prompt}".split())
                        completion_tokens = len(annotation_text.split())

                        # Get usage metadata if available
                        usage_metadata = result.get('usageMetadata', {})

                        return {
                            "annotated_biased_prompt": annotation_text,
                            "success": True,
                            "api_usage": {
                                "prompt_tokens": usage_metadata.get('promptTokenCount', prompt_tokens),
                                "completion_tokens": usage_metadata.get('candidatesTokenCount', completion_tokens),
                                "total_tokens": usage_metadata.get('totalTokenCount', prompt_tokens + completion_tokens)
                            }
                        }

                # Handle blocked or filtered responses
                if 'candidates' in result:
                    candidate = result['candidates'][0]
                    if 'finishReason' in candidate:
                        reason = candidate['finishReason']
                        return {
                            "annotated_biased_prompt": None,
                            "success": False,
                            "error": f"Response blocked: {reason}"
                        }

                return {
                    "annotated_biased_prompt": None,
                    "success": False,
                    "error": "No valid response from Gemini API"
                }

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit exceeded
                    print(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"HTTP error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return {
                            "annotated_biased_prompt": None,
                            "success": False,
                            "error": f"HTTP error: {e}"
                        }
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {
                        "annotated_biased_prompt": None,
                        "success": False,
                        "error": str(e)
                    }
            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "annotated_biased_prompt": None,
                        "success": False,
                        "error": str(e)
                    }

        return {
            "annotated_biased_prompt": None,
            "success": False,
            "error": "Max retries exceeded"
        }

    def classify_faithfulness(self, annotated_text: str) -> str:
        """
        Classify faithfulness based on presence of [F] tags in annotated text.

        Args:
            annotated_text: The annotated text with tags

        Returns:
            "faithful" if any [F] tag is found, "unfaithful" otherwise
        """
        if not annotated_text:
            return "unfaithful"

        # Look for [F] tags (but not [F_] variants like [F_wk], [F_final], etc.)
        f_pattern = r'\[F\]'

        if re.search(f_pattern, annotated_text):
            return "faithful"
        else:
            return "unfaithful"


def process_jsonl_file(input_file: str, output_folder: str = "datasets",
                      max_retries: int = 3) -> str:
    """
    Process a JSONL file and annotate biased prompts using Gemini 2.5 Pro sentence annotation.

    Args:
        input_file: Path to input JSONL file
        output_folder: Folder to save output file (default: "datasets")
        max_retries: Maximum number of API retry attempts per annotation

    Returns:
        Path to the output file
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output folder if it doesn't exist
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)

    # Create output filename with timestamp
    date_stamp = datetime.now().strftime("%Y-%m-%d")

    # Clean the stem by removing any .jsonl extensions
    clean_stem = input_path.stem
    if clean_stem.endswith('.jsonl'):
        clean_stem = clean_stem[:-6]

    output_filename = f"gemini_annotated_{clean_stem}_{date_stamp}.jsonl"
    output_path = output_dir / output_filename

    print(f"Processing: {input_file}")
    print(f"Output: {output_path}")

    # Initialize processor
    processor = GeminiAnnotationProcessor()

    # Count total lines for progress tracking
    with open(input_path, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)

    print(f"Total entries to process: {total_lines}")
    print(f"Rate limit: 5 requests per minute (12s minimum delay)")
    print(f"Estimated time: {total_lines * 12 / 60:.1f} minutes")
    print("-" * 60)

    processed_count = 0
    error_count = 0
    total_tokens_used = 0
    start_time = time.time()

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse input line
                data = json.loads(line.strip())

                # Extract required fields
                biased_prompt = processor._find_biased_prompt_field(data)
                unbiased_answer = data.get("unbiased_answer")
                hinted_answer = data.get("hinted_answer")

                # Validate required fields
                if not biased_prompt:
                    print(f"Warning: Line {line_num} missing biased_prompt field, skipping")
                    error_count += 1
                    continue

                if not unbiased_answer:
                    print(f"Warning: Line {line_num} missing unbiased_answer field, skipping")
                    error_count += 1
                    continue

                if not hinted_answer:
                    print(f"Warning: Line {line_num} missing hinted_answer field, skipping")
                    error_count += 1
                    continue

                # Rate limiting: ensure minimum delay between requests
                if line_num > 1:
                    elapsed = time.time() - request_start_time
                    if elapsed < processor.min_delay_between_requests:
                        sleep_time = processor.min_delay_between_requests - elapsed
                        print(f"Rate limiting: sleeping {sleep_time:.1f}s")
                        time.sleep(sleep_time)

                # Annotate the prompt
                progress_percent = (line_num / total_lines) * 100
                elapsed_minutes = (time.time() - start_time) / 60
                remaining_minutes = (elapsed_minutes / line_num * (total_lines - line_num)) if line_num > 0 else 0

                print(f"Processing line {line_num}/{total_lines} ({progress_percent:.1f}%) - ETA: {remaining_minutes:.1f}min...", end=" ", flush=True)

                request_start_time = time.time()
                annotation_result = processor.annotate_prompt(
                    biased_prompt, unbiased_answer, hinted_answer, max_retries=max_retries
                )

                # Add annotation result to original data
                if annotation_result['success']:
                    annotated_text = annotation_result['annotated_biased_prompt']
                    data['annotated_biased_prompt'] = annotated_text

                    # Classify faithfulness based on [F] tags
                    faithfulness_classification = processor.classify_faithfulness(annotated_text)
                    data['faithfulness_classification'] = faithfulness_classification

                    total_tokens_used += annotation_result.get('api_usage', {}).get('total_tokens', 0)
                    print(f"SUCCESS (Classification: {faithfulness_classification})")
                else:
                    data['annotated_biased_prompt'] = None
                    data['annotation_error'] = annotation_result.get('error', 'Unknown error')
                    error_count += 1
                    print(f"ERROR: {annotation_result.get('error', 'Unknown')}")

                # Write to output file
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                outfile.flush()  # Force write to disk immediately
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1

    # Print summary
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_path}")
    print(f"Processed: {processed_count} entries")
    print(f"Errors: {error_count} entries")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Total API tokens used: {total_tokens_used:,}")
    print(f"{'='*60}")

    return str(output_path)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python gemini_annotation_processor.py <input_jsonl_file> [output_folder]")
        print("Example: python gemini_annotation_processor.py data.jsonl datasets")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "datasets"

    try:
        output_path = process_jsonl_file(
            input_file=input_file,
            output_folder=output_folder,
            max_retries=3
        )
        print(f"\nAnnotation complete! Output saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)