#!/usr/bin/env python3
"""
Faithfulness Classifier for Changed Answers Dataset

This script analyzes the changed answers dataset to determine if the model responses
are faithful about the influence of biased hints on their reasoning process.

The concatenated "original_prompt" + "generated_biased_answer" text is inserted 
into the {prompt} placeholder of the faithfulness classification prompt template.

Usage:
    1. Edit the configuration variables in the main section at the bottom
    2. Set INPUT_FILE to your changed answers JSONL file path
    3. Run: python faithfulness_classifier_changed_answers.py

Configuration:
    - INPUT_FILE: Path to changed answers JSONL file
    - API_KEY: DeepSeek API key (optional, uses .env file by default)
    - BASE_URL: API base URL (optional, uses .env file by default)
    - MAX_RETRIES: Number of retry attempts per API call
    - DELAY_BETWEEN_REQUESTS: Seconds to wait between API calls
"""

import json
import os
import sys
from typing import Dict, Any, Optional
import requests
import time
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables only.")

class FaithfulnessClassifier:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize DeepSeek API client for faithfulness classification."""
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable must be set or passed as argument")
        
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Load faithfulness classification prompt
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the faithfulness classification prompt from the prompts folder."""
        prompt_path = Path(__file__).parent / "prompts" / "faithfulness_prompt.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Faithfulness prompt template not found at {prompt_path}")
    
    def classify_faithfulness(self, combined_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Classify the faithfulness of a combined prompt (original_prompt + generated_biased_answer).
        The combined_prompt will be inserted into the {prompt} placeholder in the template.
        
        Args:
            combined_prompt: The combined text (original_prompt + generated_biased_answer)
            max_retries: Maximum number of API retry attempts
            
        Returns:
            Dictionary with faithfulness classification results
        """
        # Prepare the prompt - combined_prompt goes into {prompt} placeholder
        prompt = self.prompt_template.format(prompt=combined_prompt)
        
        # API request payload
        payload = {
            "model": "deepseek-chat",  # DeepSeek V3 model
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.0,  # Deterministic for classification
            "max_tokens": 300,
            "stream": False
        }
        
        # Retry logic for API calls
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60  # Increased timeout to 60 seconds
                )
                response.raise_for_status()
                
                result = response.json()
                classification_text = result["choices"][0]["message"]["content"].strip()
                
                # Parse JSON response (handle markdown code blocks)
                json_content = classification_text
                if classification_text.startswith('```json'):
                    # Extract JSON from markdown code block
                    lines = classification_text.split('\n')
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip() == '```json':
                            in_json = True
                            continue
                        elif line.strip() == '```':
                            break
                        elif in_json:
                            json_lines.append(line)
                    json_content = '\n'.join(json_lines)
                
                try:
                    classification = json.loads(json_content)
                    return {
                        "faithfulness": classification.get("faithfulness"),
                        "reasoning": classification.get("reasoning"),
                        "api_response": classification_text,
                        "success": True
                    }
                except json.JSONDecodeError as e:
                    return {
                        "faithfulness": None,
                        "reasoning": None,
                        "api_response": classification_text,
                        "success": False,
                        "error": f"JSON parsing failed: {e}"
                    }
                
            except requests.exceptions.RequestException as e:
                error_type = type(e).__name__
                print(f"API request failed (attempt {attempt + 1}/{max_retries}): {error_type} - {e}")
                if attempt < max_retries - 1:
                    backoff_time = min(2 ** attempt, 30)  # Cap backoff at 30 seconds
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                else:
                    print(f"Max retries exceeded. Final error: {e}")
                    return {
                        "faithfulness": None,
                        "reasoning": None,
                        "api_response": None,
                        "success": False,
                        "error": f"{error_type}: {str(e)}"
                    }
        
        return {
            "faithfulness": None,
            "reasoning": None,
            "api_response": None,
            "success": False,
            "error": "Max retries exceeded"
        }

def process_changed_answers_file(input_path: str, classifier: FaithfulnessClassifier, 
                                max_retries: int = 3, delay_between_requests: float = 0.1) -> None:
    """
    Process a changed answers JSONL file and classify faithfulness.
    
    Args:
        input_path: Path to changed answers JSONL file
        classifier: FaithfulnessClassifier instance
        max_retries: Maximum number of API retry attempts per classification
        delay_between_requests: Seconds to wait between API calls
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{input_file.stem}_faithfulness_classified_{timestamp}.jsonl"
    output_path = input_file.parent / output_filename
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    
    processed_count = 0
    error_count = 0
    faithful_count = 0
    unfaithful_count = 0
    null_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse input line
                data = json.loads(line.strip())
                
                # Get the required fields
                original_prompt = data.get('original_prompt', '')
                generated_biased_answer = data.get('generated_biased_answer', '')
                
                if not original_prompt or not generated_biased_answer:
                    print(f"Warning: Line {line_num} missing required fields, skipping")
                    error_count += 1
                    continue
                
                # Combine the texts for the {prompt} placeholder
                combined_prompt = original_prompt + "\n\n" + generated_biased_answer
                
                # Classify faithfulness
                print(f"Processing line {line_num}...", end=" ")
                classification = classifier.classify_faithfulness(combined_prompt, max_retries=max_retries)
                
                # Add faithfulness classification results to original data
                data['faithfulness_classification'] = {
                    'faithfulness': classification['faithfulness'],
                    'reasoning': classification['reasoning'],
                    'timestamp': datetime.now().isoformat(),
                    'success': classification['success'],
                    'combined_prompt_length': len(combined_prompt)
                }
                
                if not classification['success']:
                    data['faithfulness_classification']['error'] = classification['error']
                    error_count += 1
                    print("ERROR")
                else:
                    faithfulness_result = classification.get('faithfulness', 'unknown')
                    if faithfulness_result == 'faithful':
                        faithful_count += 1
                    elif faithfulness_result == 'unfaithful':
                        unfaithful_count += 1
                    elif faithfulness_result == 'null':
                        null_count += 1
                    print(f"OK ({faithfulness_result})")
                
                # Write to output file
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                outfile.flush()  # Force write to disk immediately
                processed_count += 1
                
                # Configurable delay to avoid rate limiting
                time.sleep(delay_between_requests)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} entries")
    print(f"Faithful: {faithful_count} entries")
    print(f"Unfaithful: {unfaithful_count} entries") 
    print(f"Null: {null_count} entries")
    print(f"Errors: {error_count} entries")
    if processed_count > 0:
        faithful_rate = faithful_count / processed_count * 100
        unfaithful_rate = unfaithful_count / processed_count * 100
        null_rate = null_count / processed_count * 100
        print(f"Faithful rate: {faithful_rate:.1f}%")
        print(f"Unfaithful rate: {unfaithful_rate:.1f}%")
        print(f"Null rate: {null_rate:.1f}%")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input file path (Changed answers JSONL file)
    INPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_changed_answers_only.jsonl"
    
    # API Configuration (optional - will use .env file by default)
    API_KEY = None  # Set to your API key or leave None to use .env file
    BASE_URL = None  # Set to custom base URL or leave None to use .env file
    
    # Processing options
    MAX_RETRIES = 3  # Number of API retry attempts per classification
    DELAY_BETWEEN_REQUESTS = 1.0  # Seconds to wait between API calls
    
    # ========================================
    # SCRIPT EXECUTION - DON'T MODIFY BELOW
    # ========================================
    
    print("=" * 60)
    print("Faithfulness Classifier for Changed Answers")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Request delay: {DELAY_BETWEEN_REQUESTS}s")
    print("-" * 60)
    
    try:
        # Initialize classifier
        classifier = FaithfulnessClassifier(api_key=API_KEY, base_url=BASE_URL)
        
        # Process the changed answers file
        process_changed_answers_file(INPUT_FILE, classifier, 
                                   max_retries=MAX_RETRIES, 
                                   delay_between_requests=DELAY_BETWEEN_REQUESTS)
        
        print("=" * 60)
        print("Faithfulness classification completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)