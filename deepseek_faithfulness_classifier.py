#!/usr/bin/env python3
"""
DeepSeek R1 Faithfulness Classifier

This script uses DeepSeek R1 (deepseek-reasoner) to classify prompts in a JSONL file
as faithful or unfaithful using the faithfulness classification prompt.
The input prompt is inserted into the {prompt} placeholder of the faithfulness template.

Usage:
    1. Edit the configuration variables in the main section at the bottom
    2. Set INPUT_FILE to your input JSONL file path
    3. Run: python deepseek_classifier_changed_answers.py

Configuration:
    - INPUT_FILE: Path to input JSONL file
    - API_KEY: DeepSeek API key (optional, uses .env file by default)
    - BASE_URL: API base URL (optional, uses .env file by default)
    - MAX_RETRIES: Number of retry attempts per API call
    - DELAY_BETWEEN_REQUESTS: Seconds to wait between API calls
"""

import json
import os
import sys
import glob
from typing import Dict, Any, Optional, List
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

class DeepSeekClassifier:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize DeepSeek API client."""
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable must be set or passed as argument")
        
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Load classification prompt
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the classification prompt from the prompts folder."""
        prompt_path = Path("prompts") / "faithfulness.txt"
        if not prompt_path.exists():
            # Fallback to faithfulness_prompt.txt if faithfulness.txt doesn't exist
            prompt_path = Path("prompts") / "faithfulness_prompt.txt"
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    
    def classify_prompt(self, prompt_text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Classify a prompt with DeepSeek R1.
        Text will be inserted into the {prompt} placeholder in the prompt template.
        
        Args:
            prompt_text: The input prompt to classify
            max_retries: Number of retry attempts
            
        Returns:
            Dictionary with classification results
        """
        # Prepare the prompt - prompt_text goes into {prompt} placeholder
        prompt = self.prompt_template.format(prompt=prompt_text)
        
        # API request payload
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.0,  # Deterministic for classification
            "max_tokens": 512,
            "stream": False
        }
        
        # Retry logic for API calls
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30
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
                    faithfulness = classification.get("classification_faithfulness")
                    return {
                        "classification_faithfulness": faithfulness,
                        "api_response": classification_text,
                        "success": True
                    }
                except json.JSONDecodeError as e:
                    return {
                        "faithfulness": None,
                        "api_response": classification_text,
                        "success": False,
                        "error": f"JSON parsing failed: {e}"
                    }
                
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {
                        "faithfulness": None,
                        "api_response": None,
                        "success": False,
                        "error": str(e)
                    }
        
        return {
            "faithfulness": None,
            "api_response": None,
            "success": False,
            "error": "Max retries exceeded"
        }

def process_jsonl_file(input_path: str, classifier: DeepSeekClassifier, 
                      max_retries: int = 3, delay_between_requests: float = 0.1) -> None:
    """
    Process a JSONL file and classify prompts for faithfulness.
    
    Args:
        input_path: Path to input JSONL file
        classifier: DeepSeekClassifier instance
        max_retries: Maximum number of API retry attempts per classification
        delay_between_requests: Seconds to wait between API calls
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output filename with "classified_" prefix
    output_filename = f"classified_{input_file.name}"
    output_path = input_file.parent / output_filename
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse input line
                data = json.loads(line.strip())
                
                # Extract prompt text - assume there's a field containing the prompt to classify
                # This could be 'prompt', 'text', 'content', etc. - adapt as needed
                prompt_text = None
                for field in ['prompt', 'text', 'content', 'original_prompt', 'steered_prompt_complete']:
                    if field in data and data[field]:
                        prompt_text = data[field]
                        break
                
                if not prompt_text:
                    print(f"Warning: Line {line_num} missing prompt field, skipping")
                    error_count += 1
                    continue
                
                # Classify the prompt
                print(f"Processing line {line_num}...", end=" ")
                classification = classifier.classify_prompt(prompt_text, max_retries=max_retries)
                
                # Add classification results to original data
                data['classification_steered'] = classification['classification_faithfulness']
                
                if not classification['success']:
                    error_count += 1
                    print("ERROR")
                else:
                    print(f"OK (Classification: {classification.get('classification_faithfulness', 'None')})")
                
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
    print(f"Errors: {error_count} entries")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input configuration - specify pattern to match multiple files
    INPUT_PATTERN = r"C:\Users\l440\Downloads\fullsweep_layerwise_steered_prompts_mmlu_psy_val_2025-09-04\*.jsonl"
    # Alternative examples:
    # INPUT_PATTERN = "your_folder/*.jsonl"
    # INPUT_PATTERN = "**/*.jsonl"  # Recursive search
    
    # API Configuration (optional - will use .env file by default)
    API_KEY = None  # Set to your API key or leave None to use .env file
    BASE_URL = None  # Set to custom base URL or leave None to use .env file
    
    # Processing options
    MAX_RETRIES = 5  # Number of API retry attempts per classification
    DELAY_BETWEEN_REQUESTS = 1  # Seconds to wait between API calls
    
    # ========================================
    # SCRIPT EXECUTION - DON'T MODIFY BELOW
    # ========================================
    
    # Find all matching files
    input_files = glob.glob(INPUT_PATTERN)
    if not input_files:
        print(f"No files found matching pattern: {INPUT_PATTERN}")
        sys.exit(1)
    
    print("=" * 60)
    print("DeepSeek R1 Faithfulness Classifier - Batch Mode")
    print("=" * 60)
    print(f"Pattern: {INPUT_PATTERN}")
    print(f"Found {len(input_files)} files to process")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Request delay: {DELAY_BETWEEN_REQUESTS}s")
    print("-" * 60)
    
    try:
        # Initialize classifier
        classifier = DeepSeekClassifier(api_key=API_KEY, base_url=BASE_URL)
        
        # Process each file
        total_processed = 0
        total_errors = 0
        
        for i, input_file in enumerate(input_files, 1):
            print(f"\n[{i}/{len(input_files)}] Processing: {input_file}")
            try:
                process_jsonl_file(input_file, classifier, 
                                  max_retries=MAX_RETRIES, 
                                  delay_between_requests=DELAY_BETWEEN_REQUESTS)
                total_processed += 1
                print(f"✓ Completed: {input_file}")
            except Exception as e:
                print(f"✗ Failed: {input_file} - Error: {e}")
                total_errors += 1
        
        print("=" * 60)
        print("Batch Classification Summary:")
        print(f"Files processed successfully: {total_processed}")
        print(f"Files with errors: {total_errors}")
        print(f"Total files: {len(input_files)}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)