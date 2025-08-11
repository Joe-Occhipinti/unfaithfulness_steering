#!/usr/bin/env python3
"""
DeepSeek V3 Answer Classifier for Changed Answers

This script uses DeepSeek V3 to classify the changed answers dataset.
The concatenated "original_prompt" + "generated_biased_answer" text is inserted 
into the {generated_answer} placeholder of the classification prompt template.

Usage:
    1. Edit the configuration variables in the main section at the bottom
    2. Set INPUT_FILE to your changed answers JSONL file path
    3. Run: python deepseek_classifier_changed_answers.py

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
        prompt_path = r"C:\Users\l440\unfaithfulness_steering\prompts\faithfulness_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    
    def classify_answer(self, combined_text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Classify a combined text (original_prompt + generated_biased_answer) using DeepSeek V3.
        The combined_text will be inserted into the {generated_answer} placeholder in the prompt template.
        
        Args:
            combined_text: The combined text (original_prompt + generated_biased_answer)
            max_retries: Maximum number of API retry attempts
            
        Returns:
            Dictionary with classification results
        """
        # Prepare the prompt - combined_text goes into {generated_answer} placeholder
        prompt = self.prompt_template.format(generated_answer=combined_text)
        
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
            "max_tokens": 150,
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
                    return {
                        "format_compliant": classification.get("format_compliant", False),
                        "extracted_letter": classification.get("extracted_letter"),
                        "api_response": classification_text,
                        "success": True
                    }
                except json.JSONDecodeError as e:
                    return {
                        "format_compliant": False,
                        "extracted_letter": None,
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
                        "format_compliant": False,
                        "extracted_letter": None,
                        "api_response": None,
                        "success": False,
                        "error": str(e)
                    }
        
        return {
            "format_compliant": False,
            "extracted_letter": None,
            "api_response": None,
            "success": False,
            "error": "Max retries exceeded"
        }

def process_changed_answers_file(input_path: str, classifier: DeepSeekClassifier, 
                                max_retries: int = 3, delay_between_requests: float = 0.1) -> None:
    """
    Process a changed answers JSONL file and classify combined texts.
    
    Args:
        input_path: Path to changed answers JSONL file
        classifier: DeepSeekClassifier instance
        max_retries: Maximum number of API retry attempts per classification
        delay_between_requests: Seconds to wait between API calls
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{input_file.stem}_deepseek_classified_{timestamp}.jsonl"
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
                
                # Get the required fields
                original_prompt = data.get('original_prompt', '')
                generated_biased_answer = data.get('generated_biased_answer', '')
                
                if not original_prompt or not generated_biased_answer:
                    print(f"Warning: Line {line_num} missing required fields, skipping")
                    error_count += 1
                    continue
                
                # Combine the texts for the {generated_answer} placeholder
                combined_text = original_prompt + "\n\n" + generated_biased_answer
                
                # Classify the combined text (this goes into {generated_answer} in the prompt template)
                print(f"Processing line {line_num}...", end=" ")
                classification = classifier.classify_answer(combined_text, max_retries=max_retries)
                
                # Add classification results to original data
                data['deepseek_classification'] = {
                    'format_compliant': classification['format_compliant'],
                    'extracted_letter': classification['extracted_letter'],
                    'timestamp': datetime.now().isoformat(),
                    'success': classification['success'],
                    'combined_text_length': len(combined_text)
                }
                
                if not classification['success']:
                    data['deepseek_classification']['error'] = classification['error']
                    error_count += 1
                    print("ERROR")
                else:
                    print(f"OK (Letter: {classification.get('extracted_letter', 'None')})")
                
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
    
    # Input file path (Changed answers JSONL file)
    INPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_changed_answers_only_faithfulness_classified_2025-08-11_14-09-37.jsonl"
    
    # API Configuration (optional - will use .env file by default)
    API_KEY = None  # Set to your API key or leave None to use .env file
    BASE_URL = None  # Set to custom base URL or leave None to use .env file
    
    # Processing options
    MAX_RETRIES = 5  # Number of API retry attempts per classification
    DELAY_BETWEEN_REQUESTS = 0.5  # Seconds to wait between API calls
    
    # ========================================
    # SCRIPT EXECUTION - DON'T MODIFY BELOW
    # ========================================
    
    print("=" * 60)
    print("DeepSeek V3 Changed Answers Classifier")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Request delay: {DELAY_BETWEEN_REQUESTS}s")
    print("-" * 60)
    
    try:
        # Initialize classifier
        classifier = DeepSeekClassifier(api_key=API_KEY, base_url=BASE_URL)
        
        # Process the changed answers file
        process_changed_answers_file(INPUT_FILE, classifier, 
                                   max_retries=MAX_RETRIES, 
                                   delay_between_requests=DELAY_BETWEEN_REQUESTS)
        
        print("=" * 60)
        print("Classification completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)