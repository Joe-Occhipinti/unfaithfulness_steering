#!/usr/bin/env python3
"""
Faithfulness Classifier for Model Chain-of-Thought Reasoning

This script analyzes model responses to determine if they are faithful about 
the influence of hints on their reasoning process.

Usage:
    1. Edit the configuration variables in the main section at the bottom
    2. Set INPUT_FILE to your JSONL file path
    3. Run: python faithfulness_classifier.py

Configuration:
    - INPUT_FILE: Path to input JSONL file with CoT answers and question prompts
    - API_KEY: DeepSeek API key (optional, uses .env file by default)
    - BASE_URL: API base URL (optional, uses .env file by default)
    - MAX_RETRIES: Number of retry attempts per API call
    - DELAY_BETWEEN_REQUESTS: Seconds to wait between API calls

Input JSONL Format Expected:
    Each line should contain:
    - question: The original question
    - hint: The hint provided (if any)  
    - cot_answer: The model's chain-of-thought answer
    - prompt: Full prompt given to model (optional)
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
    
    def classify_faithfulness(self, question: str, hint: str, cot_answer: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Classify the faithfulness of a model's CoT answer using DeepSeek V3.
        
        Args:
            question: The original question asked
            hint: The hint provided to the model
            cot_answer: The model's chain-of-thought answer
            max_retries: Maximum number of API retry attempts
            
        Returns:
            Dictionary with faithfulness classification results
        """
        # Prepare the prompt
        prompt = self.prompt_template.format(
            question=question,
            hint=hint,
            cot_answer=cot_answer
        )
        
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
            "max_tokens": 500,
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
                        "faithfulness": classification.get("faithfulness", "unknown"),
                        "reasoning": classification.get("reasoning", ""),
                        "hint_acknowledgment": classification.get("hint_acknowledgment", False),
                        "explicit_influence_statement": classification.get("explicit_influence_statement", False),
                        "api_response": classification_text,
                        "success": True
                    }
                except json.JSONDecodeError as e:
                    return {
                        "faithfulness": "unknown",
                        "reasoning": f"Failed to parse response: {e}",
                        "hint_acknowledgment": False,
                        "explicit_influence_statement": False,
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
                        "faithfulness": "unknown",
                        "reasoning": f"API request failed: {e}",
                        "hint_acknowledgment": False,
                        "explicit_influence_statement": False,
                        "api_response": None,
                        "success": False,
                        "error": str(e)
                    }
        
        return {
            "faithfulness": "unknown",
            "reasoning": "Max retries exceeded",
            "hint_acknowledgment": False,
            "explicit_influence_statement": False,
            "api_response": None,
            "success": False,
            "error": "Max retries exceeded"
        }

def process_jsonl_file(input_path: str, classifier: FaithfulnessClassifier, 
                      max_retries: int = 3, delay_between_requests: float = 0.1) -> None:
    """
    Process a JSONL file and classify faithfulness of all CoT answers.
    
    Args:
        input_path: Path to input JSONL file
        classifier: FaithfulnessClassifier instance
        max_retries: Maximum number of API retry attempts per classification
        delay_between_requests: Seconds to wait between API calls
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{input_file.stem}_faithfulness_{timestamp}.jsonl"
    output_path = Path("datasets") / output_filename
    
    # Ensure datasets directory exists
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    
    processed_count = 0
    error_count = 0
    faithful_count = 0
    unfaithful_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse input line
                data = json.loads(line.strip())
                
                # Check required fields
                required_fields = ['question', 'cot_answer']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing required fields {missing_fields}, skipping")
                    continue
                
                question = data['question']
                hint = data.get('hint', '')  # Hint might be optional
                cot_answer = data['cot_answer']
                
                # Skip if no hint provided
                if not hint or hint.strip() == '':
                    print(f"Warning: Line {line_num} has no hint, skipping faithfulness analysis")
                    continue
                
                # Classify faithfulness
                print(f"Processing line {line_num}...", end=" ")
                classification = classifier.classify_faithfulness(
                    question=question, 
                    hint=hint, 
                    cot_answer=cot_answer, 
                    max_retries=max_retries
                )
                
                # Add classification results to original data
                data['faithfulness_classification'] = {
                    'faithfulness': classification['faithfulness'],
                    'reasoning': classification['reasoning'],
                    'hint_acknowledgment': classification['hint_acknowledgment'],
                    'explicit_influence_statement': classification['explicit_influence_statement'],
                    'timestamp': datetime.now().isoformat(),
                    'success': classification['success']
                }
                
                if not classification['success']:
                    data['faithfulness_classification']['error'] = classification['error']
                    error_count += 1
                    print("ERROR")
                else:
                    # Count faithful vs unfaithful
                    if classification['faithfulness'] == 'faithful':
                        faithful_count += 1
                    elif classification['faithfulness'] == 'unfaithful':
                        unfaithful_count += 1
                    print("OK")
                
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
    print(f"Errors: {error_count} entries")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input file path (JSONL file containing CoT answers with questions and hints)
    INPUT_FILE = r"sample_faithfulness_test.jsonl"  # Change this to your input file
    
    # API Configuration (optional - will use .env file by default)
    API_KEY = None  # Set to your API key or leave None to use .env file
    BASE_URL = None  # Set to custom base URL or leave None to use .env file
    
    # Processing options
    MAX_RETRIES = 3  # Number of API retry attempts per classification
    DELAY_BETWEEN_REQUESTS = 0.5  # Seconds to wait between API calls
    
    # ========================================
    # SCRIPT EXECUTION - DON'T MODIFY BELOW
    # ========================================
    
    print("=" * 60)
    print("Model Faithfulness Classifier")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Request delay: {DELAY_BETWEEN_REQUESTS}s")
    print("-" * 60)
    
    try:
        # Initialize classifier
        classifier = FaithfulnessClassifier(api_key=API_KEY, base_url=BASE_URL)
        
        # Process the file with configurable parameters
        process_jsonl_file(INPUT_FILE, classifier, 
                          max_retries=MAX_RETRIES, 
                          delay_between_requests=DELAY_BETWEEN_REQUESTS)
        
        print("=" * 60)
        print("Faithfulness classification completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)