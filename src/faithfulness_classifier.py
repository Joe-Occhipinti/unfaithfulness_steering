"""
faithfulness_classifier.py

Faithfulness classification using async batch infrastructure.

Provides functionality to classify whether model responses are faithful
or unfaithful using LLM-based judgment with template-specific prompts.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union
import asyncio

try:
    from .async_classifier import batch_classify, delete_checkpoint
    from .config import ModelConfig
except ImportError:
    from async_classifier import batch_classify, delete_checkpoint
    from config import ModelConfig


def load_faithfulness_prompt(hint_template: str = "professor") -> str:
    """
    Load the faithfulness annotation prompt based on hint template.
    
    Args:
        hint_template: Type of hint used (e.g., "metadata", "professor", "grader_hacking")
    
    Returns:
        Faithfulness annotation prompt string
    """
    # Map hint templates to prompt files
    prompt_files = {
        "professor": "faithfulness_global_annotation_professor.txt",
        "metadata": "faithfulness_global_annotation_metadata.txt",
        "fewshot_black_square": "faithfulness_global_annotation_fewshot_marker.txt",
        "white_square": "faithfulness_global_annotation_fewshot_marker.txt",
        "grader_hacking": "faithfulness_global_annotation_grader_hacking.txt",
        "argument": "faithfulness_global_annotation_argument.txt",
        "self-consistency": "faithfulness_global_annotation_self_consistency.txt",
    }
    
    prompt_file = prompt_files.get(hint_template, "faithfulness_global_annotation_professor.txt")
    prompt_path = Path("prompts") / prompt_file
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Faithfulness prompt not found at {prompt_path} for hint template '{hint_template}'"
        )


def parse_faithfulness(content: str) -> str:
    """
    Parse faithfulness classification from API response.
    
    Args:
        content: Raw LLM response content (expected JSON)
    
    Returns:
        "faithful", "unfaithful", or "error"
    """
    try:
        response_json = json.loads(content)
        classification = response_json.get("classification")
        
        if classification in ["faithful", "unfaithful"]:
            return classification
        return "error"
    except json.JSONDecodeError:
        return "error"


def extract_steered_prompt(record: Dict[str, Any]) -> str:
    """Extract steered prompt from record for faithfulness classification."""
    # Priority: steered_prompt > hinted_prompt > biased_prompt > concatenated fallback
    if 'steered_prompt' in record:
        return record['steered_prompt']
    elif 'hinted_prompt' in record:
        return record['hinted_prompt']
    elif 'biased_prompt' in record:
        return record['biased_prompt']
    else:
        input_prompt = record.get('hinted_input_prompt', record.get('biased_input_prompt', ''))
        generated_text = record.get('hinted_generated_text', record.get('biased_generated_text', ''))
        return input_prompt + generated_text


async def classify_faithfulness(
    records: List[Dict[str, Any]],
    hint_template: str = "professor",
    model: str = None,
    batch_size: int = 20,
    checkpoint_key: str = None,
    verbose: bool = True
) -> Dict[int, str]:
    """
    Classify faithfulness for a list of records using async batched processing.
    
    Args:
        records: List of records with steered_prompt field
        hint_template: Hint template type for selecting appropriate prompt
        model: Model to use for classification
        batch_size: Number of concurrent requests
        checkpoint_key: Optional checkpoint key for resume capability
        verbose: Print progress information
    
    Returns:
        Dictionary mapping record index to faithfulness result ("faithful", "unfaithful", or "error")
    """
    if model is None:
        model = ModelConfig.ANNOTATION_MODELS.get("gemini-2.5-flash", "google/gemini-2.5-flash")
    
    if verbose:
        print(f"\n  --- Classifying {len(records)} responses for faithfulness ---")
        print(f"  Hint template: {hint_template}")
    
    # Load the appropriate system prompt
    try:
        system_prompt = load_faithfulness_prompt(hint_template)
    except FileNotFoundError as e:
        print(f"  Error loading prompt: {e}")
        return {i: "error" for i in range(len(records))}
    
    results = await batch_classify(
        records=records,
        content_extractor=extract_steered_prompt,
        system_prompt=system_prompt,
        parse_fn=parse_faithfulness,
        model=model,
        batch_size=batch_size,
        checkpoint_key=checkpoint_key,
        verbose=verbose
    )
    
    # Clean up checkpoint on completion
    if checkpoint_key:
        delete_checkpoint(checkpoint_key)
    
    return results


def run_faithfulness_classification(
    records: List[Dict[str, Any]],
    hint_template: str = "professor",
    model: str = None,
    batch_size: int = 20,
    checkpoint_key: str = None,
    verbose: bool = True
) -> Dict[int, str]:
    """
    Synchronous wrapper for faithfulness classification.
    
    Args:
        records: List of records with steered_prompt field
        hint_template: Hint template type for selecting appropriate prompt
        model: Model to use for classification
        batch_size: Number of concurrent requests
        checkpoint_key: Optional checkpoint key for resume capability
        verbose: Print progress information
    
    Returns:
        Dictionary mapping record index to faithfulness result
    """
    return asyncio.run(classify_faithfulness(
        records=records,
        hint_template=hint_template,
        model=model,
        batch_size=batch_size,
        checkpoint_key=checkpoint_key,
        verbose=verbose
    ))
