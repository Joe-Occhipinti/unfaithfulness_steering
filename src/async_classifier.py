"""
async_classifier.py

Generic async batched classifier infrastructure for LLM API calls.

Provides:
- Async batch processing with configurable batch size
- Checkpointing for resume capability
- Rate limiting with semaphore
- Retry logic with exponential backoff
- Progress tracking via tqdm
"""

import json
import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Callable, Union, Optional
import aiohttp
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 3  # seconds

# Checkpoint directory
CHECKPOINT_DIR = Path("data/checkpoints")


def get_checkpoint_dir() -> Path:
    """Get checkpoint directory, creating if needed."""
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    return CHECKPOINT_DIR


def save_checkpoint(checkpoint_key: str, classifications: Dict[int, Any]):
    """Save classification progress to checkpoint file."""
    checkpoint_path = get_checkpoint_dir() / f"{checkpoint_key}_checkpoint.json"
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        # Convert keys to strings for JSON, handle bool values properly
        serializable = {str(k): v for k, v in classifications.items()}
        json.dump(serializable, f)


def load_checkpoint(checkpoint_key: str) -> Dict[int, Any]:
    """Load classification progress from checkpoint file."""
    checkpoint_path = get_checkpoint_dir() / f"{checkpoint_key}_checkpoint.json"
    if not checkpoint_path.exists():
        return {}
    
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        serializable = json.load(f)
        return {int(k): v for k, v in serializable.items()}


def delete_checkpoint(checkpoint_key: str):
    """Delete checkpoint file after successful completion."""
    checkpoint_path = get_checkpoint_dir() / f"{checkpoint_key}_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


async def call_llm_api(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_content: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int = DEFAULT_RETRY_DELAY
) -> str:
    """
    Make a single async LLM API call.
    
    Args:
        session: aiohttp session
        system_prompt: System prompt for the LLM
        user_content: User message content
        model: Model identifier
        semaphore: Semaphore for rate limiting
        max_retries: Maximum retry attempts
        retry_delay: Base delay between retries
    
    Returns:
        Raw response content string, or "API_ERROR" on failure
    """
    async with semaphore:
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
                
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": os.environ.get("SITE_URL", "https://github.com"),
                        "X-Title": os.environ.get("SITE_NAME", "Faithfulness Steering")
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    elif response.status == 429:  # Rate limit
                        wait_time = retry_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        print(f"API error {response.status}: {error_text[:200]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                        return "API_ERROR"
                        
            except asyncio.TimeoutError:
                print(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                return "API_ERROR"
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                return "API_ERROR"
        
        return "API_ERROR"


async def process_batch(
    session: aiohttp.ClientSession,
    batch: List[tuple],  # List of (idx, content)
    system_prompt: str,
    model: str,
    parse_fn: Callable[[str], Any],
    semaphore: asyncio.Semaphore
) -> Dict[int, Any]:
    """Process a batch of records concurrently."""
    tasks = [
        call_llm_api(session, system_prompt, content, model, semaphore)
        for idx, content in batch
    ]
    raw_results = await asyncio.gather(*tasks)
    
    parsed_results = {}
    for (idx, _), raw_response in zip(batch, raw_results):
        if raw_response == "API_ERROR":
            parsed_results[idx] = "error"
        else:
            parsed_results[idx] = parse_fn(raw_response)
    
    return parsed_results


async def batch_classify(
    records: List[Dict[str, Any]],
    content_extractor: Callable[[Dict], str],
    system_prompt: str,
    parse_fn: Callable[[str], Any],
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_key: Optional[str] = None,
    verbose: bool = True
) -> Dict[int, Any]:
    """
    Generic async batch classifier.
    
    Args:
        records: List of records to classify
        content_extractor: Function to extract content from record for LLM
        system_prompt: System prompt for the classifier
        parse_fn: Function to parse LLM response into result value
        model: Model identifier for OpenRouter
        batch_size: Number of concurrent requests
        checkpoint_key: Optional key for checkpointing (enables resume)
        verbose: Print progress information
    
    Returns:
        Dictionary mapping record index to classification result
    """
    # Load checkpoint if exists
    classifications = {}
    if checkpoint_key:
        classifications = load_checkpoint(checkpoint_key)
        if classifications and verbose:
            print(f"  Resuming from checkpoint: {len(classifications)} already classified")
    
    # Create batches only for unclassified records
    batches = []
    current_batch = []
    
    for idx, record in enumerate(records):
        if idx in classifications:
            continue  # Skip already classified
        
        content = content_extractor(record)
        current_batch.append((idx, content))
        
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    
    if current_batch:
        batches.append(current_batch)
    
    if not batches:
        if verbose:
            print(f"  All {len(records)} records already classified from checkpoint")
        return classifications
    
    remaining = sum(len(b) for b in batches)
    if verbose:
        print(f"  Processing {remaining} records in {len(batches)} batches...")
    
    # Process batches
    semaphore = asyncio.Semaphore(batch_size)
    
    async with aiohttp.ClientSession() as session:
        for batch_idx, batch in enumerate(tqdm(batches, desc="  Classifying", disable=not verbose)):
            batch_results = await process_batch(
                session, batch, system_prompt, model, parse_fn, semaphore
            )
            classifications.update(batch_results)
            
            # Save checkpoint every 5 batches
            if checkpoint_key and (batch_idx + 1) % 5 == 0:
                save_checkpoint(checkpoint_key, classifications)
            
            # Small delay between batches to avoid rate limits
            if batch_idx < len(batches) - 1:
                await asyncio.sleep(0.5)
    
    # Final checkpoint save
    if checkpoint_key:
        save_checkpoint(checkpoint_key, classifications)
    
    return classifications
