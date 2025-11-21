"""
activations_mean_pooled.py

Extract hidden state activations using mean pooling across all tokens in tagged sentences.
Instead of extracting only the last period token before closing tags, this extracts all tokens
within tagged sections and averages their activations.

Based on: src/activations.py
"""

import torch
import json
import os
import re
from tqdm import tqdm
import gc
from typing import Dict, List, Any, Tuple
from .model import load_model_for_forward_pass


class BlockOutputWrapper(torch.nn.Module):
    """A wrapper to intercept and save the hidden states of a model layer."""
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.last_hidden_state = output[0]
        return output

    def reset(self):
        self.last_hidden_state = None


def wrap_model_layers(model):
    """
    Wrap model layers for activation extraction.

    Args:
        model: Loaded model

    Returns:
        model with wrapped layers
    """
    print("\n--- Preparing model for activation extraction... ---")
    try:
        layers_path = model.model.layers
        for i, layer in enumerate(layers_path):
            layers_path[i] = BlockOutputWrapper(layer)
        print(f"Wrapped {len(layers_path)} layers successfully.")
    except AttributeError:
        print(f"Error: Could not find 'model.model.layers' in model. Adjust the path.")
        raise

    return model


def load_prompts_from_file(filename: str) -> List[Dict[str, Any]]:
    """Loads a list of JSON objects from a JSONL file."""
    data_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in {filename}")
    return data_list


def get_clean_text_and_char_ranges(annotated_text: str, target_tags: List[str]) -> Tuple[str, Dict[str, List[Tuple[int, int]]]]:
    """
    Strips all [TAG] style annotations and calculates character ranges
    for entire content between tag pairs in the clean text.

    MODIFIED from get_clean_text_and_char_indices():
    - No period requirement
    - Returns (start, end) character ranges instead of single positions
    - Handles multi-sentence and non-period-ending content

    Args:
        annotated_text: Text with [TAG]content[/TAG] annotations
        target_tags: List of tags to extract (e.g., ["F_final", "U_final"])

    Returns:
        clean_text: Text with all tags removed
        char_ranges: {tag: [(start_char, end_char), ...]} in clean text
    """
    char_ranges = {tag: [] for tag in target_tags}

    # 1. Find all tags to calculate character offsets
    all_tags_pattern = re.compile(r'\[/?[\w_]+\]')
    tag_matches = list(all_tags_pattern.finditer(annotated_text))

    def get_offset(annotated_char_pos):
        """Calculates the total length of all tags preceding a character position."""
        offset = 0
        for match in tag_matches:
            if match.start() < annotated_char_pos:
                offset += len(match.group(0))
            else:
                break
        return offset

    # 2. Find content between tag pairs and map to clean text positions
    for tag in target_tags:
        # CHANGED: No period requirement, capture entire content
        # re.DOTALL allows . to match newlines (for multi-line tagged content)
        pattern = re.compile(fr'\[{tag}\](.*?)\[/{tag}\]', re.DOTALL)
        for match in pattern.finditer(annotated_text):
            # Get content boundaries in annotated text
            content_start_annotated = match.start(1)  # Start of group 1 (content)
            content_end_annotated = match.end(1) - 1  # Last character of content (inclusive)

            # Skip empty content
            if content_start_annotated > content_end_annotated:
                continue

            # Calculate offsets from preceding tags
            offset_start = get_offset(content_start_annotated)
            offset_end = get_offset(content_end_annotated)

            # Map to clean text positions
            content_start_clean = content_start_annotated - offset_start
            content_end_clean = content_end_annotated - offset_end

            char_ranges[tag].append((content_start_clean, content_end_clean))

    # 3. Create the clean text by removing all tags
    clean_text = all_tags_pattern.sub('', annotated_text)

    return clean_text, char_ranges


def extract_activations_mean_pooled(
    jsonl_filename: str,
    prompt_field: str,
    output_dir: str,
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    target_tags: List[str] = None,
    layers_to_extract: List[int] = None,
    verbose: bool = True
) -> None:
    """
    Extract activations using mean pooling across all tokens in tagged sentences.

    MODIFIED from extract_activations_from_annotated_prompts():
    - Uses get_clean_text_and_char_ranges() instead of get_clean_text_and_char_indices()
    - Maps character ranges to token ranges
    - Extracts all tokens in each range
    - Averages activations across tokens (mean pooling)
    - Output format remains identical: one activation vector per tagged section

    Args:
        jsonl_filename: Path to JSONL file with annotated prompts
        prompt_field: Field in JSONL containing annotated text
        output_dir: Directory to save activation files
        model_id: Hugging Face model ID
        target_tags: Tags to extract activations for
        layers_to_extract: Layer indices to extract from
        verbose: Print debug information
    """
    # Default parameters
    if target_tags is None:
        target_tags = ["F_final", "U_final", "F_body", "U_body"]
    if layers_to_extract is None:
        layers_to_extract = list(range(32))  # Full sweep for DeepSeek

    # Setup
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"--- Configuration ---\nModel: {model_id}\nInput File: {jsonl_filename}\nPrompt Field: {prompt_field}\nOutput Directory: {output_dir}\nTarget Tags: {target_tags}\nLayers: All {len(layers_to_extract)}\nExtraction Mode: MEAN POOLING\n---------------------")

    # Load model and tokenizer
    model, tokenizer = load_model_for_forward_pass(model_id)

    # Wrap layers for activation extraction
    model = wrap_model_layers(model)

    # Get device
    device = next(model.parameters()).device

    # Load prompts
    print(f"\n--- Loading prompts from {jsonl_filename}... ---")
    prompts_data = load_prompts_from_file(jsonl_filename)
    print(f"Loaded {len(prompts_data)} prompts to process.")

    # Main extraction loop
    for i, data_item in enumerate(tqdm(prompts_data, desc="Extracting Activations (Mean Pooled)")):
        annotated_text = data_item.get(prompt_field)
        if not annotated_text:
            continue

        # 1. Generate clean text and get target character ranges
        clean_text, char_ranges = get_clean_text_and_char_ranges(annotated_text, target_tags)

        if not any(char_ranges.values()):
            continue

        # 2. Tokenize the CLEAN text (keep on CPU to preserve encoding mappings)
        inputs_cpu = tokenizer(clean_text, return_tensors="pt")

        token_count = len(inputs_cpu['input_ids'][0])
        if verbose:
            print(f"\n=== DEBUGGING PROMPT {i} (MEAN POOLING) ===")
            print(f"Annotated text length: {len(annotated_text)}")
            print(f"Clean text length: {len(clean_text)}")
            print(f"Token count: {token_count}")
            print(f"Annotated text preview: '{annotated_text[:200]}...'")

            tokens = tokenizer.convert_ids_to_tokens(inputs_cpu['input_ids'][0])
            print(f"First 10 tokens: {tokens[:10]}")
            print(f"Last 10 tokens: {tokens[-10:]}")

        # 3. Convert character ranges to token ranges
        token_ranges = {tag: [] for tag in target_tags}
        for tag, char_range_list in char_ranges.items():
            if verbose:
                print(f"\nProcessing tag '{tag}' with {len(char_range_list)} character ranges")

            for (start_char, end_char) in char_range_list:
                if verbose:
                    # Show context
                    if start_char < len(clean_text) and end_char < len(clean_text):
                        content = clean_text[start_char:end_char+1]
                        print(f"  Char range [{start_char}, {end_char}]: '{content[:50]}...'")
                    else:
                        print(f"  ERROR: Char range [{start_char}, {end_char}] out of bounds for clean text length {len(clean_text)}")
                        continue

                # Map both start and end to token indices
                start_token = inputs_cpu.char_to_token(0, start_char)
                end_token = inputs_cpu.char_to_token(0, end_char)

                if verbose:
                    print(f"  Token range: [{start_token}, {end_token}]")

                if start_token is not None and end_token is not None:
                    if start_token >= token_count or end_token >= token_count:
                        if verbose:
                            print(f"  ERROR: Token range [{start_token}, {end_token}] out of bounds for token count {token_count}")
                        continue

                    token_ranges[tag].append((start_token, end_token))
                else:
                    if verbose:
                        print(f"  WARNING: char_to_token returned None for range [{start_char}, {end_char}]")

        if verbose:
            print(f"\nFinal token ranges: {token_ranges}")
            print("=== END DEBUGGING ===")

        # 4. Move inputs to device for forward pass
        inputs = inputs_cpu.to(device)

        # 5. Run the forward pass on the CLEAN text
        with torch.no_grad():
            model(**inputs)

        prompt_activations = {layer_idx: {} for layer_idx in layers_to_extract}

        # 6. Extract and average activations for each tag's token ranges
        for layer_idx in layers_to_extract:
            wrapped_layer = model.model.layers[layer_idx]
            hidden_state_output = wrapped_layer.last_hidden_state

            if verbose:
                print(f"\nLayer {layer_idx} debug:")
                print(f"  Hidden state shape: {hidden_state_output.shape if hasattr(hidden_state_output, 'shape') else 'No shape attr'}")

            # Hidden state is [seq_len, hidden_dim]
            full_activations_tensor = hidden_state_output

            for tag, ranges in token_ranges.items():
                if ranges:
                    # Store each sentence's averaged activation individually
                    individual_activations = []
                    for (start_token, end_token) in ranges:
                        # Extract all tokens in range (inclusive)
                        span_activations = full_activations_tensor[start_token:end_token+1, :]  # Shape: [n_tokens, hidden_dim]

                        # Mean pooling across tokens
                        mean_activation = span_activations.mean(dim=0)  # Shape: [hidden_dim]

                        individual_activations.append(mean_activation.cpu())

                        if verbose and layer_idx == 0:  # Only print for first layer to avoid spam
                            print(f"  Tag '{tag}': Averaged {span_activations.shape[0]} tokens -> {mean_activation.shape}")

                    prompt_activations[layer_idx][tag] = individual_activations

            wrapped_layer.reset()

        # 7. Save the activations for the current prompt
        save_path = os.path.join(output_dir, f"prompt_{i}_activations.pt")
        torch.save(prompt_activations, save_path)

        # Clean up memory after each prompt
        gc.collect()
        torch.cuda.empty_cache()

    print("\n--- Activation extraction (mean pooled) complete! ---")


# Reuse utility functions from original activations.py
# These don't need modification as they work on the saved .pt files
from .activations import (
    get_activation_statistics,
    print_activation_statistics,
    build_activation_dataset,
    save_activation_dataset,
    print_dataset_summary
)
