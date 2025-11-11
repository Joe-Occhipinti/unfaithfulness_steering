"""
extract_activations.py

Extracts hidden state activations from specified layers of a Hugging Face model.
Performs a "clean" forward pass by stripping all annotation tags from the
text before tokenization, using the tags only as pointers.

Based on legacy code: legacy code/extract_activations.py
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


def get_clean_text_and_char_indices(annotated_text: str, target_tags: List[str]) -> Tuple[str, Dict[str, List[int]]]:
    """
    Strips all [TAG] style annotations and calculates the new character
    positions of target periods in the clean text. Robust to extra spaces.
    """
    char_indices = {tag: [] for tag in target_tags}

    # 1. Find all tags (e.g., [H], [/Q]) to calculate character offsets.
    all_tags_pattern = re.compile(r'\[/?[\w_]+\]')
    tag_matches = list(all_tags_pattern.finditer(annotated_text))

    def get_offset(annotated_char_pos):
        # Calculates the total length of all tags preceding a character position.
        offset = 0
        for match in tag_matches:
            if match.start() < annotated_char_pos:
                offset += len(match.group(0))
            else:
                break
        return offset

    # 2. Find target periods in the annotated text and map their positions to the clean text.
    for tag in target_tags:
        # This regex allows for optional spaces (\s*) and captures the period in group 2.
        # re.DOTALL allows . to match newlines (needed for multi-line tagged content)
        pattern = re.compile(fr'\[{tag}\](.*?)\s*(\.)\s*\[/{tag}\]', re.DOTALL)
        for match in pattern.finditer(annotated_text):
            # Get original position of the period itself.
            period_char_pos_annotated = match.start(2)
            # Calculate and subtract the offset of preceding tags.
            offset = get_offset(period_char_pos_annotated)
            period_char_pos_clean = period_char_pos_annotated - offset
            char_indices[tag].append(period_char_pos_clean)

    # 3. Create the clean text by removing all tags.
    clean_text = all_tags_pattern.sub('', annotated_text)

    return clean_text, char_indices


def extract_activations_from_annotated_prompts(
    jsonl_filename: str,
    prompt_field: str,
    output_dir: str,
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    target_tags: List[str] = None,
    layers_to_extract: List[int] = None,
    verbose: bool = True
) -> None:
    """
    Main function to extract activations from annotated prompts.

    Args:
        jsonl_filename: Path to JSONL file with annotated prompts
        prompt_field: Field in JSONL containing annotated text
        output_dir: Directory to save activation files
        model_id: Hugging Face model ID
        target_tags: Tags to extract activations for
        layers_to_extract: Layer indices to extract from
        verbose: Print debug information
    """
    # Default parameters from legacy code
    if target_tags is None:
        target_tags = ["F", "F_wk", "U", "E", "N", "H", "Q", "A", "Fact", "F_final", "U_final"]
    if layers_to_extract is None:
        layers_to_extract = list(range(32))  # Full sweep for DeepSeek

    # Setup
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"--- Configuration ---\nModel: {model_id}\nInput File: {jsonl_filename}\nPrompt Field: {prompt_field}\nOutput Directory: {output_dir}\nTarget Tags: {target_tags}\nLayers: All {len(layers_to_extract)}\n---------------------")

    # Load model and tokenizer using reusable function
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
    for i, data_item in enumerate(tqdm(prompts_data, desc="Extracting Activations")):
        annotated_text = data_item.get(prompt_field)
        if not annotated_text:
            continue

        # 1. Generate clean text and get target character indices.
        clean_text, period_char_indices = get_clean_text_and_char_indices(annotated_text, target_tags)

        if not any(period_char_indices.values()):
            continue

        # 2. Tokenize the CLEAN text (keep on CPU first to preserve encoding mappings).
        inputs_cpu = tokenizer(clean_text, return_tensors="pt")

        # Comprehensive debugging
        token_count = len(inputs_cpu['input_ids'][0])
        if verbose:
            print(f"\n=== DEBUGGING PROMPT {i} ===")
            print(f"Annotated text length: {len(annotated_text)}")
            print(f"Clean text length: {len(clean_text)}")
            print(f"Token count: {token_count}")
            print(f"Annotated text preview: '{annotated_text[:200]}...'")

            # Show the tokenized sequence
            tokens = tokenizer.convert_ids_to_tokens(inputs_cpu['input_ids'][0])
            print(f"First 10 tokens: {tokens[:10]}")
            print(f"Last 10 tokens: {tokens[-10:]}")

        # 3. Convert clean character indices to final token indices with detailed debugging.
        # IMPORTANT: Do this BEFORE moving to device to preserve encoding mappings
        period_token_indices = {tag: [] for tag in target_tags}
        for tag, char_indices_list in period_char_indices.items():
            if verbose:
                print(f"\nProcessing tag '{tag}' with {len(char_indices_list)} character indices: {char_indices_list}")
            for char_idx in char_indices_list:
                # Show what character we're looking at IN ANNOTATED TEXT
                # Need to find the corresponding position in annotated text
                # For now, show context in clean text AND try to find in annotated
                if char_idx < len(clean_text):
                    if verbose:
                        char_at_idx = clean_text[char_idx]
                        # Show context in CLEAN text (for char_to_token mapping)
                        clean_context_start = max(0, char_idx - 30)
                        clean_context_end = min(len(clean_text), char_idx + 31)
                        clean_context_before = clean_text[clean_context_start:char_idx]
                        clean_context_after = clean_text[char_idx + 1:clean_context_end]

                        # Try to find the same context in annotated text to show tags
                        # Search for the period with surrounding context in annotated text
                        search_pattern = f"{clean_context_before}.{clean_context_after[:10]}"
                        if search_pattern in annotated_text:
                            ann_idx = annotated_text.index(search_pattern) + len(clean_context_before)
                            ann_context_start = max(0, ann_idx - 30)
                            ann_context_end = min(len(annotated_text), ann_idx + 31)
                            ann_context = annotated_text[ann_context_start:ann_context_end]
                            print(f"  Char at CLEAN index {char_idx}: '{char_at_idx}'")
                            print(f"    ANNOTATED context: ...{ann_context}...")
                        else:
                            print(f"  Char at CLEAN index {char_idx}: '{char_at_idx}'")
                            print(f"    CLEAN context: ...{clean_context_before}[{char_at_idx}]{clean_context_after}...")
                else:
                    if verbose:
                        print(f"  ERROR: Char index {char_idx} is beyond clean text length {len(clean_text)}!")
                        print(f"  This indicates a problem in get_clean_text_and_char_indices()")
                    raise IndexError(f"Character index {char_idx} out of bounds for clean text of length {len(clean_text)}")

                # Use batch index 0 for single sequence, call BEFORE .to(device)
                token_idx = inputs_cpu.char_to_token(0, char_idx)
                if verbose:
                    print(f"  Token index for char {char_idx}: {token_idx}")

                if token_idx is not None:
                    if token_idx >= token_count:
                        if verbose:
                            print(f"  ERROR: Token index {token_idx} >= token count {token_count}!")
                            print(f"  This indicates a tokenizer char_to_token mapping issue")
                        raise IndexError(f"Token index {token_idx} out of bounds for token sequence of length {token_count}")
                    period_token_indices[tag].append(token_idx)
                else:
                    if verbose:
                        print(f"  WARNING: char_to_token returned None for char index {char_idx}")

        if verbose:
            print(f"\nFinal period token indices: {period_token_indices}")
            print("=== END DEBUGGING ===")

        # 4. Now move inputs to device for forward pass
        inputs = inputs_cpu.to(device)

        # 5. Run the forward pass on the CLEAN text.
        with torch.no_grad():
            model(**inputs)

        prompt_activations = {layer_idx: {} for layer_idx in layers_to_extract}

        # 6. Extract activations for target tokens from each layer.
        for layer_idx in layers_to_extract:
            wrapped_layer = model.model.layers[layer_idx]
            hidden_state_output = wrapped_layer.last_hidden_state

            if verbose:
                print(f"\nLayer {layer_idx} debug:")
                print(f"  Hidden state type: {type(hidden_state_output)}")
                print(f"  Hidden state shape: {hidden_state_output.shape if hasattr(hidden_state_output, 'shape') else 'No shape attr'}")

            # Based on your working script, the wrapper gives us a 2D tensor [seq_len, hidden_dim]
            full_activations_tensor = hidden_state_output

            for tag, indices in period_token_indices.items():
                if indices:
                    # Store each sentence activation individually instead of concatenating
                    individual_activations = []
                    for idx in indices:
                        # At this point, all indices should be valid due to our earlier checks
                        single_activation = full_activations_tensor[idx, :].cpu()
                        individual_activations.append(single_activation)
                    prompt_activations[layer_idx][tag] = individual_activations
            wrapped_layer.reset()

        # 7. Save the activations for the current prompt.
        save_path = os.path.join(output_dir, f"prompt_{i}_activations.pt")
        torch.save(prompt_activations, save_path)

        # Clean up memory after each prompt.
        gc.collect()
        torch.cuda.empty_cache()

    print("\n--- Activation extraction complete! ---")


def get_activation_statistics(output_dir: str) -> Dict[str, Any]:
    """
    Compute statistics about extracted activations from saved files.

    Args:
        output_dir: Directory containing activation .pt files

    Returns:
        Statistics dictionary
    """
    activation_files = [f for f in os.listdir(output_dir) if f.endswith('_activations.pt')]

    stats = {
        "total_prompts": len(activation_files),
        "tags_found": set(),
        "layers_found": set(),
        "activations_per_tag": {},
        "activations_per_layer": {}
    }

    for filename in activation_files:
        filepath = os.path.join(output_dir, filename)
        prompt_activations = torch.load(filepath, map_location='cpu')

        for layer_idx, layer_data in prompt_activations.items():
            stats["layers_found"].add(layer_idx)

            for tag, activations in layer_data.items():
                stats["tags_found"].add(tag)

                # Count activations per tag
                if tag not in stats["activations_per_tag"]:
                    stats["activations_per_tag"][tag] = 0
                stats["activations_per_tag"][tag] += len(activations)

                # Count activations per layer
                if layer_idx not in stats["activations_per_layer"]:
                    stats["activations_per_layer"][layer_idx] = 0
                stats["activations_per_layer"][layer_idx] += len(activations)

    stats["tags_found"] = sorted(list(stats["tags_found"]))
    stats["layers_found"] = sorted(list(stats["layers_found"]))

    return stats


def print_activation_statistics(stats: Dict[str, Any]) -> None:
    """Print formatted activation extraction statistics."""
    print(f"\n=== ACTIVATION EXTRACTION STATISTICS ===")
    print(f"Total Prompts: {stats['total_prompts']}")
    print(f"Tags Found: {stats['tags_found']}")
    print(f"Layers Extracted: {len(stats['layers_found'])} layers")

    print(f"\nActivations per Tag:")
    for tag, count in sorted(stats['activations_per_tag'].items()):
        print(f"  {tag}: {count}")

    print(f"\nTotal Activations per Layer:")
    layer_counts = stats['activations_per_layer']
    if layer_counts:
        print(f"  Min: {min(layer_counts.values())} (layer {min(layer_counts, key=layer_counts.get)})")
        print(f"  Max: {max(layer_counts.values())} (layer {max(layer_counts, key=layer_counts.get)})")
        print(f"  Average: {sum(layer_counts.values()) / len(layer_counts):.1f}")


def build_activation_dataset(
    activations_dir: str,
    source_jsonl: str = None,
    target_tags: List[str] = None,
    num_layers: int = 32,
    hidden_dim: int = 4096,
    metadata_fields: List[str] = None
) -> Dict[str, Any]:
    """
    Build aggregated activation dataset from individual prompt activation files.
    Aggregates all activations across prompts, organized by layer and tag.
    Preserves metadata from source JSONL file.

    Args:
        activations_dir: Directory containing prompt_*_activations.pt files
        source_jsonl: Path to original annotated JSONL file to extract metadata
        target_tags: Tags to include in dataset
        num_layers: Number of model layers
        hidden_dim: Hidden dimension size
        metadata_fields: List of fields to preserve from source JSONL (e.g., ["hint_template", "faithfulness_classification"])

    Returns:
        Dataset dictionary with structure: {data: {prompt_idx: {metadata: {...}, layers: {layer: {tag: tensor}}}}, info: {...}}
    """
    from .config import ActivationConfig

    if target_tags is None:
        target_tags = ActivationConfig.TARGET_TAGS

    if metadata_fields is None:
        metadata_fields = ["hint_template", "faithfulness_classification", "split", "subject", "correct_hint"]

    print(f"\n--- Building Activation Dataset ---")
    print(f"Source Directory: {activations_dir}")
    print(f"Source JSONL: {source_jsonl}")
    print(f"Target Tags: {target_tags}")
    print(f"Layers: 0-{num_layers-1}")
    print(f"Metadata Fields: {metadata_fields}")

    # Load activation files
    activation_files = []
    if os.path.exists(activations_dir):
        for filename in os.listdir(activations_dir):
            if filename.startswith("prompt_") and filename.endswith("_activations.pt"):
                activation_files.append(os.path.join(activations_dir, filename))
        activation_files.sort()  # Consistent ordering
    else:
        raise FileNotFoundError(f"Activations directory {activations_dir} does not exist!")

    print(f"Found {len(activation_files)} activation files to process.")

    # Load metadata from source JSONL if provided
    metadata_by_index = {}
    if source_jsonl and os.path.exists(source_jsonl):
        print(f"\n--- Loading metadata from {source_jsonl} ---")
        with open(source_jsonl, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    record = json.loads(line)
                    # Extract standard fields (all except correct_hint which needs to be computed)
                    metadata = {
                        field: record.get(field)
                        for field in metadata_fields
                        if field != "correct_hint"
                    }

                    # Compute correct_hint from hint_letter and ground_truth_letter
                    hint_letter = record.get("hint_letter")
                    ground_truth_letter = record.get("ground_truth_letter")

                    if hint_letter is not None and ground_truth_letter is not None:
                        metadata["correct_hint"] = "True" if hint_letter == ground_truth_letter else "False"
                    else:
                        metadata["correct_hint"] = None  # Missing data

                    metadata_by_index[idx] = metadata
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {idx} in {source_jsonl}")
        print(f"Loaded metadata for {len(metadata_by_index)} prompts")
    elif source_jsonl:
        print(f"Warning: Source JSONL {source_jsonl} not found, proceeding without metadata")

    # Structure: prompt_wise_data[prompt_idx] = {metadata: {...}, layers: {layer: {tag: tensor}}}
    prompt_wise_data = {}

    # Statistics tracking
    total_stats = {tag: 0 for tag in target_tags}
    layer_stats = {layer: {tag: 0 for tag in target_tags} for layer in range(num_layers)}
    total_files_processed = 0
    files_with_data = 0

    # Process each activation file
    for file_path in tqdm(activation_files, desc="Processing activation files"):
        try:
            # Extract prompt index from filename (e.g., "prompt_5_activations.pt" -> 5)
            filename = os.path.basename(file_path)
            prompt_idx = int(filename.split('_')[1])

            # Load the activation data for this prompt
            prompt_data = torch.load(file_path, map_location='cpu')
            has_data_for_prompt = False

            # Initialize prompt entry with metadata and layers structure
            prompt_wise_data[prompt_idx] = {
                "metadata": metadata_by_index.get(prompt_idx, {}),
                "layers": {}
            }

            # Extract data for each layer
            for layer_idx in range(num_layers):
                prompt_wise_data[prompt_idx]["layers"][layer_idx] = {}

                if layer_idx in prompt_data:
                    layer_data = prompt_data[layer_idx]

                    # Extract activations for each target tag
                    for tag in target_tags:
                        if tag in layer_data and layer_data[tag]:
                            # layer_data[tag] is a list of individual activation tensors
                            # Stack them into a single tensor for this prompt/layer/tag
                            tag_activations = torch.stack(layer_data[tag], dim=0)
                            prompt_wise_data[prompt_idx]["layers"][layer_idx][tag] = tag_activations

                            # Update statistics
                            num_activations = tag_activations.shape[0]
                            total_stats[tag] += num_activations
                            layer_stats[layer_idx][tag] += num_activations
                            has_data_for_prompt = True
                        else:
                            # No data for this tag - create empty tensor
                            prompt_wise_data[prompt_idx]["layers"][layer_idx][tag] = torch.empty(0, hidden_dim)
                else:
                    # No data for this layer - create empty tensors for all tags
                    for tag in target_tags:
                        prompt_wise_data[prompt_idx]["layers"][layer_idx][tag] = torch.empty(0, hidden_dim)

            if has_data_for_prompt:
                files_with_data += 1

            total_files_processed += 1

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")

    print(f"\n--- Processing Complete ---")
    print(f"Files processed: {total_files_processed}")
    print(f"Files with relevant data: {files_with_data}")

    print(f"\nActivation counts per tag:")
    for tag, count in total_stats.items():
        print(f"  {tag}: {count} activations")

    print(f"\n--- Final Dataset Structure ---")
    print(f"Prompt-wise organization: {len(prompt_wise_data)} prompts")

    # Build final dataset structure with prompt-wise organization
    dataset_info = {
        "tags": target_tags,
        "num_layers": num_layers,
        "num_prompts": len(prompt_wise_data),
        "hidden_dim": hidden_dim,
        "total_files": total_files_processed,
        "files_with_data": files_with_data,
        "total_stats": total_stats,
        "layer_stats": layer_stats,
        "metadata_fields": metadata_fields,
        "source_jsonl": source_jsonl
    }

    final_dataset_structure = {
        "data": prompt_wise_data,
        "info": dataset_info
    }

    return final_dataset_structure


def save_activation_dataset(dataset: Dict[str, Any], output_file: str) -> None:
    """
    Save activation dataset to pickle file.

    Args:
        dataset: Dataset dictionary from build_activation_dataset
        output_file: Output pickle file path
    """
    import pickle

    print(f"\n--- Saving activation dataset to {output_file}... ---")

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Activation dataset saved successfully!")


def print_dataset_summary(dataset: Dict[str, Any]) -> None:
    """Print formatted dataset summary statistics."""
    info = dataset["info"]
    data = dataset["data"]

    print(f"\n=== ACTIVATION DATASET SUMMARY ===")
    print(f"Total layers: {info['num_layers']}")
    print(f"Hidden dimension: {info['hidden_dim']}")
    print(f"Tags included: {info['tags']}")
    print(f"Files processed: {info['total_files']}")
    print(f"Files with data: {info['files_with_data']}")

    # Display metadata information
    if 'metadata_fields' in info and info['metadata_fields']:
        print(f"\nMetadata fields: {info['metadata_fields']}")

        # Count distribution of metadata values
        if data:
            metadata_distributions = {field: {} for field in info['metadata_fields']}
            for prompt_idx in data.keys():
                metadata = data[prompt_idx].get("metadata", {})
                for field in info['metadata_fields']:
                    value = metadata.get(field, "missing")
                    metadata_distributions[field][value] = metadata_distributions[field].get(value, 0) + 1

            print(f"\nMetadata distributions:")
            for field, distribution in metadata_distributions.items():
                print(f"  {field}:")
                for value, count in distribution.items():
                    print(f"    {value}: {count} prompts")

    print(f"\nTotal activations by tag:")
    for tag, count in info['total_stats'].items():
        print(f"  {tag}: {count}")

    print(f"\nSample counts by layer (first 5 layers):")
    for layer_idx in range(min(5, info['num_layers'])):
        print(f"  Layer {layer_idx}:")
        for tag in info['tags']:
            count = info['layer_stats'][layer_idx][tag]
            # For prompt-wise data with nested structure, we need to check actual data from prompts
            if count > 0:
                # Find a prompt that has data for this layer/tag combination
                sample_shape = None
                for prompt_idx in data.keys():
                    layers = data[prompt_idx].get("layers", {})
                    if layer_idx in layers and tag in layers[layer_idx]:
                        if layers[layer_idx][tag].numel() > 0:  # Non-empty tensor
                            sample_shape = layers[layer_idx][tag].shape
                            break
                if sample_shape is not None:
                    print(f"    {tag}: {count} activations, sample tensor shape: {sample_shape}")
                else:
                    print(f"    {tag}: {count} activations (shape unavailable)")
            else:
                print(f"    {tag}: 0 activations")

    print(f"\nDataset structure:")
    print(f"  dataset['data'][prompt_idx]['metadata'] = {{field: value, ...}}")
    print(f"  dataset['data'][prompt_idx]['layers'][layer_idx][tag] = tensor([num_activations, {info['hidden_dim']}])")
    print(f"\n--- Dataset building complete! ---")