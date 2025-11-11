"""
separability.py

Core logic for analyzing separability between positive and negative activations.
Provides three separate investigations: cosine similarity, norm distributions, and linear separability.
"""

import torch
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from tqdm import tqdm


def load_activation_dataset(file_path: str) -> Dict[str, Any]:
    """
    Load activation dataset from pickle file.

    Args:
        file_path: Path to the dataset pickle file

    Returns:
        Dataset dictionary with structure: {'data': {layer: {tag: tensor}}, 'info': {...}}
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def split_dataset_by_prompts(
    dataset: Dict[str, Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.3,
    random_seed: int = 42,
    balance_val_split: bool = False,
    positive_tags: List[str] = None,
    negative_tags: List[str] = None,
    stratify_by_metadata: bool = True,
    use_existing_split: bool = True
) -> Dict[str, Dict]:
    """
    Split dataset by prompt indices to ensure activations from same prompt stay together.

    Args:
        dataset: Activation dataset with prompt-wise structure
        train_ratio: Proportion for training set (ignored if use_existing_split finds splits)
        val_ratio: Proportion for validation set (ignored if use_existing_split finds splits)
        random_seed: Random seed for reproducibility
        balance_val_split: If True, balance positive and negative samples in val split
        positive_tags: Tags for positive class (required if balance_val_split=True)
        negative_tags: Tags for negative class (required if balance_val_split=True)
        stratify_by_metadata: If True, use stratified splitting by (faithfulness_classification, hint_template)
        use_existing_split: If True, use existing "split" field from metadata if available (default: True)

    Returns:
        Dictionary with train/val splits: {'train': dataset_subset, 'val': dataset_subset}
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    data = dataset['data']
    info = dataset['info']

    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Check if existing split field should be used
    train_prompts = None
    val_prompts = None

    if use_existing_split and 'split' in info.get('metadata_fields', []):
        # Check if prompts actually have split values
        prompts_with_split = []
        for prompt_idx in data.keys():
            split_val = data[prompt_idx].get('metadata', {}).get('split')
            if split_val in ['train', 'val']:
                prompts_with_split.append(prompt_idx)

        if prompts_with_split:
            print(f"Found 'split' field in metadata - using existing splits")
            train_prompts = []
            val_prompts = []

            for prompt_idx in data.keys():
                split_val = data[prompt_idx].get('metadata', {}).get('split')
                if split_val == 'train':
                    train_prompts.append(prompt_idx)
                elif split_val == 'val':
                    val_prompts.append(prompt_idx)
                elif split_val is None or split_val == '':
                    # No split assigned - default to train
                    train_prompts.append(prompt_idx)
                else:
                    # Unknown split value (e.g., 'test') - skip or warn
                    print(f"  Warning: Prompt {prompt_idx} has split='{split_val}', assigning to train")
                    train_prompts.append(prompt_idx)

            print(f"Loaded existing splits: {len(train_prompts)} train, {len(val_prompts)} val prompts")

    # Fallback to stratified or random splitting if no existing splits
    if train_prompts is None:
        if stratify_by_metadata and 'metadata_fields' in info and info['metadata_fields']:
            # STRATIFIED SPLITTING by (faithfulness_classification, hint_template)
            print("Using stratified splitting by metadata (faithfulness × hint_template)")

            # Group prompts by (faithfulness_classification, hint_template)
            groups = {}
            for prompt_idx in data.keys():
                prompt_data = data[prompt_idx]
                metadata = prompt_data.get("metadata", {})

                faithfulness = metadata.get("faithfulness_classification", "unknown")
                hint_template = metadata.get("hint_template", "unknown")

                group_key = (faithfulness, hint_template)
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(prompt_idx)

            # Print group statistics
            print(f"\nFound {len(groups)} groups:")
            for (faith, template), indices in sorted(groups.items()):
                print(f"  ({faith}, {template}): {len(indices)} prompts")

            # Split each group separately using the same ratios
            train_prompts = []
            val_prompts = []

            for group_key, group_indices in groups.items():
                # Shuffle this group
                np.random.shuffle(group_indices)

                # Split this group
                n_group = len(group_indices)
                train_end = int(train_ratio * n_group)

                # Warn if group is too small for proper splitting
                if n_group < 3:
                    print(f"  Warning: Group {group_key} has only {n_group} samples - may lead to imbalanced splits")

                group_train = group_indices[:train_end]
                group_val = group_indices[train_end:]

                train_prompts.extend(group_train)
                val_prompts.extend(group_val)

            # Shuffle the final splits to mix the groups
            np.random.shuffle(train_prompts)
            np.random.shuffle(val_prompts)

            print(f"\nStratified split result: {len(train_prompts)} train, {len(val_prompts)} val prompts")

        else:
            # RANDOM SPLITTING (original behavior)
            if stratify_by_metadata:
                print("Warning: stratify_by_metadata=True but no metadata available, using random splitting")

            # Get actual prompt indices from the dataset
            prompt_indices = list(data.keys())
            total_prompts = len(prompt_indices)

            # Shuffle prompt indices
            np.random.shuffle(prompt_indices)

            # Split indices
            train_end = int(train_ratio * total_prompts)

            train_prompts = prompt_indices[:train_end]
            val_prompts = prompt_indices[train_end:]

            print(f"Random split: {len(train_prompts)} train, {len(val_prompts)} val prompts")

    # Verify stratification worked correctly
    if stratify_by_metadata and 'metadata_fields' in info and info['metadata_fields']:
        print("\nVerifying stratification:")
        for split_name, split_indices in [('train', train_prompts), ('val', val_prompts)]:
            split_groups = {}
            for prompt_idx in split_indices:
                prompt_data = data[prompt_idx]
                metadata = prompt_data.get("metadata", {})
                faithfulness = metadata.get("faithfulness_classification", "unknown")
                hint_template = metadata.get("hint_template", "unknown")
                group_key = (faithfulness, hint_template)
                split_groups[group_key] = split_groups.get(group_key, 0) + 1

            print(f"\n  {split_name.upper()} split distribution:")
            for (faith, template), count in sorted(split_groups.items()):
                print(f"    ({faith}, {template}): {count} prompts")

    # Calculate balance info for both splits if requested
    train_balance_info = {'balance_enabled': False}
    val_balance_info = {'balance_enabled': False}

    if balance_val_split:
        if positive_tags is None or negative_tags is None:
            raise ValueError("positive_tags and negative_tags must be provided when balance_val_split=True")

        # Calculate balance info for TRAIN split
        train_pos_counts = []
        train_neg_counts = []
        for prompt_idx in train_prompts:
            prompt_data = data[prompt_idx]
            # Handle both old structure (direct layer access) and new structure (nested "layers")
            layers_data = prompt_data.get("layers", prompt_data)
            pos_count = sum(layers_data[0].get(tag, torch.empty(0)).shape[0]
                          for tag in positive_tags if tag in layers_data[0])
            neg_count = sum(layers_data[0].get(tag, torch.empty(0)).shape[0]
                          for tag in negative_tags if tag in layers_data[0])
            train_pos_counts.append(pos_count)
            train_neg_counts.append(neg_count)

        train_total_pos = sum(train_pos_counts)
        train_total_neg = sum(train_neg_counts)
        train_min_samples = min(train_total_pos, train_total_neg)

        print(f"Train split before balancing: {train_total_pos} positive, {train_total_neg} negative samples")
        print(f"Balancing to {train_min_samples} samples per class")

        train_balance_info = {
            'balance_enabled': True,
            'target_samples_per_class': train_min_samples,
            'original_pos': train_total_pos,
            'original_neg': train_total_neg
        }

        # Calculate balance info for VAL split
        val_pos_counts = []
        val_neg_counts = []
        for prompt_idx in val_prompts:
            prompt_data = data[prompt_idx]
            # Handle both old structure (direct layer access) and new structure (nested "layers")
            layers_data = prompt_data.get("layers", prompt_data)
            pos_count = sum(layers_data[0].get(tag, torch.empty(0)).shape[0]
                          for tag in positive_tags if tag in layers_data[0])
            neg_count = sum(layers_data[0].get(tag, torch.empty(0)).shape[0]
                          for tag in negative_tags if tag in layers_data[0])
            val_pos_counts.append(pos_count)
            val_neg_counts.append(neg_count)

        val_total_pos = sum(val_pos_counts)
        val_total_neg = sum(val_neg_counts)
        val_min_samples = min(val_total_pos, val_total_neg)

        print(f"Val split before balancing: {val_total_pos} positive, {val_total_neg} negative samples")
        print(f"Balancing to {val_min_samples} samples per class")

        val_balance_info = {
            'balance_enabled': True,
            'target_samples_per_class': val_min_samples,
            'original_pos': val_total_pos,
            'original_neg': val_total_neg
        }

    print(f"Split dataset: {len(train_prompts)} train, {len(val_prompts)} val prompts")

    # Create splits by selecting specific prompts
    splits = {}
    for split_name, prompt_list in [('train', train_prompts), ('val', val_prompts)]:
        split_data = {}

        # Include only the prompts assigned to this split
        for prompt_idx in prompt_list:
            split_data[prompt_idx] = data[prompt_idx]

        split_info = {
            **info,
            'split': split_name,
            'num_prompts': len(prompt_list),
            'prompt_indices': prompt_list
        }

        # Add balance info to each split
        if split_name == 'train' and balance_val_split:
            split_info['balance_info'] = train_balance_info
        elif split_name == 'val' and balance_val_split:
            split_info['balance_info'] = val_balance_info

        splits[split_name] = {
            'data': split_data,
            'info': split_info
        }

    return splits


def extract_tag_activations(
    dataset: Dict[str, Any],
    positive_tags: List[str],
    negative_tags: List[str],
    split: str = None,
    correct_hint_filter: Optional[str] = None
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Extract and combine activations for positive and negative tags by layer.

    Args:
        dataset: Activation dataset or split with prompt-wise structure
        positive_tags: List of tags to treat as positive class
        negative_tags: List of tags to treat as negative class
        split: Split name if dataset contains multiple splits
        correct_hint_filter: Filter prompts by correct_hint value (None: use all, "True": only correct, "False": only incorrect)

    Returns:
        Tuple of (positive_activations_by_layer, negative_activations_by_layer)
        where each is {layer_idx: combined_tensor}
    """
    if split and split in dataset:
        data = dataset[split]['data']
        info = dataset[split]['info']
    else:
        data = dataset['data']
        info = dataset['info']

    num_layers = info['num_layers']
    hidden_dim = info['hidden_dim']

    positive_by_layer = {}
    negative_by_layer = {}

    # Initialize layer dictionaries
    for layer_idx in range(num_layers):
        positive_by_layer[layer_idx] = []
        negative_by_layer[layer_idx] = []

    # Process each prompt in the dataset/split
    for prompt_idx in data:
        prompt_data = data[prompt_idx]

        # Apply correct_hint filter if specified
        if correct_hint_filter is not None:
            metadata = prompt_data.get('metadata', {})
            prompt_correct_hint = metadata.get('correct_hint')
            if prompt_correct_hint != correct_hint_filter:
                continue  # Skip this prompt

        # Handle both old structure (direct layer access) and new structure (nested "layers")
        layers_data = prompt_data.get("layers", prompt_data)

        for layer_idx in range(num_layers):
            if layer_idx in layers_data:
                layer_data = layers_data[layer_idx]

                # Collect positive tag activations
                for tag in positive_tags:
                    if tag in layer_data and layer_data[tag].numel() > 0:
                        positive_by_layer[layer_idx].append(layer_data[tag])

                # Collect negative tag activations
                for tag in negative_tags:
                    if tag in layer_data and layer_data[tag].numel() > 0:
                        negative_by_layer[layer_idx].append(layer_data[tag])

    # Concatenate activations for each layer
    final_positive = {}
    final_negative = {}

    for layer_idx in range(num_layers):
        # Combine positive activations
        if positive_by_layer[layer_idx]:
            final_positive[layer_idx] = torch.cat(positive_by_layer[layer_idx], dim=0)
        else:
            final_positive[layer_idx] = torch.empty(0, hidden_dim)

        # Combine negative tag activations
        if negative_by_layer[layer_idx]:
            final_negative[layer_idx] = torch.cat(negative_by_layer[layer_idx], dim=0)
        else:
            final_negative[layer_idx] = torch.empty(0, hidden_dim)

    return final_positive, final_negative


def extract_tag_activations_with_metadata(
    dataset: Dict[str, Any],
    positive_tags: List[str],
    negative_tags: List[str],
    split: str = None
) -> Tuple[Dict[int, Dict[str, List]], Dict[int, Dict[str, List]]]:
    """
    Extract activations with template metadata preserved for balanced downsampling.

    Args:
        dataset: Activation dataset or split with prompt-wise structure
        positive_tags: List of tags to treat as positive class
        negative_tags: List of tags to treat as negative class
        split: Split name if dataset contains multiple splits

    Returns:
        Tuple of (positive_by_template, negative_by_template) where each is:
        {layer_idx: {template_name: [list of activation tensors]}}
    """
    if split and split in dataset:
        data = dataset[split]['data']
        info = dataset[split]['info']
    else:
        data = dataset['data']
        info = dataset['info']

    num_layers = info['num_layers']

    # Structure: {layer_idx: {template: [activations]}}
    positive_by_template = {layer_idx: {} for layer_idx in range(num_layers)}
    negative_by_template = {layer_idx: {} for layer_idx in range(num_layers)}

    # Process each prompt in the dataset/split
    for prompt_idx in data:
        prompt_data = data[prompt_idx]

        # Get metadata
        metadata = prompt_data.get("metadata", {})
        hint_template = metadata.get("hint_template", "unknown")

        # Handle both old structure (direct layer access) and new structure (nested "layers")
        layers_data = prompt_data.get("layers", prompt_data)

        for layer_idx in range(num_layers):
            if layer_idx in layers_data:
                layer_data = layers_data[layer_idx]

                # Initialize template dict if needed
                if hint_template not in positive_by_template[layer_idx]:
                    positive_by_template[layer_idx][hint_template] = []
                if hint_template not in negative_by_template[layer_idx]:
                    negative_by_template[layer_idx][hint_template] = []

                # Collect positive tag activations
                for tag in positive_tags:
                    if tag in layer_data and layer_data[tag].numel() > 0:
                        positive_by_template[layer_idx][hint_template].append(layer_data[tag])

                # Collect negative tag activations
                for tag in negative_tags:
                    if tag in layer_data and layer_data[tag].numel() > 0:
                        negative_by_template[layer_idx][hint_template].append(layer_data[tag])

    # Concatenate activations within each template
    for layer_idx in range(num_layers):
        for template in positive_by_template[layer_idx]:
            if positive_by_template[layer_idx][template]:
                positive_by_template[layer_idx][template] = torch.cat(
                    positive_by_template[layer_idx][template], dim=0
                )
            else:
                positive_by_template[layer_idx][template] = torch.empty(0, info['hidden_dim'])

        for template in negative_by_template[layer_idx]:
            if negative_by_template[layer_idx][template]:
                negative_by_template[layer_idx][template] = torch.cat(
                    negative_by_template[layer_idx][template], dim=0
                )
            else:
                negative_by_template[layer_idx][template] = torch.empty(0, info['hidden_dim'])

    return positive_by_template, negative_by_template


def compute_cosine_similarity_by_layer(
    dataset: Dict[str, Any],
    positive_tags: List[str],
    negative_tags: List[str]
) -> Dict[int, float]:
    """
    Compute cosine similarity between positive and negative means for each layer.
    Uses full dataset for better statistical estimates.

    Args:
        dataset: Full activation dataset
        positive_tags: Tags to treat as positive class
        negative_tags: Tags to treat as negative class

    Returns:
        Dictionary: {layer_idx: cosine_similarity_value}
    """
    positive_by_layer, negative_by_layer = extract_tag_activations(dataset, positive_tags, negative_tags)

    cosine_similarities = {}

    for layer_idx in positive_by_layer:
        positive_acts = positive_by_layer[layer_idx]
        negative_acts = negative_by_layer[layer_idx]

        if positive_acts.numel() == 0 or negative_acts.numel() == 0:
            cosine_similarities[layer_idx] = 0.0
            continue

        # Compute means
        positive_mean = positive_acts.mean(dim=0)
        negative_mean = negative_acts.mean(dim=0)

        # Compute cosine similarity between means
        cosine_sim = torch.cosine_similarity(positive_mean, negative_mean, dim=0).item()
        cosine_similarities[layer_idx] = cosine_sim

    return cosine_similarities


def compute_mean_differences_by_layer(
    dataset: Dict[str, Any],
    positive_tags: List[str],
    negative_tags: List[str]
) -> Dict[int, Dict[str, Any]]:
    """
    Compute distance between positive and negative means for each layer.
    Uses full dataset for better statistical estimates.

    Args:
        dataset: Full activation dataset
        positive_tags: Tags to treat as positive class
        negative_tags: Tags to treat as negative class

    Returns:
        Dictionary: {layer_idx: {
            'mean_diff_norm': float  # ||mean_positive - mean_negative||
        }}
    """
    positive_by_layer, negative_by_layer = extract_tag_activations(dataset, positive_tags, negative_tags)

    mean_differences = {}

    for layer_idx in positive_by_layer:
        positive_acts = positive_by_layer[layer_idx]
        negative_acts = negative_by_layer[layer_idx]

        if positive_acts.numel() == 0 or negative_acts.numel() == 0:
            mean_differences[layer_idx] = {
                'mean_diff_norm': 0.0
            }
            continue

        # Compute means and distance between them
        positive_mean = positive_acts.mean(dim=0)
        negative_mean = negative_acts.mean(dim=0)
        mean_diff_norm = torch.norm(positive_mean - negative_mean).item()

        mean_differences[layer_idx] = {
            'mean_diff_norm': mean_diff_norm
        }

    return mean_differences


def extract_prompts_by_class_and_template(
    dataset: Dict[str, Any],
    positive_tags: List[str],
    negative_tags: List[str],
    split: str = None
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Extract prompt indices grouped by template for positive and negative classes.

    Args:
        dataset: Activation dataset or split with prompt-wise structure
        positive_tags: Tags to treat as positive class
        negative_tags: Tags to treat as negative class
        split: Split name if dataset contains multiple splits

    Returns:
        Tuple of (pos_prompts_by_template, neg_prompts_by_template) where each is:
        {template_name: [list of prompt indices]}
    """
    if split and split in dataset:
        data = dataset[split]['data']
    else:
        data = dataset['data']

    pos_prompts_by_template = {}
    neg_prompts_by_template = {}

    # Iterate through all prompts
    for prompt_idx in data:
        prompt_data = data[prompt_idx]
        metadata = prompt_data.get("metadata", {})
        hint_template = metadata.get("hint_template", "unknown")

        # Handle both old and new structure
        layers_data = prompt_data.get("layers", prompt_data)

        # Check if this prompt has positive or negative tags (using layer 0 as representative)
        if 0 in layers_data:
            layer_0 = layers_data[0]

            # Check for positive tags
            has_positive = any(tag in layer_0 and layer_0[tag].numel() > 0
                             for tag in positive_tags)

            # Check for negative tags
            has_negative = any(tag in layer_0 and layer_0[tag].numel() > 0
                             for tag in negative_tags)

            if has_positive:
                if hint_template not in pos_prompts_by_template:
                    pos_prompts_by_template[hint_template] = []
                pos_prompts_by_template[hint_template].append(prompt_idx)

            if has_negative:
                if hint_template not in neg_prompts_by_template:
                    neg_prompts_by_template[hint_template] = []
                neg_prompts_by_template[hint_template].append(prompt_idx)

    return pos_prompts_by_template, neg_prompts_by_template


def balance_prompts_with_templates(
    pos_prompts_by_template: Dict[str, List[int]],
    neg_prompts_by_template: Dict[str, List[int]],
    random_seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Balance prompt counts between classes while preserving template distribution from minority class.

    Args:
        pos_prompts_by_template: {template_name: [prompt_indices]} for positive class
        neg_prompts_by_template: {template_name: [prompt_indices]} for negative class
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (selected_pos_prompts, selected_neg_prompts) as lists of prompt indices
    """
    # Calculate total prompt counts
    pos_total = sum(len(prompts) for prompts in pos_prompts_by_template.values())
    neg_total = sum(len(prompts) for prompts in neg_prompts_by_template.values())

    if pos_total == 0 or neg_total == 0:
        return [], []

    # Determine minority and majority classes
    if pos_total <= neg_total:
        minority_by_template = pos_prompts_by_template
        majority_by_template = neg_prompts_by_template
        minority_total = pos_total
        swap = False
    else:
        minority_by_template = neg_prompts_by_template
        majority_by_template = pos_prompts_by_template
        minority_total = neg_total
        swap = True

    # Calculate template proportions from minority class
    template_proportions = {}
    for template, prompts in minority_by_template.items():
        template_proportions[template] = len(prompts) / minority_total if minority_total > 0 else 0

    # Downsample majority class prompts to match minority template distribution
    np.random.seed(random_seed)
    majority_selected = []

    for template, target_proportion in template_proportions.items():
        target_count = int(np.round(minority_total * target_proportion))

        if template in majority_by_template:
            available_prompts = majority_by_template[template]
            current_count = len(available_prompts)

            if current_count >= target_count:
                # Randomly select target_count prompts
                selected = np.random.choice(available_prompts, size=target_count, replace=False).tolist()
                majority_selected.extend(selected)
            else:
                # Not enough prompts, take all
                majority_selected.extend(available_prompts)

    # Combine all minority prompts
    minority_selected = []
    for prompts in minority_by_template.values():
        minority_selected.extend(prompts)

    # Return in correct order
    if swap:
        return majority_selected, minority_selected  # pos, neg
    else:
        return minority_selected, majority_selected  # pos, neg


def balance_activations_with_templates(
    pos_by_template: Dict[str, torch.Tensor],
    neg_by_template: Dict[str, torch.Tensor],
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Balance activation counts between classes while preserving template distribution from minority class.

    Args:
        pos_by_template: {template_name: activations_tensor} for positive class
        neg_by_template: {template_name: activations_tensor} for negative class
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (balanced_pos_activations, balanced_neg_activations)
    """
    # Calculate total activation counts
    pos_total = sum(acts.shape[0] for acts in pos_by_template.values() if acts.numel() > 0)
    neg_total = sum(acts.shape[0] for acts in neg_by_template.values() if acts.numel() > 0)

    if pos_total == 0 or neg_total == 0:
        # Return empty tensors if either class is empty
        hidden_dim = 4096  # Default
        if pos_by_template:
            for acts in pos_by_template.values():
                if acts.numel() > 0:
                    hidden_dim = acts.shape[1]
                    break
        elif neg_by_template:
            for acts in neg_by_template.values():
                if acts.numel() > 0:
                    hidden_dim = acts.shape[1]
                    break
        return torch.empty(0, hidden_dim), torch.empty(0, hidden_dim)

    # Determine minority and majority classes
    if pos_total <= neg_total:
        minority_by_template = pos_by_template
        majority_by_template = neg_by_template
        minority_total = pos_total
        swap = False
    else:
        minority_by_template = neg_by_template
        majority_by_template = pos_by_template
        minority_total = neg_total
        swap = True

    # Calculate template proportions from minority class
    template_proportions = {}
    for template, acts in minority_by_template.items():
        if acts.numel() > 0:
            template_proportions[template] = acts.shape[0] / minority_total

    # Downsample majority class activations to match minority template distribution
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    majority_downsampled = []

    for template, target_proportion in template_proportions.items():
        target_count = int(np.round(minority_total * target_proportion))

        if template in majority_by_template and majority_by_template[template].numel() > 0:
            acts = majority_by_template[template]
            current_count = acts.shape[0]

            if current_count >= target_count:
                # Randomly select target_count activations
                indices = torch.randperm(current_count, generator=torch.Generator().manual_seed(random_seed))[:target_count]
                majority_downsampled.append(acts[indices])
            else:
                # Not enough activations in this template, take all
                majority_downsampled.append(acts)

    # Combine all templates for majority class
    if majority_downsampled:
        majority_balanced = torch.cat(majority_downsampled, dim=0)
    else:
        # Find hidden_dim from any available tensor
        hidden_dim = 4096  # Default
        for template_dict in [minority_by_template, majority_by_template]:
            for acts in template_dict.values():
                if acts.numel() > 0:
                    hidden_dim = acts.shape[1]
                    break
            if hidden_dim != 4096:
                break
        majority_balanced = torch.empty(0, hidden_dim)

    # Combine all templates for minority class (keep all)
    minority_combined_list = [acts for acts in minority_by_template.values() if acts.numel() > 0]
    if minority_combined_list:
        minority_combined = torch.cat(minority_combined_list, dim=0)
    else:
        # Find hidden_dim from any available tensor
        hidden_dim = 4096  # Default
        for template_dict in [minority_by_template, majority_by_template]:
            for acts in template_dict.values():
                if acts.numel() > 0:
                    hidden_dim = acts.shape[1]
                    break
            if hidden_dim != 4096:
                break
        minority_combined = torch.empty(0, hidden_dim)

    # Return in correct order
    if swap:
        return majority_balanced, minority_combined  # pos, neg
    else:
        return minority_combined, majority_balanced  # pos, neg


def train_linear_probes_by_layer(
    dataset_splits: Dict[str, Dict],
    positive_tags: List[str],
    negative_tags: List[str],
    random_seed: int = 42,
    balance_templates: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Train linear probes for each layer using train/val splits with template-aware balancing.

    Args:
        dataset_splits: Dictionary with train/val splits
        positive_tags: Tags to treat as positive class (label 0)
        negative_tags: Tags to treat as negative class (label 1)
        random_seed: Random seed for reproducibility
        balance_templates: If True, balance hint templates when downsampling classes

    Returns:
        Dictionary: {layer_idx: {
            'train_acc': float,
            'val_acc': float,
            'classifier': LogisticRegression model,
            'train_samples': int,
            'val_samples': int
        }}
    """
    # Check if we should use template-aware balancing
    use_template_balancing = (balance_templates and
                              'metadata_fields' in dataset_splits['train']['info'] and
                              dataset_splits['train']['info']['metadata_fields'])

    if use_template_balancing:
        print("Using template-aware activation-level balanced downsampling")

        # Extract activations with template metadata preserved
        train_pos_by_template, train_neg_by_template = extract_tag_activations_with_metadata(
            dataset_splits, positive_tags, negative_tags, 'train'
        )
        val_pos_by_template, val_neg_by_template = extract_tag_activations_with_metadata(
            dataset_splits, positive_tags, negative_tags, 'val'
        )

    else:
        print("Using standard extraction (no template balancing)")
        # Extract all activations
        train_pos, train_neg = extract_tag_activations(dataset_splits, positive_tags, negative_tags, 'train')
        val_pos, val_neg = extract_tag_activations(dataset_splits, positive_tags, negative_tags, 'val')

    probe_results = {}
    # Determine number of layers from the appropriate data structure
    if use_template_balancing:
        num_layers = len(train_pos_by_template)
        layer_indices = sorted(train_pos_by_template.keys())
    else:
        num_layers = len(train_pos)
        layer_indices = sorted(train_pos.keys())

    print(f"\nTraining linear probes for {num_layers} layers...")

    for layer_idx in tqdm(layer_indices, desc="Training probes"):
        # Get activations for this layer
        if use_template_balancing:
            # Apply activation-level template-aware balancing
            train_pos_acts, train_neg_acts = balance_activations_with_templates(
                train_pos_by_template[layer_idx],
                train_neg_by_template[layer_idx],
                random_seed
            )
            val_pos_acts, val_neg_acts = balance_activations_with_templates(
                val_pos_by_template[layer_idx],
                val_neg_by_template[layer_idx],
                random_seed
            )
        else:
            # Use old behavior (no balancing)
            train_pos_acts = train_pos[layer_idx]
            train_neg_acts = train_neg[layer_idx]
            val_pos_acts = val_pos[layer_idx]
            val_neg_acts = val_neg[layer_idx]

        # Check if we have sufficient data
        if (train_pos_acts.numel() == 0 or train_neg_acts.numel() == 0 or
            val_pos_acts.numel() == 0 or val_neg_acts.numel() == 0):

            probe_results[layer_idx] = {
                'train_acc': 0.0,
                'val_acc': 0.0,
                'classifier': None,
                'train_samples': 0,
                'val_samples': 0,
                'error': 'Insufficient data'
            }
            continue

        # Prepare training data
        X_train = torch.cat([train_pos_acts, train_neg_acts], dim=0).float().numpy()
        y_train = torch.cat([
            torch.zeros(train_pos_acts.shape[0]),  # 0 for positive
            torch.ones(train_neg_acts.shape[0])    # 1 for negative
        ]).numpy()

        # Prepare validation data
        X_val = torch.cat([val_pos_acts, val_neg_acts], dim=0).float().numpy()
        y_val = torch.cat([
            torch.zeros(val_pos_acts.shape[0]),
            torch.ones(val_neg_acts.shape[0])
        ]).numpy()

        # Train classifier
        classifier = LogisticRegression(random_state=random_seed, max_iter=1000)
        classifier.fit(X_train, y_train)

        # Evaluate on train and val
        train_acc = accuracy_score(y_train, classifier.predict(X_train))
        val_acc = accuracy_score(y_val, classifier.predict(X_val))

        probe_results[layer_idx] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'classifier': classifier,
            'train_samples': len(y_train),
            'val_samples': len(y_val)
        }

    return probe_results


def compute_pca_analysis_by_layer(
    dataset: Dict[str, Any],
    positive_tags: List[str],
    negative_tags: List[str],
    n_components: int = 2,
    max_samples: int = 5000
) -> Dict[int, Dict[str, Any]]:
    """
    Compute PCA analysis for each layer to visualize separability.
    Uses full dataset for better representation.

    Args:
        dataset: Full activation dataset
        positive_tags: Tags to treat as positive class
        negative_tags: Tags to treat as negative class
        n_components: Number of PCA components (2 or 3 for visualization)
        max_samples: Maximum samples per class to use (for memory efficiency)

    Returns:
        Dictionary: {layer_idx: {
            'pca_model': fitted PCA object,
            'positive_transformed': transformed positive activations,
            'negative_transformed': transformed negative activations,
            'explained_variance': explained variance ratio,
            'explained_variance_cumulative': cumulative explained variance
        }}
    """
    positive_by_layer, negative_by_layer = extract_tag_activations(dataset, positive_tags, negative_tags)

    pca_results = {}

    print(f"Computing PCA with {n_components} components for {len(positive_by_layer)} layers...")

    for layer_idx in tqdm(positive_by_layer, desc="PCA analysis"):
        positive_acts = positive_by_layer[layer_idx]
        negative_acts = negative_by_layer[layer_idx]

        if positive_acts.numel() == 0 or negative_acts.numel() == 0:
            pca_results[layer_idx] = {
                'pca_model': None,
                'positive_transformed': None,
                'negative_transformed': None,
                'explained_variance': [],
                'explained_variance_cumulative': [],
                'error': 'Insufficient data'
            }
            continue

        # Sample if too many activations (for memory efficiency)
        if positive_acts.shape[0] > max_samples:
            indices = torch.randperm(positive_acts.shape[0])[:max_samples]
            positive_acts = positive_acts[indices]

        if negative_acts.shape[0] > max_samples:
            indices = torch.randperm(negative_acts.shape[0])[:max_samples]
            negative_acts = negative_acts[indices]

        # Combine for PCA fitting
        combined = torch.cat([positive_acts, negative_acts], dim=0).float().numpy()

        # Fit PCA
        pca = PCA(n_components=min(n_components, combined.shape[1], combined.shape[0]))
        pca.fit(combined)

        # Transform separately
        positive_transformed = pca.transform(positive_acts.float().numpy())
        negative_transformed = pca.transform(negative_acts.float().numpy())

        pca_results[layer_idx] = {
            'pca_model': pca,
            'positive_transformed': positive_transformed,
            'negative_transformed': negative_transformed,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'explained_variance_cumulative': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_positive': len(positive_transformed),
            'n_negative': len(negative_transformed)
        }

    return pca_results


def print_separability_summary(
    cosine_similarities: Dict[int, float],
    mean_differences: Dict[int, Dict[str, Any]],
    probe_results: Dict[int, Dict[str, Any]],
    positive_tags: List[str],
    negative_tags: List[str]
) -> None:
    """
    Print formatted summary of separability analysis results.

    Args:
        cosine_similarities: Results from compute_cosine_similarity_by_layer
        mean_differences: Results from compute_mean_differences_by_layer
        probe_results: Results from train_linear_probes_by_layer
        positive_tags: Positive class tags
        negative_tags: Negative class tags
    """
    print(f"\n=== SEPARABILITY ANALYSIS SUMMARY ===")
    print(f"Positive tags: {positive_tags}")
    print(f"Negative tags: {negative_tags}")
    print(f"Layers analyzed: {len(cosine_similarities)}")

    print(f"\n--- Cosine Similarity (between means) ---")
    cos_values = list(cosine_similarities.values())
    print(f"Range: {min(cos_values):.3f} to {max(cos_values):.3f}")
    print(f"Mean: {np.mean(cos_values):.3f}")

    print(f"\n--- Distance Between Means (||mean_pos - mean_neg||) ---")
    mean_diffs = [mean_differences[layer]['mean_diff_norm'] for layer in mean_differences]
    print(f"Range: {min(mean_diffs):.3f} to {max(mean_diffs):.3f}")
    print(f"Mean: {np.mean(mean_diffs):.3f}")

    print(f"\n--- Linear Probe Performance ---")
    valid_results = [r for r in probe_results.values() if 'error' not in r]
    if valid_results:
        train_accs = [r['train_acc'] for r in valid_results]
        val_accs = [r['val_acc'] for r in valid_results]
        print(f"Train accuracy - Range: {min(train_accs):.3f} to {max(train_accs):.3f}, Mean: {np.mean(train_accs):.3f}")
        print(f"Val accuracy - Range: {min(val_accs):.3f} to {max(val_accs):.3f}, Mean: {np.mean(val_accs):.3f}")

        # Find best performing layer
        best_layer = max(valid_results, key=lambda x: x['val_acc'])
        best_layer_idx = [k for k, v in probe_results.items() if v == best_layer][0]
        print(f"Best layer: {best_layer_idx} (train_acc: {best_layer['train_acc']:.3f}, val_acc: {best_layer['val_acc']:.3f})")
    else:
        print("No valid probe results found.")

    print(f"\n=== Analysis complete ===")