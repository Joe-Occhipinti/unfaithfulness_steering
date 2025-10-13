"""
separability.py

Core logic for analyzing separability between positive and negative activations.
Provides three separate investigations: cosine similarity, norm distributions, and linear separability.
"""

import torch
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any
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
    negative_tags: List[str] = None
) -> Dict[str, Dict]:
    """
    Split dataset by prompt indices to ensure activations from same prompt stay together.

    Args:
        dataset: Activation dataset with prompt-wise structure
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        random_seed: Random seed for reproducibility
        balance_val_split: If True, balance positive and negative samples in val split
        positive_tags: Tags for positive class (required if balance_val_split=True)
        negative_tags: Tags for negative class (required if balance_val_split=True)

    Returns:
        Dictionary with train/val splits: {'train': dataset_subset, 'val': dataset_subset}
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    data = dataset['data']
    info = dataset['info']

    # Get actual prompt indices from the dataset
    prompt_indices = list(data.keys())
    total_prompts = len(prompt_indices)

    # Shuffle prompt indices
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    np.random.shuffle(prompt_indices)

    # Split indices
    train_end = int(train_ratio * total_prompts)

    train_prompts = prompt_indices[:train_end]
    val_prompts = prompt_indices[train_end:]

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
            pos_count = sum(prompt_data[0].get(tag, torch.empty(0)).shape[0]
                          for tag in positive_tags if tag in prompt_data[0])
            neg_count = sum(prompt_data[0].get(tag, torch.empty(0)).shape[0]
                          for tag in negative_tags if tag in prompt_data[0])
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
            pos_count = sum(prompt_data[0].get(tag, torch.empty(0)).shape[0]
                          for tag in positive_tags if tag in prompt_data[0])
            neg_count = sum(prompt_data[0].get(tag, torch.empty(0)).shape[0]
                          for tag in negative_tags if tag in prompt_data[0])
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
    split: str = None
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Extract and combine activations for positive and negative tags by layer.

    Args:
        dataset: Activation dataset or split with prompt-wise structure
        positive_tags: List of tags to treat as positive class
        negative_tags: List of tags to treat as negative class
        split: Split name if dataset contains multiple splits

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

        for layer_idx in range(num_layers):
            if layer_idx in prompt_data:
                layer_data = prompt_data[layer_idx]

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


def train_linear_probes_by_layer(
    dataset_splits: Dict[str, Dict],
    positive_tags: List[str],
    negative_tags: List[str],
    random_seed: int = 42
) -> Dict[int, Dict[str, Any]]:
    """
    Train linear probes for each layer using train/val splits.

    Args:
        dataset_splits: Dictionary with train/val splits
        positive_tags: Tags to treat as positive class (label 0)
        negative_tags: Tags to treat as negative class (label 1)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary: {layer_idx: {
            'train_acc': float,
            'val_acc': float,
            'classifier': LogisticRegression model,
            'train_samples': int,
            'val_samples': int
        }}
    """
    # Extract activations for each split
    train_pos, train_neg = extract_tag_activations(dataset_splits, positive_tags, negative_tags, 'train')
    val_pos, val_neg = extract_tag_activations(dataset_splits, positive_tags, negative_tags, 'val')

    probe_results = {}

    # Debug: Check sample counts
    print(f"\nDebug - Sample counts for layer 0:")
    print(f"  Train: F={train_pos[0].shape[0] if 0 in train_pos else 0}, U={train_neg[0].shape[0] if 0 in train_neg else 0}")
    print(f"  Val: F={val_pos[0].shape[0] if 0 in val_pos else 0}, U={val_neg[0].shape[0] if 0 in val_neg else 0}")

    print(f"\nTraining linear probes for {len(train_pos)} layers...")

    # Check if we need to apply balancing
    train_balance_info = dataset_splits['train']['info'].get('balance_info', {'balance_enabled': False})
    val_balance_info = dataset_splits['val']['info'].get('balance_info', {'balance_enabled': False})

    for layer_idx in tqdm(train_pos, desc="Training probes"):
        # Get activations for this layer
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

        # Apply downbalancing to training data if enabled
        if train_balance_info['balance_enabled']:
            target_samples = train_balance_info['target_samples_per_class']
            n_pos = train_pos_acts.shape[0]
            n_neg = train_neg_acts.shape[0]

            # Downsample the larger class
            if n_pos > target_samples:
                indices = torch.randperm(n_pos, generator=torch.Generator().manual_seed(random_seed))[:target_samples]
                train_pos_acts = train_pos_acts[indices]
            if n_neg > target_samples:
                indices = torch.randperm(n_neg, generator=torch.Generator().manual_seed(random_seed))[:target_samples]
                train_neg_acts = train_neg_acts[indices]

        # Prepare training data
        X_train = torch.cat([train_pos_acts, train_neg_acts], dim=0).float().numpy()
        y_train = torch.cat([
            torch.zeros(train_pos_acts.shape[0]),  # 0 for positive
            torch.ones(train_neg_acts.shape[0])    # 1 for negative
        ]).numpy()

        # Apply downbalancing to validation data if enabled
        if val_balance_info['balance_enabled']:
            target_samples = val_balance_info['target_samples_per_class']
            n_pos = val_pos_acts.shape[0]
            n_neg = val_neg_acts.shape[0]

            # Downsample the larger class
            if n_pos > target_samples:
                indices = torch.randperm(n_pos, generator=torch.Generator().manual_seed(random_seed))[:target_samples]
                val_pos_acts = val_pos_acts[indices]
            if n_neg > target_samples:
                indices = torch.randperm(n_neg, generator=torch.Generator().manual_seed(random_seed))[:target_samples]
                val_neg_acts = val_neg_acts[indices]

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