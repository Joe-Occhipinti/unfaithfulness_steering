"""
src/probe.py

Comprehensive module for training and evaluating linear and non-linear probes
on LLM activations to detect faithfulness vs unfaithfulness.

Includes:
- Data loading and balancing
- Model definitions (Logistic Regression, MLP)
- Training functions with early stopping
- Evaluation and analysis functions
"""

import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# =============================================================================
# DATA LOADING AND BALANCING
# =============================================================================

def load_dataset(pkl_path: str) -> Dict:
    """Load activation dataset from pickle file."""
    print(f"Loading dataset from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded: {len(dataset['data'])} prompts")
    return dataset


def extract_activations_by_subclass(
    dataset: Dict,
    layer_idx: int,
    tags: List[str] = ['F_body', 'U_body'],
    filter_field: str = 'correct_hint',
    filter_value: str = 'False'
) -> Dict[Tuple[str, str, str], List[torch.Tensor]]:
    """
    Extract activations grouped by (tag, hint_template, split).
    
    Args:
        dataset: Loaded activation dataset
        layer_idx: Layer index to extract from
        tags: Which tags to extract (default: F_body, U_body)
        filter_field: Metadata field to filter on
        filter_value: Value to filter for
    
    Returns:
        Dictionary mapping (tag, hint_template, split) -> list of activation tensors
    """
    subclass_data = defaultdict(list)
    
    for prompt_idx, prompt_data in dataset['data'].items():
        metadata = prompt_data['metadata']
        
        # Filter by correct_hint
        if metadata.get(filter_field) != filter_value:
            continue
        
        hint_template = metadata['hint_template']
        split = metadata['split']
        
        # Get layer data
        if layer_idx not in prompt_data['layers']:
            continue
        
        layer_data = prompt_data['layers'][layer_idx]
        
        # Extract activations for each tag
        for tag in tags:
            if tag in layer_data:
                tensor = layer_data[tag]  # Shape: [num_tokens, hidden_dim]
                
                # Store each activation (row) separately
                for activation in tensor:
                    subclass_data[(tag, hint_template, split)].append(activation)
    
    return subclass_data


def find_minority_count(
    subclass_data: Dict[Tuple[str, str, str], List],
    split: str
) -> int:
    """Find the minimum count among all subclasses in a given split."""
    counts = []
    
    for (tag, hint_template, s), activations in subclass_data.items():
        if s == split:
            counts.append(len(activations))
    
    if not counts:
        raise ValueError(f"No data found for split: {split}")
    
    return min(counts)


def downsample_to_minority(
    subclass_data: Dict[Tuple[str, str, str], List],
    split: str,
    target_count: int,
    random_seed: int = 42
) -> Dict[Tuple[str, str, str], List]:
    """
    Downsample all subclasses to target count for perfect balance.
    
    Args:
        subclass_data: Dictionary of subclass -> activations
        split: Which split to downsample ('train' or 'val')
        target_count: Target number of samples per subclass
        random_seed: Random seed for reproducibility
    
    Returns:
        Balanced data dictionary
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    balanced_data = {}
    
    for (tag, hint_template, s), activations in subclass_data.items():
        if s != split:
            continue
        
        if len(activations) > target_count:
            # Random downsample
            indices = random.sample(range(len(activations)), target_count)
            sampled = [activations[i] for i in indices]
        else:
            # Keep all (shouldn't happen if target_count is truly the minority)
            sampled = activations
        
        balanced_data[(tag, hint_template, s)] = sampled
    
    return balanced_data


def create_datasets(
    balanced_data: Dict[Tuple[str, str, str], List],
    tags: List[str],
    hint_templates: List[str],
    split: str
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, str]]]:
    """
    Create X, y tensors and metadata list from balanced data.
    
    Args:
        balanced_data: Balanced activation data
        tags: List of tags (e.g., ['F_body', 'U_body'])
        hint_templates: List of hint templates
        split: Which split to create ('train' or 'val')
    
    Returns:
        X: Activation tensor [num_samples, hidden_dim]
        y: Label tensor [num_samples]
        metadata: List of (tag, hint_template) for each sample
    """
    X_list = []
    y_list = []
    metadata_list = []
    
    for tag in tags:
        label = 0 if tag == 'F_body' else 1
        
        for hint_template in hint_templates:
            key = (tag, hint_template, split)
            if key not in balanced_data:
                continue
            
            activations = balanced_data[key]
            X_list.extend(activations)
            y_list.extend([label] * len(activations))
            metadata_list.extend([(tag, hint_template)] * len(activations))
    
    # Convert to tensors
    X = torch.stack(X_list).float()  # Convert to float32
    y = torch.tensor(y_list, dtype=torch.float32)
    
    return X, y, metadata_list


def load_balanced_data_for_layer(
    dataset: Dict,
    layer_idx: int,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Complete pipeline: load and balance data for one layer.
    
    Returns:
        train_X, train_y, val_X, val_y, train_metadata, val_metadata
    """
    # Extract activations
    subclass_data = extract_activations_by_subclass(dataset, layer_idx)
    
    # Find minority counts per split
    train_minority = find_minority_count(subclass_data, 'train')
    val_minority = find_minority_count(subclass_data, 'val')
    
    # Downsample to minority
    train_balanced = downsample_to_minority(subclass_data, 'train', train_minority, random_seed)
    val_balanced = downsample_to_minority(subclass_data, 'val', val_minority, random_seed)
    
    # Merge
    balanced_data = {**train_balanced, **val_balanced}
    
    # Get unique hint templates
    hint_templates = list(set(ht for (_, ht, _) in balanced_data.keys()))
    tags = ['F_body', 'U_body']
    
    # Create datasets
    train_X, train_y, train_metadata = create_datasets(balanced_data, tags, hint_templates, 'train')
    val_X, val_y, val_metadata = create_datasets(balanced_data, tags, hint_templates, 'val')
    
    return train_X, train_y, val_X, val_y, train_metadata, val_metadata


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class MLPProbe(nn.Module):
    """
    MLP probe with configurable architecture.
    
    Architecture:
        Input (input_dim) -> [Linear -> ReLU] × num_hidden_layers -> Linear -> Output (1)
    """
    
    def __init__(self, input_dim: int = 4096, hidden_dim: int = 8, num_hidden_layers: int = 2):
        super().__init__()
        
        layers = []
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (binary classification)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Store config for serialization
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
    
    def forward(self, x):
        """Forward pass. Returns logits (no sigmoid)."""
        return self.network(x)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_logistic_probe(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> Tuple[LogisticRegression, Dict]:
    """
    Train logistic regression probe.
    
    Args:
        train_X: Training activations [num_samples, hidden_dim]
        train_y: Training labels [num_samples]
        val_X: Validation activations
        val_y: Validation labels
        C: Inverse regularization strength
        max_iter: Maximum iterations
        random_state: Random seed
    
    Returns:
        model: Trained sklearn LogisticRegression
        metrics: Dictionary of performance metrics
    """
    # Convert to numpy
    X_train = train_X.cpu().numpy()
    y_train = train_y.cpu().numpy()
    X_val = val_X.cpu().numpy()
    y_val = val_y.cpu().numpy()
    
    # Train
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, val_preds, average='binary', zero_division=0
    )
    
    metrics = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_f1': float(val_f1),
        'train_predictions': train_preds,
        'val_predictions': val_preds,
    }
    
    return model, metrics


def train_mlp_probe(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    hidden_dim: int = 8,
    num_hidden_layers: int = 2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    max_epochs: int = 200,
    weight_decay: float = 0.01,
    patience: int = 20,
    min_delta: float = 0.0001,
    random_seed: int = 42,
    verbose: bool = False
) -> Tuple[MLPProbe, Dict]:
    """
    Train MLP probe with early stopping.
    
    Args:
        train_X: Training activations
        train_y: Training labels
        val_X: Validation activations
        val_y: Validation labels
        hidden_dim: Neurons per hidden layer
        num_hidden_layers: Number of hidden layers
        learning_rate: Adam learning rate
        batch_size: Batch size for training
        max_epochs: Maximum training epochs
        weight_decay: L2 regularization strength
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        random_seed: Random seed for reproducibility
        verbose: Print training progress
    
    Returns:
        model: Trained MLPProbe
        metrics: Dictionary of performance metrics and training history
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Initialize model
    input_dim = train_X.shape[1]
    model = MLPProbe(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Data loader
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            logits = model(batch_X).squeeze()
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Compute accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X).squeeze()
            val_loss = criterion(val_logits, val_y).item()
            
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == val_y).float().mean().item()
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Early stopping check
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if verbose and (epoch % 20 == 0 or epoch < 5):
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_logits = model(train_X).squeeze()
        train_preds = (torch.sigmoid(train_logits) > 0.5).float()
        train_acc_final = (train_preds == train_y).float().mean().item()
        
        val_logits = model(val_X).squeeze()
        val_preds = (torch.sigmoid(val_logits) > 0.5).float()
        val_acc_final = (val_preds == val_y).float().mean().item()
        
        # Additional metrics for val
        val_preds_np = val_preds.cpu().numpy()
        val_y_np = val_y.cpu().numpy()
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_y_np, val_preds_np, average='binary', zero_division=0
        )
    
    metrics = {
        'train_accuracy': float(train_acc_final),
        'val_accuracy': float(val_acc_final),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_f1': float(val_f1),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
        'train_predictions': train_preds.cpu().numpy(),
        'val_predictions': val_preds.cpu().numpy(),
    }
    
    return model, metrics


# =============================================================================
# EVALUATION AND ANALYSIS
# =============================================================================

def compute_per_template_performance(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    metadata: List[Tuple[str, str]],
    model_type: str = 'mlp'
) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy per hint_template.
    
    Args:
        model: Trained model (LogisticRegression or MLPProbe)
        X: Activation data
        y: True labels
        metadata: List of (tag, hint_template) for each sample
        model_type: 'logreg' or 'mlp'
    
    Returns:
        Dictionary mapping hint_template -> {accuracy, correct, total}
    """
    # Get predictions
    if model_type == 'logreg':
        predictions = model.predict(X.cpu().numpy())
    else:  # mlp
        model.eval()
        with torch.no_grad():
            logits = model(X).squeeze()
            predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy()
    
    # Group by hint_template
    template_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    y_np = y.cpu().numpy()
    for i, (pred, true_label) in enumerate(zip(predictions, y_np)):
        tag, hint_template = metadata[i]
        
        template_results[hint_template]['total'] += 1
        if pred == true_label:
            template_results[hint_template]['correct'] += 1
    
    # Compute accuracies
    for template in template_results:
        correct = template_results[template]['correct']
        total = template_results[template]['total']
        template_results[template]['accuracy'] = correct / total if total > 0 else 0.0
    
    return dict(template_results)


def print_layer_results(layer_idx: int, logreg_metrics: Dict, mlp_metrics: Dict):
    """Print formatted results for one layer."""
    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx} RESULTS")
    print(f"{'='*80}")
    print(f"Logistic Regression:")
    print(f"  Train Acc: {logreg_metrics['train_accuracy']:.4f}")
    print(f"  Val Acc:   {logreg_metrics['val_accuracy']:.4f}")
    print(f"  Val F1:    {logreg_metrics['val_f1']:.4f}")
    
    print(f"\nMLP Probe:")
    print(f"  Train Acc: {mlp_metrics['train_accuracy']:.4f}")
    print(f"  Val Acc:   {mlp_metrics['val_accuracy']:.4f}")
    print(f"  Val F1:    {mlp_metrics['val_f1']:.4f}")
    print(f"  Best Epoch: {mlp_metrics['best_epoch']}/{mlp_metrics['total_epochs']}")
