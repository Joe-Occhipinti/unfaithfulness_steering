"""
Helper functions for gradient-based steering using MLP probes.

These functions extend src/steering.py to support per-prompt steering vectors
computed via gradient optimization toward target logits.
"""

import torch
from typing import Dict, List, Optional, Any
from torch import nn


def compute_steering_vector_gradient(
    activation: torch.Tensor,
    mlp_model: nn.Module,
    target_value: float,  # This is 't' from the paper (always positive magnitude)
    direction: str = "offensive",  # "offensive" (increase logit) or "defensive" (decrease logit)
    learning_rate: float = 0.05,
    num_steps: int = 50
) -> torch.Tensor:
    """
    Compute steering vector by optimizing activation toward a dynamic target.
    
    Implements the logic from the paper:
    - Offensive: Push logit p toward max(t, p + t)
    - Defensive: Push logit p toward min(-t, p - t)
    
    Args:
        activation: Original activation tensor [4096]
        mlp_model: Trained MLP probe
        target_value: The magnitude 't' (e.g., 5, 10, 20)
        direction: "offensive" (positive/unfaithful) or "defensive" (negative/faithful)
        learning_rate: SGD learning rate
        num_steps: Number of optimization steps
    
    Returns:
        steering_vector: The delta to add to activation [4096]
    """
    # Clone original
    original = activation.clone().detach()
    
    # Get initial logit 'p'
    with torch.no_grad():
        initial_logit = mlp_model(original).item()
    
    # Determine dynamic target based on paper's formula
    if direction == "offensive":
        # Push toward max(t, p + t)
        # If p is low, push to t. If p is high, push further by t.
        target_logit = max(target_value, initial_logit + target_value)
    else:  # defensive
        # Push toward min(-t, p - t)
        # If p is high, push to -t. If p is low, push further down by t.
        target_logit = min(-target_value, initial_logit - target_value)
        
    # Optimize
    x = original.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=learning_rate)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        logit = mlp_model(x).squeeze()
        loss = (logit - target_logit) ** 2  # MSE to dynamic target
        
        loss.backward()
        optimizer.step()
    
    # Steering vector = difference between optimized and original
    steering_vector = (x.detach() - original)
    
    return steering_vector


def compute_batch_steering_vectors(
    prompts_activations: List[torch.Tensor],
    mlp_model: nn.Module,
    target_value: float,
    direction: str = "offensive",
    learning_rate: float = 0.05,
    num_steps: int = 50
) -> List[torch.Tensor]:
    """
    Compute steering vectors for a batch of prompts (sequential CPU version).
    """
    steering_vectors = []
    
    for activation in prompts_activations:
        if activation is None:
            steering_vectors.append(None)
            continue
        
        steering_vec = compute_steering_vector_gradient(
            activation=activation,
            mlp_model=mlp_model,
            target_value=target_value,
            direction=direction,
            learning_rate=learning_rate,
            num_steps=num_steps
        )
        
        steering_vectors.append(steering_vec)
    
    return steering_vectors


def compute_steering_vectors_gpu_batched(
    activations_list: List[Optional[torch.Tensor]],
    mlp_model: nn.Module,
    target_value: float,
    direction: str = "offensive",
    learning_rate: float = 0.05,
    num_steps: int = 50,
    device: str = "cuda"
) -> List[Optional[torch.Tensor]]:
    """
    Compute steering vectors for all prompts in parallel on GPU.
    
    This is significantly faster than sequential CPU computation (~10-100x speedup).
    
    Args:
        activations_list: List of activation tensors [hidden_dim] or None
        mlp_model: Trained MLP probe
        target_value: The magnitude 't' (e.g., 5, 10, 20)
        direction: "offensive" or "defensive"
        learning_rate: SGD learning rate
        num_steps: Number of optimization steps
        device: Device to use (default: "cuda")
    
    Returns:
        List of steering vectors (same length as input, None preserved)
    """
    # Separate valid activations from None entries
    valid_indices = []
    valid_activations = []
    
    for i, act in enumerate(activations_list):
        if act is not None:
            valid_indices.append(i)
            valid_activations.append(act.float())
    
    if not valid_activations:
        return [None] * len(activations_list)
    
    # Stack valid activations into batch tensor and move to GPU
    batch_activations = torch.stack(valid_activations).to(device)  # [N, hidden_dim]
    original_batch = batch_activations.clone().detach()
    
    # Move MLP to GPU
    mlp_model = mlp_model.to(device).float()
    
    # Get initial logits for dynamic target computation
    with torch.no_grad():
        initial_logits = mlp_model(original_batch).squeeze(-1)  # [N]
    
    # Compute per-prompt dynamic targets
    if direction == "offensive":
        # target = max(t, p + t) for each prompt
        target_logits = torch.maximum(
            torch.full_like(initial_logits, target_value),
            initial_logits + target_value
        )
    else:  # defensive
        # target = min(-t, p - t) for each prompt
        target_logits = torch.minimum(
            torch.full_like(initial_logits, -target_value),
            initial_logits - target_value
        )
    
    # Initialize optimized activations
    x = original_batch.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=learning_rate)
    
    # Batched optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        
        logits = mlp_model(x).squeeze(-1)  # [N]
        loss = ((logits - target_logits) ** 2).mean()  # MSE across batch
        
        loss.backward()
        optimizer.step()
    
    # Compute steering vectors = optimized - original
    steering_vectors_batch = (x.detach() - original_batch).cpu()  # [N, hidden_dim]
    
    # Reassemble results with None entries preserved
    results = [None] * len(activations_list)
    for batch_idx, original_idx in enumerate(valid_indices):
        results[original_idx] = steering_vectors_batch[batch_idx]
    
    return results


def extract_prompt_activations(
    prompt_indices: List[int],
    activations_dict: Dict[int, Dict],
    layer_idx: int,
    preferred_tag: str = 'F_body'
) -> List[Optional[torch.Tensor]]:
    """
    Extract activations for a batch of prompts from the activation dataset.
    """
    batch_activations = []
    
    for prompt_idx in prompt_indices:
        if prompt_idx not in activations_dict:
            batch_activations.append(None)
            continue
        
        layer_data = activations_dict[prompt_idx]['layers'][layer_idx]
        
        # Try preferred tag first, then fallback
        if preferred_tag in layer_data and layer_data[preferred_tag].numel() > 0:
            activation_tensor = layer_data[preferred_tag]
        elif 'U_body' in layer_data and layer_data['U_body'].numel() > 0:
            activation_tensor = layer_data['U_body']
        elif 'F_body' in layer_data and layer_data['F_body'].numel() > 0:
            activation_tensor = layer_data['F_body']
        else:
            batch_activations.append(None)
            continue
        
        # Average over tokens
        mean_activation = activation_tensor.mean(dim=0)  # [4096]
        batch_activations.append(mean_activation)
    
    return batch_activations
