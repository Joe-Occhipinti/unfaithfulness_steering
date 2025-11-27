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
    Compute steering vectors for a batch of prompts.
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
