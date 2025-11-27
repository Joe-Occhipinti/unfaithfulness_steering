"""
Per-prompt steering wrapper for gradient-based steering.

Extends the steering framework to support different steering vectors
for different prompts in the same batch.
"""

import torch
from typing import Optional


class PerPromptSteeringWrapper(torch.nn.Module):
    """
    Wrapper for model layers to apply DIFFERENT steering vectors to each prompt in batch.
    
    This differs from LayerSteeringWrapper which applies the SAME vector to all prompts.
    """
    
    def __init__(self, block: torch.nn.Module, layer_idx: int):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.steering_vectors = None  # List of vectors, one per batch element
        self.coefficient = 1.0  # Usually 1.0 for gradient steering
        self.active = False
    
    def forward(self, *args, **kwargs):
        """Forward pass with per-prompt steering applied to last token."""
        output = self.block(*args, **kwargs)
        
        if self.active and self.steering_vectors is not None and output is not None:
            try:
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                    batch_size = hidden_states.shape[0]
                    modified_hidden_states = hidden_states.clone()
                    
                    # Apply different steering vector to each batch element
                    for batch_idx in range(batch_size):
                        if batch_idx < len(self.steering_vectors):
                            steering_vec = self.steering_vectors[batch_idx]
                            
                            if steering_vec is not None:
                                # Add to last token
                                modified_hidden_states[batch_idx, -1, :] = (
                                    modified_hidden_states[batch_idx, -1, :] +
                                    self.coefficient * steering_vec
                                )
                    
                    # Return in same format
                    if isinstance(output, tuple):
                        output = (modified_hidden_states,) + output[1:]
                    else:
                        output = modified_hidden_states
            
            except Exception as e:
                print(f"Error in layer {self.layer_idx} per-prompt steering: {e}")
        
        return output
    
    def set_steering(self, vectors: list, coefficient: float = 1.0):
        """
        Set per-prompt steering vectors.
        
        Args:
            vectors: List of tensors [hidden_dim], one per batch element
                     Can include None for prompts that shouldn't be steered
            coefficient: Scaling factor (usually 1.0 for gradient steering)
        """
        # Move all vectors to device
        device = self.block.weight.device if hasattr(self.block, 'weight') else 'cuda'
        
        self.steering_vectors = []
        for vec in vectors:
            if vec is not None:
                self.steering_vectors.append(vec.to(device))
            else:
                self.steering_vectors.append(None)
        
        self.coefficient = coefficient
        self.active = True
    
    def reset(self):
        """Reset to inactive state."""
        self.active = False
        self.steering_vectors = None
        self.coefficient = 1.0


def apply_per_prompt_steering_to_model(
    model,
    layers_to_wrap: list
) -> dict:
    """
    Replace model layers with per-prompt steering wrappers.
    
    Args:
        model: The HuggingFace model
        layers_to_wrap: List of layer indices to wrap
    
    Returns:
        Dict of layer_idx -> wrapper instance
    """
    wrapped_layers = {}
    
    for layer_idx in layers_to_wrap:
        original_layer = model.model.layers[layer_idx]
        wrapped_layer = PerPromptSteeringWrapper(original_layer, layer_idx)
        model.model.layers[layer_idx] = wrapped_layer
        wrapped_layers[layer_idx] = wrapped_layer
    
    print(f"Applied per-prompt steering wrappers to {len(wrapped_layers)} layers")
    return wrapped_layers
