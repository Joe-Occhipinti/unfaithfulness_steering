"""
Per-prompt steering wrapper for gradient-based steering.

Extends the steering framework to support different steering vectors
for different prompts in the same batch.
"""

import torch
from typing import Optional, List


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
    
    def _get_block(self) -> torch.nn.Module:
        """Access the wrapped block without triggering __getattr__."""
        return object.__getattribute__(self, '_modules')['block']
    
    def __getattr__(self, name: str):
        """Proxy attribute access to wrapped block for model compatibility (e.g., Qwen3's attention_type)."""
        # nn.Module stores submodules in _modules, not __dict__
        # Use object.__getattribute__ to avoid recursion
        try:
            modules = object.__getattribute__(self, '_modules')
            block = modules.get('block')
            if block is not None:
                return getattr(block, name)
        except (AttributeError, KeyError):
            pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def forward(self, *args, **kwargs):
        """Forward pass with per-prompt steering applied to last token."""
        block = self._get_block()
        output = block(*args, **kwargs)
        
        # Get steering_vectors and active via object.__getattribute__ to avoid __getattr__
        active = object.__getattribute__(self, 'active')
        steering_vectors = object.__getattribute__(self, 'steering_vectors')
        coefficient = object.__getattribute__(self, 'coefficient')
        layer_idx = object.__getattribute__(self, 'layer_idx')
        
        if active and steering_vectors is not None and output is not None:
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
                        if batch_idx < len(steering_vectors):
                            steering_vec = steering_vectors[batch_idx]
                            
                            if steering_vec is not None:
                                # Add to last token
                                modified_hidden_states[batch_idx, -1, :] = (
                                    modified_hidden_states[batch_idx, -1, :] +
                                    coefficient * steering_vec
                                )
                    
                    # Return in same format
                    if isinstance(output, tuple):
                        output = (modified_hidden_states,) + output[1:]
                    else:
                        output = modified_hidden_states
            
            except Exception as e:
                print(f"Error in layer {layer_idx} per-prompt steering: {e}")
        
        return output
    
    def set_steering(self, vectors: List, coefficient: float = 1.0):
        """
        Set per-prompt steering vectors.
        
        Args:
            vectors: List of tensors [hidden_dim], one per batch element
                     Can include None for prompts that shouldn't be steered
            coefficient: Scaling factor (usually 1.0 for gradient steering)
        """
        # Get device from wrapped block
        block = self._get_block()
        
        # Try to find device from block's parameters
        device = 'cuda'
        try:
            for param in block.parameters():
                device = param.device
                break
        except:
            pass
        
        processed_vectors = []
        for vec in vectors:
            if vec is not None:
                processed_vectors.append(vec.to(device))
            else:
                processed_vectors.append(None)
        
        # Use object.__setattr__ to set attributes without triggering __getattr__
        object.__setattr__(self, 'steering_vectors', processed_vectors)
        object.__setattr__(self, 'coefficient', coefficient)
        object.__setattr__(self, 'active', True)
    
    def reset(self):
        """Reset to inactive state."""
        object.__setattr__(self, 'active', False)
        object.__setattr__(self, 'steering_vectors', None)
        object.__setattr__(self, 'coefficient', 1.0)


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
