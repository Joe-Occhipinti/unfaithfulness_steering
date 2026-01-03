"""
convert_pkl_to_gguf.py

Utility to convert pickle steering vectors to GGUF format for EasySteer.

Usage:
    # As a module:
    from utilities.convert_pkl_to_gguf import convert_pkl_to_gguf
    gguf_paths = convert_pkl_to_gguf(pkl_path, output_dir, layers=[8, 13, 15])
    
    # As a standalone script:
    python utilities/convert_pkl_to_gguf.py vectors.pkl output_dir --layers 8 13 15
"""

import pickle
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Import gguf library (required for EasySteer)
try:
    import gguf
except ImportError:
    raise ImportError(
        "gguf library required. Install with: pip install gguf"
    )


def convert_pkl_to_gguf(
    pkl_path: str,
    output_dir: str,
    layers: Optional[List[int]] = None,
    model_type: str = "llama",
    method: str = "diffmean"
) -> Dict[int, str]:
    """
    Convert pickle steering vectors to per-layer GGUF files.
    
    Args:
        pkl_path: Path to pickle file with structure {'steering_vectors': {layer: tensor}}
        output_dir: Directory to save GGUF files
        layers: Optional list of layers to convert (default: all available)
        model_type: Model type hint for GGUF metadata (default: "llama")
        method: Method hint for GGUF metadata (default: "diffmean")
        
    Returns:
        Dict mapping layer index to GGUF file path
    """
    print(f"Loading steering vectors from: {pkl_path}")
    
    # Load pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract steering vectors
    if 'steering_vectors' in data:
        steering_vectors = data['steering_vectors']
    else:
        # Assume the pickle is the vectors dict directly
        steering_vectors = data
    
    available_layers = sorted(steering_vectors.keys())
    print(f"Found vectors for {len(available_layers)} layers: {available_layers}")
    
    # Filter to requested layers if specified
    if layers is not None:
        layers_to_convert = [l for l in layers if l in steering_vectors]
        if len(layers_to_convert) != len(layers):
            missing = set(layers) - set(layers_to_convert)
            print(f"Warning: Requested layers not found in pkl: {missing}")
    else:
        layers_to_convert = available_layers
    
    print(f"Converting {len(layers_to_convert)} layers to GGUF")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each layer to a separate GGUF file
    gguf_paths = {}
    
    for layer_idx in layers_to_convert:
        vector = steering_vectors[layer_idx]
        
        # Convert torch tensor to numpy if needed
        if hasattr(vector, 'cpu'):
            vector = vector.cpu().float().numpy()
        elif not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        # Ensure float32
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        
        # Create GGUF file for this layer
        gguf_path = output_dir / f"steering_layer_{layer_idx}.gguf"
        
        # Write GGUF file using EasySteer-compatible format
        arch = "controlvector"
        writer = gguf.GGUFWriter(str(gguf_path), arch)
        
        # Add metadata
        writer.add_string(f"{arch}.model_hint", model_type)
        writer.add_string(f"{arch}.method", method)
        writer.add_uint32(f"{arch}.layer_count", 1)  # Single layer per file
        
        # Add the steering vector tensor
        # Use the layer index as the key (e.g., "direction.13")
        writer.add_tensor(f"direction.{layer_idx}", vector)
        
        # Write file
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        
        gguf_paths[layer_idx] = str(gguf_path)
        print(f"  Layer {layer_idx}: {gguf_path} (shape: {vector.shape})")
    
    print(f"Conversion complete: {len(gguf_paths)} GGUF files created")
    return gguf_paths


def verify_gguf(gguf_path: str) -> Dict:
    """
    Verify a GGUF file and return its contents.
    
    Args:
        gguf_path: Path to GGUF file
        
    Returns:
        Dict with metadata and tensor info
    """
    reader = gguf.GGUFReader(gguf_path)
    
    info = {
        'tensors': {},
        'metadata': {}
    }
    
    # Extract tensors
    for tensor in reader.tensors:
        info['tensors'][tensor.name] = {
            'shape': tensor.data.shape,
            'dtype': str(tensor.data.dtype)
        }
    
    # Extract metadata fields
    for field_name, field in reader.fields.items():
        if field_name.startswith("controlvector."):
            key = field_name.replace("controlvector.", "")
            try:
                info['metadata'][key] = field.parts[0] if field.parts else None
            except:
                pass
    
    return info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert pickle steering vectors to GGUF format"
    )
    parser.add_argument("pkl_path", help="Path to pickle file")
    parser.add_argument("output_dir", help="Output directory for GGUF files")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to convert (default: all)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llama",
        help="Model type hint (default: llama)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify GGUF files after conversion"
    )
    
    args = parser.parse_args()
    
    gguf_paths = convert_pkl_to_gguf(
        args.pkl_path,
        args.output_dir,
        layers=args.layers,
        model_type=args.model_type
    )
    
    if args.verify:
        print("\n=== Verifying GGUF files ===")
        for layer, path in gguf_paths.items():
            info = verify_gguf(path)
            print(f"Layer {layer}:")
            for tensor_name, tensor_info in info['tensors'].items():
                print(f"  {tensor_name}: {tensor_info}")
