#!/usr/bin/env python3
"""
Contrastive Activation Dataset Builder

This script builds a contrastive dataset by matching layer 15 activations 
with their corresponding faithfulness labels from the faithful/unfaithful dataset.
The output can be used for contrastive activation addition to compute steering vectors.

Usage:
    python build_contrastive_activation_dataset.py

Configuration:
    - FAITHFUL_UNFAITHFUL_FILE: Path to faithful/unfaithful labeled dataset
    - ACTIVATIONS_FOLDER: Path to folder containing activation files
    - OUTPUT_FILE: Path to output contrastive dataset
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pickle

def load_activation_file(activation_path: Path) -> Optional[torch.Tensor]:
    """Load activation tensor from .pt file."""
    try:
        activation = torch.load(activation_path, map_location='cpu')
        return activation
    except Exception as e:
        print(f"Error loading {activation_path}: {e}")
        return None

def build_contrastive_dataset(faithful_unfaithful_file: str, activations_folder: str, 
                            output_file: str) -> None:
    """
    Build contrastive activation dataset matching activations with faithfulness labels.
    
    Args:
        faithful_unfaithful_file: Path to faithful/unfaithful labeled dataset
        activations_folder: Path to folder with activation files
        output_file: Path to output contrastive dataset
    """
    faithful_unfaithful_path = Path(faithful_unfaithful_file)
    activations_path = Path(activations_folder)
    output_path = Path(output_file)
    
    if not faithful_unfaithful_path.exists():
        raise FileNotFoundError(f"Faithful/unfaithful file not found: {faithful_unfaithful_file}")
    if not activations_path.exists():
        raise FileNotFoundError(f"Activations folder not found: {activations_folder}")
    
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Faithful/unfaithful file: {faithful_unfaithful_file}")
    print(f"Activations folder: {activations_folder}")
    print(f"Output file: {output_file}")
    print("-" * 60)
    
    # Load faithful/unfaithful dataset
    print("Loading faithful/unfaithful dataset...")
    labeled_data = []
    with open(faithful_unfaithful_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                faithfulness_class = data.get('faithfulness_classification', {})
                faithfulness_result = faithfulness_class.get('faithfulness')
                
                if faithfulness_result in ['faithful', 'unfaithful']:
                    labeled_data.append({
                        'prompt_index': line_num - 1,  # 0-indexed for matching activation files
                        'faithfulness_label': faithfulness_result,
                        'question': data.get('question', ''),
                        'original_letter': data.get('extracted_letter', ''),
                        'biased_letter': data.get('extracted_biased_letter', ''),
                        'hint_letter': data.get('biased_hint_letter', ''),
                        'forward_pass_prompt': data.get('forward_pass_prompt', '')
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    
    print(f"Loaded {len(labeled_data)} labeled entries")
    
    # Find available activation files
    print("Scanning activation files...")
    activation_files = list(activations_path.glob("prompt_*_activations.pt"))
    activation_files.sort(key=lambda x: int(x.stem.split('_')[1]))  # Sort by prompt number
    
    print(f"Found {len(activation_files)} activation files")
    
    # Build contrastive dataset
    print("Building contrastive dataset...")
    contrastive_data = {
        'faithful_activations': [],
        'unfaithful_activations': [],
        'faithful_metadata': [],
        'unfaithful_metadata': [],
        'dataset_info': {
            'creation_timestamp': datetime.now().isoformat(),
            'source_faithful_unfaithful_file': faithful_unfaithful_file,
            'source_activations_folder': activations_folder,
            'layer': 15,
            'total_entries_processed': 0,
            'faithful_count': 0,
            'unfaithful_count': 0,
            'missing_activations': 0
        }
    }
    
    processed_count = 0
    faithful_count = 0
    unfaithful_count = 0
    missing_activations = 0
    
    for entry in labeled_data:
        prompt_idx = entry['prompt_index']
        faithfulness_label = entry['faithfulness_label']
        
        # Find corresponding activation file
        activation_file = activations_path / f"prompt_{prompt_idx}_activations.pt"
        
        if not activation_file.exists():
            missing_activations += 1
            print(f"Warning: Missing activation file for prompt {prompt_idx}")
            continue
        
        # Load activation
        activation_dict = load_activation_file(activation_file)
        if activation_dict is None:
            missing_activations += 1
            continue
        
        # Extract layer 15 activation from the dictionary
        if isinstance(activation_dict, dict) and 15 in activation_dict:
            activation = activation_dict[15]
        else:
            print(f"Warning: Layer 15 not found in {activation_file}")
            missing_activations += 1
            continue
        
        # Convert to numpy for easier handling
        if isinstance(activation, torch.Tensor):
            # Convert BFloat16 to Float32 for numpy compatibility
            activation_np = activation.detach().cpu().float().numpy()
        else:
            activation_np = np.array(activation)
        
        # Add to appropriate category
        metadata = {
            'prompt_index': prompt_idx,
            'question': entry['question'][:100] + '...' if len(entry['question']) > 100 else entry['question'],
            'original_letter': entry['original_letter'],
            'biased_letter': entry['biased_letter'],
            'hint_letter': entry['hint_letter'],
            'activation_shape': activation_np.shape,
            'activation_file': str(activation_file.name)
        }
        
        if faithfulness_label == 'faithful':
            contrastive_data['faithful_activations'].append(activation_np)
            contrastive_data['faithful_metadata'].append(metadata)
            faithful_count += 1
        elif faithfulness_label == 'unfaithful':
            contrastive_data['unfaithful_activations'].append(activation_np)
            contrastive_data['unfaithful_metadata'].append(metadata)
            unfaithful_count += 1
        
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} entries...")
    
    # Update dataset info
    contrastive_data['dataset_info']['total_entries_processed'] = processed_count
    contrastive_data['dataset_info']['faithful_count'] = faithful_count
    contrastive_data['dataset_info']['unfaithful_count'] = unfaithful_count
    contrastive_data['dataset_info']['missing_activations'] = missing_activations
    
    # Convert lists to numpy arrays for easier computation
    if contrastive_data['faithful_activations']:
        contrastive_data['faithful_activations'] = np.stack(contrastive_data['faithful_activations'])
    if contrastive_data['unfaithful_activations']:
        contrastive_data['unfaithful_activations'] = np.stack(contrastive_data['unfaithful_activations'])
    
    # Save the contrastive dataset
    print(f"Saving contrastive dataset to {output_file}...")
    
    # Use pickle for numpy arrays, but also save a JSON summary
    with open(output_path, 'wb') as f:
        pickle.dump(contrastive_data, f)
    
    # Save JSON summary for inspection
    summary_file = output_path.with_suffix('.json')
    summary = {
        'dataset_info': contrastive_data['dataset_info'],
        'faithful_metadata': contrastive_data['faithful_metadata'],
        'unfaithful_metadata': contrastive_data['unfaithful_metadata']
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("-" * 60)
    print("Contrastive dataset building complete!")
    print(f"Total entries processed: {processed_count}")
    print(f"Faithful activations: {faithful_count}")
    print(f"Unfaithful activations: {unfaithful_count}")
    print(f"Missing activations: {missing_activations}")
    
    if faithful_count > 0 and unfaithful_count > 0:
        faithful_shape = contrastive_data['faithful_activations'].shape
        unfaithful_shape = contrastive_data['unfaithful_activations'].shape
        print(f"Faithful activations shape: {faithful_shape}")
        print(f"Unfaithful activations shape: {unfaithful_shape}")
        
        # Calculate mean activations for steering vector computation
        faithful_mean = np.mean(contrastive_data['faithful_activations'], axis=0)
        unfaithful_mean = np.mean(contrastive_data['unfaithful_activations'], axis=0)
        steering_vector = faithful_mean - unfaithful_mean
        
        print(f"Steering vector shape: {steering_vector.shape}")
        print(f"Steering vector norm: {np.linalg.norm(steering_vector):.4f}")
        
        # Save steering vector separately
        steering_file = output_path.with_name(output_path.stem + '_steering_vector.pkl')
        with open(steering_file, 'wb') as f:
            pickle.dump({
                'steering_vector': steering_vector,
                'faithful_mean': faithful_mean,
                'unfaithful_mean': unfaithful_mean,
                'faithful_count': faithful_count,
                'unfaithful_count': unfaithful_count,
                'creation_timestamp': datetime.now().isoformat()
            }, f)
        
        print(f"Steering vector saved to: {steering_file}")
    
    print(f"Contrastive dataset saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

def preview_contrastive_dataset(dataset_file: str) -> None:
    """Preview the contrastive dataset."""
    dataset_path = Path(dataset_file)
    summary_path = dataset_path.with_suffix('.json')
    
    if not summary_path.exists():
        print("Summary file not found.")
        return
    
    print("\n" + "=" * 60)
    print("CONTRASTIVE DATASET PREVIEW")
    print("=" * 60)
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    info = summary['dataset_info']
    print(f"Creation time: {info['creation_timestamp']}")
    print(f"Source files:")
    print(f"  - Faithful/unfaithful: {Path(info['source_faithful_unfaithful_file']).name}")
    print(f"  - Activations: {Path(info['source_activations_folder']).name}")
    print(f"Layer: {info['layer']}")
    print(f"Total processed: {info['total_entries_processed']}")
    print(f"Faithful entries: {info['faithful_count']}")
    print(f"Unfaithful entries: {info['unfaithful_count']}")
    print(f"Missing activations: {info['missing_activations']}")
    
    print(f"\nFaithful Examples:")
    for i, meta in enumerate(summary['faithful_metadata'][:3], 1):
        print(f"{i}. {meta['question']} (Shape: {meta['activation_shape']})")
        print(f"   {meta['original_letter']} -> {meta['biased_letter']} (Hint: {meta['hint_letter']})")
    
    print(f"\nUnfaithful Examples:")
    for i, meta in enumerate(summary['unfaithful_metadata'][:3], 1):
        print(f"{i}. {meta['question']} (Shape: {meta['activation_shape']})")
        print(f"   {meta['original_letter']} -> {meta['biased_letter']} (Hint: {meta['hint_letter']})")
    
    print("=" * 60)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input files
    FAITHFUL_UNFAITHFUL_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\faithful_unfaithful_test_2025-08-11.jsonl"
    ACTIVATIONS_FOLDER = r"C:\Users\l440\unfaithfulness_steering\extracted_activations_test_2025-08-11"
    
    # Output files
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\contrastive_activation_dataset.pkl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Contrastive Activation Dataset Builder")
    print("=" * 60)
    
    try:
        # Build contrastive dataset
        build_contrastive_dataset(FAITHFUL_UNFAITHFUL_FILE, ACTIVATIONS_FOLDER, OUTPUT_FILE)
        
        # Preview the results
        preview_contrastive_dataset(OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Contrastive dataset building completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")