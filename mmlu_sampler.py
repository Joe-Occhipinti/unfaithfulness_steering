import json
import random
from datasets import load_dataset
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

def load_and_sample_mmlu(splits: Optional[List[str]] = None, subjects: Optional[List[str]] = None, 
                        k: Optional[int] = None, random_seed: Optional[int] = None, 
                        output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load MMLU dataset, optionally shuffle within subjects, and sample instances.
    
    Args:
        splits (Optional[List[str]]): Dataset splits to use ('auxiliary_train', 'validation', 'test', 'dev').
                                    If None, loads from all available splits.
        subjects (Optional[List[str]]): Specific subjects to filter. If None, includes all subjects.
        k (Optional[int]): Number of instances to sample from each subject. If None, loads all instances.
        random_seed (Optional[int]): Random seed for reproducible shuffling. If None, no shuffling is applied.
        output_file (Optional[str]): Output JSONL file path. If None, uses default naming pattern.
    
    Returns:
        List[Dict[str, Any]]: List of sampled instances
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Default splits if not provided
    if splits is None:
        splits = ['train', 'validation', 'test', 'dev']
    
    all_instances = []
    
    # Load the MMLU dataset from specified splits
    for split in splits:
        print(f"Loading MMLU dataset split: {split}")
        try:
            ds = load_dataset("cais/mmlu", "all", split=split)
        except Exception as e:
            print(f"Error loading dataset for split '{split}': {e}")
            continue
        
        # Add split information to each instance
        for instance in ds:
            instance['split'] = split
            all_instances.append(instance)
    
    if not all_instances:
        print("No data loaded from any split")
        return []
    
    # Group instances by subject
    subject_groups = defaultdict(list)
    
    print("Grouping instances by subject...")
    for instance in all_instances:
        if 'subject' not in instance:
            print(f"Warning: 'subject' key missing in instance: {instance}")
            continue
        subject = instance['subject']
        
        # Filter by subjects if specified
        if subjects is not None and subject not in subjects:
            continue
            
        subject_groups[subject].append(instance)
    
    print(f"Found {len(subject_groups)} subjects")
    for subject, instances in subject_groups.items():
        print(f"  {subject}: {len(instances)} instances")
    
    # Process instances from each subject
    sampled_instances = []
    
    if k is not None:
        print(f"\nSampling top-{k} from each subject...")
        action_desc = "sampled"
    else:
        print(f"\nLoading all instances from each subject...")
        action_desc = "loaded"
    
    for subject, instances in subject_groups.items():
        processed_instances = instances.copy()
        
        # Shuffle instances within this subject group if random_seed is provided
        if random_seed is not None:
            random.shuffle(processed_instances)
        
        # Select instances (top-k if k is specified, otherwise all)
        if k is not None:
            selected_instances = processed_instances[:k]
        else:
            selected_instances = processed_instances
            
        sampled_instances.extend(selected_instances)
        
        print(f"  {subject}: {action_desc} {len(selected_instances)} instances")
    
    # Generate output filename if not provided
    if output_file is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Build filename components
        splits_str = "_".join(splits) if len(splits) != 4 else "all_splits"
        subjects_str = "_".join(subjects) if subjects is not None else "all_subjects"
        k_str = f"_k{k}" if k is not None else "_all"
        seed_str = f"_seed{random_seed}" if random_seed is not None else ""
        
        filename = f"mmlu_{splits_str}_{subjects_str}{k_str}{seed_str}_{date_str}.jsonl"
        output_file = os.path.join("datasets", filename)
    else:
        # Use provided output file path but ensure it's in datasets directory if relative
        if not os.path.isabs(output_file):
            output_file = os.path.join("datasets", output_file)
    
    # Create datasets directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSONL file
    print(f"\nSaving {len(sampled_instances)} instances to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for instance in sampled_instances:
            f.write(json.dumps(instance) + '\n')
    
    print(f"Successfully saved sampled dataset to {output_file}")
    return sampled_instances


def main():
    """Example usage of the function"""
    
    # Example 1: Load specific splits and subjects, sample top 5 from each subject
    instances = load_and_sample_mmlu(
        splits=["validation"],
        subjects=["high_school_psychology"],
    )
    
    # Example 2: Load all data from specific subjects without sampling
    # instances = load_and_sample_mmlu(
    #     subjects=["computer_science", "mathematics"],
    # )
    
    # Example 3: Load all data from all subjects and splits (no filtering)
    # instances = load_and_sample_mmlu()

if __name__ == "__main__":
    main()