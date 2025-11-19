"""
Temporary script to inspect activation dataset structure
Run this to see subjects in the dataset
"""
import pickle
import torch

# Load the dataset
INPUT_FILE = "data/sprint_4_2025-10-15/datasets/new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"

with open(INPUT_FILE, 'rb') as f:
    dataset = pickle.load(f)

# Inspect structure
print('=== DATASET STRUCTURE ===')
print('Top-level keys:', list(dataset.keys()))
print()

# Info section
if 'info' in dataset:
    print('=== INFO SECTION ===')
    for key, val in dataset['info'].items():
        if isinstance(val, (list, dict)) and len(str(val)) > 200:
            print(f'{key}: {type(val).__name__} (length={len(val)})')
        else:
            print(f'{key}: {val}')
    print()

# Sample prompt structure
if 'data' in dataset:
    print('=== DATA SECTION ===')
    print(f'Number of prompts: {len(dataset["data"])}')

    # Get first few prompts
    first_indices = list(dataset['data'].keys())[:3]

    for idx in first_indices:
        prompt = dataset['data'][idx]
        print(f'\n--- Prompt {idx} ---')
        print('Keys:', list(prompt.keys()))

        if 'metadata' in prompt:
            print('Metadata:', prompt['metadata'])

    # Collect all subjects
    subjects = set()
    hint_templates = set()
    correct_hints = set()

    for idx, prompt_data in dataset['data'].items():
        if 'metadata' in prompt_data:
            metadata = prompt_data['metadata']
            if 'subject' in metadata:
                subjects.add(metadata['subject'])
            if 'hint_template' in metadata:
                hint_templates.add(metadata['hint_template'])
            if 'correct_hint' in metadata:
                correct_hints.add(str(metadata['correct_hint']))

    print(f'\n=== ALL SUBJECTS FOUND ({len(subjects)}) ===')
    for subj in sorted(subjects):
        print(f'  - {subj}')

    print(f'\n=== HINT TEMPLATES ({len(hint_templates)}) ===')
    for ht in sorted(hint_templates):
        print(f'  - {ht}')

    print(f'\n=== CORRECT HINT VALUES ===')
    for ch in sorted(correct_hints):
        print(f'  - {ch}')

    # Count prompts per subject
    subject_counts = {}
    for idx, prompt_data in dataset['data'].items():
        if 'metadata' in prompt_data and 'subject' in prompt_data['metadata']:
            subj = prompt_data['metadata']['subject']
            subject_counts[subj] = subject_counts.get(subj, 0) + 1

    print(f'\n=== PROMPTS PER SUBJECT ===')
    for subj in sorted(subject_counts.keys()):
        print(f'  {subj}: {subject_counts[subj]} prompts')

    # Count by configuration (hint_template x correct_hint)
    config_counts = {}
    for idx, prompt_data in dataset['data'].items():
        if 'metadata' in prompt_data:
            metadata = prompt_data['metadata']
            ht = metadata.get('hint_template', 'unknown')
            ch = str(metadata.get('correct_hint', 'unknown'))
            config = f"{ht} | correct_hint={ch}"
            config_counts[config] = config_counts.get(config, 0) + 1

    print(f'\n=== PROMPTS PER CONFIGURATION ===')
    for config in sorted(config_counts.keys()):
        print(f'  {config}: {config_counts[config]} prompts')
