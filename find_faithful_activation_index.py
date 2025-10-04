"""
Find the index of the faithful prompt in the annotated dataset
and check if it has activations extracted.
"""

import os
from src.data import load_jsonl

ANNOTATED_FILE = "data/annotated/annotated_biased_high_school_macroeconomics_microeconomics_2025-10-03.jsonl"
ACTIVATIONS_DIR = "data/activations/biased_high_school_macroeconomics_microeconomics_2025-10-03.jsonl"

print(f"Loading {ANNOTATED_FILE}...")
annotated = load_jsonl(ANNOTATED_FILE)

# Find faithful prompt
faithful_idx = None
for i, r in enumerate(annotated):
    if r.get('faithfulness_classification') == 'faithful':
        faithful_idx = i
        print(f"\nFaithful prompt found at index {i} in annotated dataset")
        print(f"  Question: {r.get('question', '')[:80]}...")
        print(f"  hinted_answer_letter: {r.get('hinted_answer_letter')}")
        print(f"  bias_label: {r.get('bias_label')}")
        print(f"  Has [F] tags in annotated text: {'[F]' in r.get('annotated_biased_prompt', '')}")
        break

if faithful_idx is not None:
    # Check if activation file exists
    activation_file = os.path.join(ACTIVATIONS_DIR, f"prompt_{faithful_idx}_activations.pt")
    exists = os.path.exists(activation_file)

    print(f"\nExpected activation file: {activation_file}")
    print(f"File exists: {exists}")

    if exists:
        print("\nACTIVATIONS EXIST for this faithful prompt!")
        print("The [F] tagged activations should be in the dataset.")
    else:
        print("\nACTIVATIONS DO NOT EXIST for this faithful prompt!")
        print("This is why there are 0 [F] tags in the activation dataset.")
        print("\nPossible reasons:")
        print("  1. This prompt was added to annotated AFTER activations were extracted")
        print("  2. This prompt had None answer when activations were extracted")
        print("  3. Activation extraction failed for this specific prompt")
else:
    print("\nNo faithful prompt found in annotated dataset!")
