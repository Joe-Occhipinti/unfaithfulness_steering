"""
Check the actual data types of correct_hint in the dataset
"""
import pickle

INPUT_FILE = "data/sprint_4_2025-10-15/datasets/new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"

with open(INPUT_FILE, 'rb') as f:
    dataset = pickle.load(f)

# Sample a few prompts to check correct_hint type
print("Checking correct_hint values in dataset:")
count = 0
for idx, prompt_data in dataset['data'].items():
    if count >= 10:
        break

    metadata = prompt_data.get('metadata', {})
    correct_hint = metadata.get('correct_hint')

    print(f"Prompt {idx}: correct_hint = {correct_hint!r} (type: {type(correct_hint).__name__})")
    count += 1

# Count all unique correct_hint values
unique_values = set()
for idx, prompt_data in dataset['data'].items():
    metadata = prompt_data.get('metadata', {})
    correct_hint = metadata.get('correct_hint')
    unique_values.add((correct_hint, type(correct_hint).__name__))

print(f"\nAll unique correct_hint values:")
for val, typ in sorted(unique_values):
    print(f"  {val!r} ({typ})")
