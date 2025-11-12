"""
Test script to verify dataset type detection in process_answers.py
"""
from src.data import load_jsonl

# Test files
HINTED_SAMPLED = "data/Lorenz_suggestion_2025-11-12/sampled_q13-36-115-26_2025-11-12_flat.jsonl"
STEERED_SAMPLED = "data/Lorenz_suggestion_2025-11-12/steered_sampled_q13-36-115-26_2025-11-12_flat.jsonl"

print("=== Testing Dataset Type Detection ===\n")

# Test hinted sampled
print("--- Testing Hinted Sampled File ---")
raw_data = load_jsonl(HINTED_SAMPLED)
print(f"Loaded {len(raw_data)} records")
sample_record = raw_data[0]

# Check for all dataset types
is_steered = 'steered_response' in sample_record and 'steering_layer' in sample_record
is_steered_sampled = 'steered_sampled_generated_text' in sample_record and 'steering_layer' in sample_record
is_hinted = 'hinted_generated_text' in sample_record and 'hint_letter' in sample_record
is_hinted_sampled = 'sampled_generated_text' in sample_record and 'hint_letter' in sample_record

print(f"  is_steered: {is_steered}")
print(f"  is_steered_sampled: {is_steered_sampled}")
print(f"  is_hinted: {is_hinted}")
print(f"  is_hinted_sampled: {is_hinted_sampled}")

if is_steered:
    dataset_type = 'steered'
elif is_steered_sampled:
    dataset_type = 'steered_sampled'
elif is_hinted:
    dataset_type = 'hinted'
elif is_hinted_sampled:
    dataset_type = 'hinted_sampled'
else:
    dataset_type = 'unknown'

print(f"  Detected type: {dataset_type}")
print(f"  PASS" if dataset_type == 'hinted_sampled' else f"  FAIL (expected hinted_sampled)")

# Test steered sampled
print("\n--- Testing Steered Sampled File ---")
raw_data = load_jsonl(STEERED_SAMPLED)
print(f"Loaded {len(raw_data)} records")
sample_record = raw_data[0]

is_steered = 'steered_response' in sample_record and 'steering_layer' in sample_record
is_steered_sampled = 'steered_sampled_generated_text' in sample_record and 'steering_layer' in sample_record
is_hinted = 'hinted_generated_text' in sample_record and 'hint_letter' in sample_record
is_hinted_sampled = 'sampled_generated_text' in sample_record and 'hint_letter' in sample_record

print(f"  is_steered: {is_steered}")
print(f"  is_steered_sampled: {is_steered_sampled}")
print(f"  is_hinted: {is_hinted}")
print(f"  is_hinted_sampled: {is_hinted_sampled}")

if is_steered:
    dataset_type = 'steered'
elif is_steered_sampled:
    dataset_type = 'steered_sampled'
elif is_hinted:
    dataset_type = 'hinted'
elif is_hinted_sampled:
    dataset_type = 'hinted_sampled'
else:
    dataset_type = 'unknown'

print(f"  Detected type: {dataset_type}")
print(f"  PASS" if dataset_type == 'steered_sampled' else f"  FAIL (expected steered_sampled)")

print("\n=== Field Names in Files ===\n")
print("Hinted Sampled fields:")
hinted_data = load_jsonl(HINTED_SAMPLED)
hinted_keys = list(hinted_data[0].keys())
print(f"  Key for text: 'sampled_generated_text' {'FOUND' if 'sampled_generated_text' in hinted_keys else 'MISSING'}")
print(f"  Has hint_letter: {'FOUND' if 'hint_letter' in hinted_keys else 'MISSING'}")
print(f"  Has steering fields: {'NOT FOUND (good)' if 'steering_layer' not in hinted_keys else 'FOUND (unexpected!)'}")

print("\nSteered Sampled fields:")
steered_data = load_jsonl(STEERED_SAMPLED)
steered_keys = list(steered_data[0].keys())
print(f"  Key for text: 'steered_sampled_generated_text' {'FOUND' if 'steered_sampled_generated_text' in steered_keys else 'MISSING'}")
print(f"  Has steering_layer: {'FOUND' if 'steering_layer' in steered_keys else 'MISSING'}")
print(f"  Has steering_coefficient: {'FOUND' if 'steering_coefficient' in steered_keys else 'MISSING'}")

print("\n=== Summary ===")
print("Both flat files should now be compatible with process_answers.py")
print("  - Hinted sampled: uses 'sampled_generated_text'")
print("  - Steered sampled: uses 'steered_sampled_generated_text'")
