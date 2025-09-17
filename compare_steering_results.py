#!/usr/bin/env python3
"""Compare positive vs negative steering results to see if they're identical"""

import json
import os

def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compare_files(pos_file, neg_file):
    pos_data = load_jsonl(pos_file)
    neg_data = load_jsonl(neg_file)

    print(f"\nComparing {os.path.basename(pos_file)} vs {os.path.basename(neg_file)}")
    print(f"Pos file: {len(pos_data)} items")
    print(f"Neg file: {len(neg_data)} items")

    if len(pos_data) != len(neg_data):
        print("ERROR: Different number of items!")
        return

    identical_count = 0
    different_count = 0

    for i, (pos_item, neg_item) in enumerate(zip(pos_data, neg_data)):
        pos_response = pos_item.get('steered_prompt_complete', '')
        neg_response = neg_item.get('steered_prompt_complete', '')

        if pos_response == neg_response:
            identical_count += 1
            if identical_count <= 3:  # Show first few identical cases
                print(f"\nIDENTICAL #{identical_count} (item {i}):")
                print(f"Pos coeff: {pos_item.get('coefficient', 'N/A')}")
                print(f"Neg coeff: {neg_item.get('coefficient', 'N/A')}")
                print(f"Response: {pos_response[:200]}...")
        else:
            different_count += 1
            if different_count <= 3:  # Show first few different cases
                print(f"\nDIFFERENT #{different_count} (item {i}):")
                print(f"Pos ({pos_item.get('coefficient', 'N/A')}): {pos_response[:150]}...")
                print(f"Neg ({neg_item.get('coefficient', 'N/A')}): {neg_response[:150]}...")

    print(f"\nSUMMARY:")
    print(f"Identical responses: {identical_count}/{len(pos_data)} ({identical_count/len(pos_data)*100:.1f}%)")
    print(f"Different responses: {different_count}/{len(pos_data)} ({different_count/len(pos_data)*100:.1f}%)")

# Test a few layer combinations
base_dir = "./caos of steering/body_final/"
test_layers = [10, 15, 20, 25]

for layer in test_layers:
    pos_file = os.path.join(base_dir, f"steered_layer_{layer}_coeff_pos5.0.jsonl")
    neg_file = os.path.join(base_dir, f"steered_layer_{layer}_coeff_neg5.0.jsonl")

    if os.path.exists(pos_file) and os.path.exists(neg_file):
        compare_files(pos_file, neg_file)
    else:
        print(f"\nSkipping layer {layer} - files not found")

print("\n=== DONE ===")