import json
import os

input_file = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_6_2025-12-15\off_policy_responses.jsonl"
output_file = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_6_2025-12-15\off_policy_responses_filtered.jsonl"

fields_to_keep = ["biased_input_prompt", "ground_truth_letter", "hint_letter", "off_policy_response"]

print(f"Processing {input_file}...")
count = 0
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            filtered_data = {key: data.get(key) for key in fields_to_keep if key in data}
            outfile.write(json.dumps(filtered_data) + '\n')
            count += 1
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line[:50]}... Error: {e}")

print(f"Finished processing. Wrote {count} records to {output_file}")
