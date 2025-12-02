
import json
from collections import Counter
import os

INPUT_JSONL = "data/sprint_5_2025-11-15/steered/steered_val_gradient_2025-12-01_shard_1.jsonl"

def inspect_dataset():
    target_values = []
    directions = []
    layers = []
    
    try:
        if not os.path.exists(INPUT_JSONL):
            print(f"File not found: {INPUT_JSONL}")
            return

        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if 'steering_target_value' in record:
                    target_values.append(record['steering_target_value'])
                if 'steering_direction' in record:
                    directions.append(record['steering_direction'])
                if 'steering_layer' in record:
                    layers.append(record['steering_layer'])
                    
        print(f"Total records: {len(target_values)}")
        print(f"Unique Target Values: {sorted(list(set(target_values)))}")
        print(f"Target Value Counts: {Counter(target_values)}")
        print(f"Unique Layers: {sorted(list(set(layers)))}")
        print(f"Layer Counts: {Counter(layers)}")
        print(f"Unique Directions: {sorted(list(set(directions)))}")
        print(f"Direction Counts: {Counter(directions)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_dataset()
