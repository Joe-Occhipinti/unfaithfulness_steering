
import json
import os

INPUT_JSONL = "data/sprint_5_2025-11-15/steered/steered_val_gradient_2025-12-01_shard_1.jsonl"
TEMP_JSONL = "data/sprint_5_2025-11-15/steered/steered_val_gradient_2025-12-01_shard_1_filtered.jsonl"

def filter_dataset():
    kept_count = 0
    removed_count = 0
    
    try:
        with open(INPUT_JSONL, 'r', encoding='utf-8') as fin, \
             open(TEMP_JSONL, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                record = json.loads(line)
                
                # Check if we should remove this record
                # Remove layer 28
                if record.get('steering_layer') == 28:
                    removed_count += 1
                else:
                    fout.write(json.dumps(record) + '\n')
                    kept_count += 1
                    
        print(f"Filtering complete.")
        print(f"Removed records: {removed_count}")
        print(f"Kept records: {kept_count}")
        
        # Overwrite original file
        os.replace(TEMP_JSONL, INPUT_JSONL)
        print(f"Overwrote original file: {INPUT_JSONL}")
        
    except FileNotFoundError:
        print(f"File not found: {INPUT_JSONL}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    filter_dataset()
