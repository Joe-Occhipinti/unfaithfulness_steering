
import json

file_path = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\definitive_pipeline_data\Qwen3-32B\steered_linear_Qwen3-32B_2026-01-04.jsonl"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        if line:
            data = json.loads(line)
            print("Keys in first record:")
            print(list(data.keys()))
            
            # Check for 'biased_input_prompt' or similar
            if 'biased_input_prompt' in data:
                print(f"\n'biased_input_prompt' found. Length: {len(data['biased_input_prompt'])}")
            else:
                print("\n'biased_input_prompt' NOT found.")
                # Look for potential candidates
                for k in data.keys():
                    if 'prompt' in k or 'input' in k:
                        print(f"Candidate key: {k}")
        else:
            print("File is empty.")
except Exception as e:
    print(f"Error: {e}")
