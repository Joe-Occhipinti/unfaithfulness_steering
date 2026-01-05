
import json

file_path = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\definitive_pipeline_data\Qwen3-32B\faithfulness_annotated_Qwen3-32B_2025-12-29.jsonl"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        if line:
            data = json.loads(line)
            print("Keys in first record:")
            print(list(data.keys()))
            
            # Check for potential prompt keys
            candidates = ['prompt', 'input', 'question', 'text', 'content']
            found = [k for k in data.keys() if any(c in k for c in candidates)]
            print(f"\nCandidate keys found: {found}")
            
            for k in found:
                val = data[k]
                if isinstance(val, str):
                    print(f"  {k}: {val[:50]}...")
        else:
            print("File is empty.")
except Exception as e:
    print(f"Error: {e}")
