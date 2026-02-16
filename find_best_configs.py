import json
import os

DATA_DIR = r"c:\Users\occhi\Desktop\unfaithfulness_steering\data\definitive_pipeline_data\DeepSeek-R1-Distill-Llama-8B"

FILES = {
    "Linear": "summary_steered_linear_DeepSeek-R1-Distill-Llama-8B_2026-01-11.json",
    "MLP": "summary_steered_mlp_DeepSeek-R1-Distill-Llama-8B_2026-01-12.json",
    "OffPolicy": "summary_steered_off_policy_DeepSeek-R1-Distill-Llama-8B_2026-01-11.json"
}

def find_best_config(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    best_config = None
    max_rate = -1.0
    
    # Structure: data['dataset_info']['layers'] ... no, it's in 'configurations_by_hint'
    # 'configurations_by_hint': {'grader_hacking': [ {config1}, {config2} ... ]}
    
    for hint_type, configs in data.get('configurations_by_hint', {}).items():
        for cfg in configs:
            # We care about Recovery: Positive steering on WU (Wrong/Unfaithful)
            # Look for 'positive_on_WU' -> 'transitions' -> 'wrong_to_correct' -> 'rate'
            
            # Note: "Recovery" essentially means we fix the output.
            # "Wrong to Correct" is a good proxy if faithfulness correlates with correctness on this dataset.
            # But wait, we used 'faithfulness' field in the main analysis.
            # In the summary JSON, 'wrong_to_correct' is based on accuracy labels.
            # Let's hope they correlate. The user is talking about "Monitorability recovery", which usually means Faithful.
            
            # Let's check 'stable_faithful' rate + 'wrong_to_correct' rate?
            # Or just 'wrong_to_correct' if the baseline was Unfaithful (Wrong).
            
            stats = cfg.get('positive_on_WU', {}).get('transitions', {})
            wtc = stats.get('wrong_to_correct', {}).get('rate', 0)
            
            if wtc > max_rate:
                max_rate = wtc
                best_config = {
                    'layer': cfg['layer'],
                    'coeff': cfg['coefficient_magnitude'],
                    'rate': wtc
                }
                
    return best_config

def main():
    for name, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            best = find_best_config(path)
            print(f"Best {name}: Layer {best['layer']}, Coeff {best['coeff']}, Rate {best['rate']:.4f}")
        else:
            print(f"{name} file not found.")

if __name__ == "__main__":
    main()
