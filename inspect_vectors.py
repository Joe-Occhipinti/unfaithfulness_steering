"""
Quick script to inspect the structure of vector pickle files
"""
import pickle

files = [
    "data/sprint_5_2025-11-15/vectors/vectors_hints-weighting_cie_hist_psy_X_grader_prof_meta_2025-11-15.pkl",
    "data/sprint_5_2025-11-15/vectors/vectors_domains-weighting_cie_hist_psy_X_grader_prof_meta_2025-11-15.pkl"
]

for file_path in files:
    print(f"\n{'='*80}")
    print(f"FILE: {file_path}")
    print('='*80)

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Top-level keys: {list(data.keys())}")

        if 'steering_vectors' in data:
            sv = data['steering_vectors']
            print(f"steering_vectors type: {type(sv)}")
            print(f"steering_vectors length: {len(sv) if hasattr(sv, '__len__') else 'N/A'}")
            if hasattr(sv, 'keys'):
                print(f"steering_vectors keys: {list(sv.keys())[:10]}")  # First 10

        if 'computation_stats' in data:
            cs = data['computation_stats']
            print(f"\ncomputation_stats keys: {list(cs.keys())}")
            print(f"  config_fields: {cs.get('config_fields')}")
            print(f"  total_configs: {cs.get('total_configs')}")
            print(f"  layers_computed: {cs.get('layers_computed', [])[:10]}")  # First 10

    except Exception as e:
        print(f"ERROR loading file: {e}")
