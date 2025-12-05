import pickle
import torch

def inspect_vectors(pkl_path):
    print(f"Loading {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("File not found!")
        return

    print(f"Keys: {list(data.keys())}")
    
    vectors = data.get('steering_vectors', {})
    print(f"Number of vectors: {len(vectors)}")
    
    if vectors:
        first_layer = list(vectors.keys())[0]
        vec = vectors[first_layer]
        print(f"Layer {first_layer} vector shape: {vec.shape}")
        print(f"Layer {first_layer} vector norm: {vec.norm().item():.4f}")
        
    stats = data.get('computation_stats', {})
    print(f"Stats keys: {list(stats.keys())}")
    
    metadata = data.get('metadata', {})
    print(f"Metadata: {metadata}")

if __name__ == "__main__":
    inspect_vectors("results/activations_run2/steering_vectors.pkl")
