import pickle
import sys

def check_labels(pkl_path):
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    
    data = dataset['data']
    print(f"Total items: {len(data)}")
    
    print("\nFirst 10 labels:")
    for i in range(10):
        if i in data:
            print(f"  {i}: {data[i]['metadata'].get('label')}")
            
    print("\nLabel Counts:")
    counts = {}
    for idx, item in data.items():
        label = item['metadata'].get('label')
        counts[label] = counts.get(label, 0) + 1
    print(counts)

if __name__ == "__main__":
    check_labels("results/activations_run1/activations_dataset.pkl")
