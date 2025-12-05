import pickle
import sys

def inspect_dataset(pkl_path):
    print(f"Loading {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("File not found!")
        return

    data = dataset['data']
    print(f"Total items: {len(data)}")
    
    if len(data) > 0:
        first_item = data[0]
        print(f"\nMetadata keys in first item: {list(first_item['metadata'].keys())}")
        # print(f"First item text (truncated): {first_item['text'][:100]}...")
        
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
    inspect_dataset("results/activations_run2/activations_dataset.pkl")
