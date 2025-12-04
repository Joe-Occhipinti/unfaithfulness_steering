
from generate_off_policy_data import load_and_filter_data, sample_balanced_data, INPUT_FILE

def verify():
    print("--- Verifying Sampling Logic ---")
    records = load_and_filter_data(INPUT_FILE)
    sampled = sample_balanced_data(records)
    
    print("\n--- Verification Results ---")
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    
    for r in sampled:
        template = r['hint_template']
        letter = r['hint_letter']
        counts[template][letter] += 1
        
    print(f"{'Template':<20} | {'A':<5} {'B':<5} {'C':<5} {'D':<5} | {'Total':<5}")
    print("-" * 60)
    
    for template in sorted(counts.keys()):
        c = counts[template]
        total = sum(c.values())
        print(f"{template:<20} | {c.get('A',0):<5} {c.get('B',0):<5} {c.get('C',0):<5} {c.get('D',0):<5} | {total:<5}")

if __name__ == "__main__":
    verify()
