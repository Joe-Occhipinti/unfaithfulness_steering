
import json
from collections import defaultdict

INPUT_FILE = "data/sprint_4_2025-10-15/annotated/touse_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"
TARGET_TEMPLATES = ['grader_hacking', 'metadata', 'professor']

def check_counts():
    print(f"Checking counts in {INPUT_FILE}...")
    
    # Structure: counts[template][letter] = count
    counts = defaultdict(lambda: defaultdict(int))
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                
                # Filter: biased prompts only
                gt = record.get('ground_truth_letter')
                hint = record.get('hint_letter')
                template = record.get('hint_template')
                
                if gt and hint and gt != hint and template in TARGET_TEMPLATES:
                    counts[template][hint] += 1
                    
            except json.JSONDecodeError:
                continue

    print("\n--- Available Biased Samples (GT != Hint) ---")
    print(f"{'Template':<20} | {'A':<5} {'B':<5} {'C':<5} {'D':<5} | {'Total':<5}")
    print("-" * 60)
    
    for template in TARGET_TEMPLATES:
        c = counts[template]
        total = sum(c.values())
        print(f"{template:<20} | {c.get('A',0):<5} {c.get('B',0):<5} {c.get('C',0):<5} {c.get('D',0):<5} | {total:<5}")

    print("\nTarget per letter (for 54 total): ~13-14")

if __name__ == "__main__":
    check_counts()
