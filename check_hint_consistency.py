"""
Quick script to check hint_template consistency across sampled and steered datasets.
"""

from src.data import load_jsonl

# Load datasets
sampled = load_jsonl('data/Lorenz_suggestion_2025-11-12/sampled_q13-36-115-26_2025-11-12_flat_validated.jsonl')
steered = load_jsonl('data/Lorenz_suggestion_2025-11-12/steered_sampled_q13-36-115-26_2025-11-12_flat_validated.jsonl')

print("Checking hint_template consistency across questions...\n")

for qid in [13, 36, 115, 26]:
    sampled_qid = [r for r in sampled if r.get('question_id') == qid]
    steered_qid = [r for r in steered if r.get('question_id') == qid]

    sampled_template = set(r.get('hint_template') for r in sampled_qid)
    steered_template = set(r.get('hint_template') for r in steered_qid)

    match = "MATCH" if sampled_template == steered_template else "MISMATCH"

    print(f"Q{qid}: {match}")
    print(f"  Sampled (pre-steering):  {sampled_template}")
    print(f"  Steered (post-steering): {steered_template}")
    print(f"  Sampled records: {len(sampled_qid)}")
    print(f"  Steered records: {len(steered_qid)}")
    print()
