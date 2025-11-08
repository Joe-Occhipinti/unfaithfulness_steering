import json
from collections import defaultdict

# Load the dataset
file_path = r"c:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint_3_2025-09-15\annotated_biased_csXcons_scieXgrad_physXarg_psyXprof_histXmeta_econXsquare.jsonl"

# Storage for analysis
subjects = set()
hint_templates = set()
baseline_match_subjects = defaultdict(lambda: {"match": 0, "no_match": 0})
overall_stats = {"match": 0, "no_match": 0}

# Read and analyze
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)

        # 1. Collect subject values
        subject = record.get('subject')
        subjects.add(subject)

        # 2. Collect hint_template values
        hint_template = record.get('hint_template')
        hint_templates.add(hint_template)

        # 3. Check baseline_answer_letter vs ground_truth_letter
        baseline_answer = record.get('baseline_answer_letter')
        ground_truth = record.get('ground_truth_letter')

        if baseline_answer == ground_truth:
            overall_stats["match"] += 1
            baseline_match_subjects[subject]["match"] += 1
        else:
            overall_stats["no_match"] += 1
            baseline_match_subjects[subject]["no_match"] += 1

# Print results
print("=" * 80)
print("1. SUBJECT VALUES:")
print("=" * 80)
for subject in sorted(subjects):
    print(f"  - {subject}")

print("\n" + "=" * 80)
print("2. HINT_TEMPLATE VALUES:")
print("=" * 80)
for template in sorted(hint_templates):
    print(f"  - {template}")

print("\n" + "=" * 80)
print("3. BASELINE ANSWER vs GROUND TRUTH:")
print("=" * 80)
print(f"Total records with baseline_answer_letter == ground_truth_letter: {overall_stats['match']}")
print(f"Total records with baseline_answer_letter != ground_truth_letter: {overall_stats['no_match']}")

print("\n" + "=" * 80)
print("4. BASELINE MATCH/NO-MATCH BY SUBJECT:")
print("=" * 80)
for subject in sorted(baseline_match_subjects.keys()):
    match = baseline_match_subjects[subject]["match"]
    no_match = baseline_match_subjects[subject]["no_match"]
    total = match + no_match

    has_both = "[HAS BOTH]" if match > 0 and no_match > 0 else "[ONLY ONE TYPE]"

    print(f"\n{subject}:")
    print(f"  Match (baseline == ground_truth):     {match:4d} ({match/total*100:5.1f}%)")
    print(f"  No-Match (baseline != ground_truth):  {no_match:4d} ({no_match/total*100:5.1f}%)")
    print(f"  Total:                                 {total:4d}")
    print(f"  Status: {has_both}")

print("\n" + "=" * 80)
