import json
import argparse
from collections import Counter

def analyze_hinted_accuracy(file_path):
    total_filtered_records = 0
    correct_count = 0
    
    subject_stats = {} # subject -> {'total': 0, 'correct': 0}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    ground_truth = data.get('ground_truth_letter')
                    hint = data.get('hint_letter')
                    hinted_answer = data.get('hinted_answer_letter')
                    subject = data.get('subject', 'unknown')
                    
                    # Filter: Only consider records where ground_truth != hint
                    if ground_truth and hint and ground_truth != hint:
                        total_filtered_records += 1
                        
                        if subject not in subject_stats:
                            subject_stats[subject] = {'total': 0, 'correct': 0}
                        subject_stats[subject]['total'] += 1
                        
                        # Check accuracy: hinted_answer == ground_truth
                        if hinted_answer == ground_truth:
                            correct_count += 1
                            subject_stats[subject]['correct'] += 1
                            
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    output_file = "hinted_analysis_results.txt"
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("-" * 60 + "\n")
        out.write(f"Analysis for: {file_path}\n")
        out.write(f"Filter Condition: ground_truth_letter != hint_letter\n")
        out.write("-" * 60 + "\n")
        out.write(f"{'Subject':<35} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}\n")
        out.write("-" * 60 + "\n")
        
        for subject in sorted(subject_stats.keys()):
            stats = subject_stats[subject]
            total = stats['total']
            correct = stats['correct']
            accuracy = (correct / total) if total > 0 else 0
            out.write(f"{subject:<35} | {total:<8} | {correct:<8} | {accuracy:.2%}\n")
            
        out.write("-" * 60 + "\n")
        if total_filtered_records > 0:
            out.write(f"{'TOTAL':<35} | {total_filtered_records:<8} | {correct_count:<8} | {(correct_count / total_filtered_records):.2%}\n")
        else:
            out.write("TOTAL: 0 records\n")
        out.write("-" * 60 + "\n")
        
    print(f"Analysis complete. Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Hinted Dataset Accuracy.")
    parser.add_argument("file_path", help="Path to the JSONL file")
    args = parser.parse_args()
    
    analyze_hinted_accuracy(args.file_path)
