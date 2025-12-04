import json
import argparse
from collections import Counter

def analyze_dataset(file_path):
    print(f"Analyzing file: {file_path}")
    
    total_records = 0
    correct_count = 0
    subjects = set()
    subject_counts = Counter()
    
    subject_correct_counts = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    total_records += 1
                    
                    subject = data.get('subject')
                    if subject:
                        subjects.add(subject)
                        subject_counts[subject] += 1
                        
                        # Count correct answers per subject
                        if data.get('accuracy_label') == 'correct':
                            correct_count += 1
                            subject_correct_counts[subject] += 1
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    output_file = "analysis_results.txt"
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("-" * 60 + "\n")
        out.write(f"{'Subject':<35} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}\n")
        out.write("-" * 60 + "\n")
        
        for subject in sorted(subjects):
            total = subject_counts[subject]
            correct = subject_correct_counts[subject]
            accuracy = (correct / total) if total > 0 else 0
            out.write(f"{subject:<35} | {total:<8} | {correct:<8} | {accuracy:.2%}\n")
            
        out.write("-" * 60 + "\n")
        out.write(f"{'TOTAL':<35} | {total_records:<8} | {correct_count:<8} | {(correct_count / total_records):.2%}\n")
        out.write("-" * 60 + "\n")
        
    print(f"Analysis complete. Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze JSONL dataset.")
    parser.add_argument("file_path", help="Path to the JSONL file")
    args = parser.parse_args()
    
    analyze_dataset(args.file_path)
