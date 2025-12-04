import json
import argparse
from collections import defaultdict

def analyze_stratified_accuracy(file_path):
    # Define subject mappings
    subject_mapping = {
        'high_school_european_history': 'History',
        'high_school_us_history': 'History',
        'high_school_world_history': 'History',
        'prehistory': 'History',
        'high_school_biology': 'Science',
        'high_school_chemistry': 'Science',
        'college_biology': 'Science',
        'college_chemistry': 'Science',
        'high_school_psychology': 'Psychology'
    }

    # Storage: Domain -> Template -> {'total': 0, 'correct': 0}
    stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))
    
    total_filtered = 0
    total_correct = 0

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
                    subject = data.get('subject')
                    hint_template = data.get('hint_template', 'unknown')
                    
                    # Filter: Only consider records where ground_truth != hint
                    if ground_truth and hint and ground_truth != hint:
                        total_filtered += 1
                        
                        # Map subject to domain
                        domain = subject_mapping.get(subject, 'Other')
                        
                        stats[domain][hint_template]['total'] += 1
                        
                        if hinted_answer == ground_truth:
                            stats[domain][hint_template]['correct'] += 1
                            total_correct += 1
                            
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Aggregate by Hint Template
    template_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for domain in stats:
        for template in stats[domain]:
            template_stats[template]['total'] += stats[domain][template]['total']
            template_stats[template]['correct'] += stats[domain][template]['correct']

    output_file = "stratified_analysis_results.txt"
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("-" * 80 + "\n")
        out.write(f"Stratified Analysis (Domain x Hint Template)\n")
        out.write(f"Filter: ground_truth_letter != hint_letter\n")
        out.write("-" * 80 + "\n")
        out.write(f"{'Domain':<15} | {'Hint Template':<20} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}\n")
        out.write("-" * 80 + "\n")
        
        for domain in sorted(stats.keys()):
            domain_stats = stats[domain]
            for template in sorted(domain_stats.keys()):
                s = domain_stats[template]
                total = s['total']
                correct = s['correct']
                accuracy = (correct / total) if total > 0 else 0
                out.write(f"{domain:<15} | {template:<20} | {total:<8} | {correct:<8} | {accuracy:.2%}\n")
            out.write("-" * 80 + "\n") # Separator between domains

        out.write("\n")
        out.write("-" * 80 + "\n")
        out.write(f"Aggregated by Hint Template\n")
        out.write("-" * 80 + "\n")
        out.write(f"{'Hint Template':<20} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}\n")
        out.write("-" * 80 + "\n")
        
        for template in sorted(template_stats.keys()):
            s = template_stats[template]
            total = s['total']
            correct = s['correct']
            accuracy = (correct / total) if total > 0 else 0
            out.write(f"{template:<20} | {total:<8} | {correct:<8} | {accuracy:.2%}\n")
            
        out.write("-" * 80 + "\n")
        out.write(f"{'OVERALL':<38} | {total_filtered:<8} | {total_correct:<8} | {(total_correct/total_filtered):.2%}\n")
        out.write("-" * 80 + "\n")

    print(f"Analysis complete. Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Stratified Accuracy.")
    parser.add_argument("file_path", help="Path to the JSONL file")
    args = parser.parse_args()
    
    analyze_stratified_accuracy(args.file_path)
