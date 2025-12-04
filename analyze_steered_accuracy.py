import json
from collections import defaultdict

def analyze_steered_accuracy(file_path):
    # Data structures for aggregation
    # Specific: Layer 8, Coeff +-0.75
    # Structure: hint_template -> direction -> {'total': 0, 'correct': 0}
    specific_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))
    
    # Total: All layers, All coeffs
    # Structure: hint_template -> direction -> {'total': 0, 'correct': 0}
    total_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

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
                    steered_accuracy = data.get('steered_accuracy')
                    hint_template = data.get('hint_template', 'unknown')
                    steering_layer = data.get('steering_layer')
                    steering_coefficient = data.get('steering_coefficient')
                    
                    # Filter 1: ground_truth != hint
                    if ground_truth and hint and ground_truth != hint:
                        
                        # Determine direction
                        if steering_coefficient is not None:
                            try:
                                coeff_val = float(steering_coefficient)
                                if coeff_val > 0:
                                    direction = "Positive (+)"
                                elif coeff_val < 0:
                                    direction = "Negative (-)"
                                else:
                                    direction = "Neutral (0)" # Should not happen based on request but good to handle
                            except ValueError:
                                continue
                        else:
                            continue

                        is_correct = (steered_accuracy == 'correct')

                        # Update Total Stats (All layers, all coeffs)
                        total_stats[hint_template][direction]['total'] += 1
                        if is_correct:
                            total_stats[hint_template][direction]['correct'] += 1

                        # Update Specific Stats (Layer 8, Coeff +-0.75)
                        # Use a small epsilon for float comparison if needed, or just check equality if values are clean
                        if steering_layer == 8 and abs(coeff_val) == 0.75:
                             specific_stats[hint_template][direction]['total'] += 1
                             if is_correct:
                                 specific_stats[hint_template][direction]['correct'] += 1

                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Output Results
    output = []
    
    output.append("=== Specific Analysis (Layer 8, Coeff +/- 0.75) ===\n")
    output.append(f"{'Hint Template':<30} | {'Direction':<15} | {'Total':<10} | {'Correct':<10} | {'Accuracy':<10}")
    output.append("-" * 85)
    
    for template in sorted(specific_stats.keys()):
        for direction in sorted(specific_stats[template].keys()):
            stats = specific_stats[template][direction]
            total = stats['total']
            correct = stats['correct']
            accuracy = (correct / total * 100) if total > 0 else 0.0
            output.append(f"{template:<30} | {direction:<15} | {total:<10} | {correct:<10} | {accuracy:.2f}%")
    
    output.append("\n" + "="*85 + "\n")
    
    output.append("=== Total Analysis (All Layers, All Coefficients) ===\n")
    output.append(f"{'Hint Template':<30} | {'Direction':<15} | {'Total':<10} | {'Correct':<10} | {'Accuracy':<10}")
    output.append("-" * 85)
    
    for template in sorted(total_stats.keys()):
        for direction in sorted(total_stats[template].keys()):
            stats = total_stats[template][direction]
            total = stats['total']
            correct = stats['correct']
            accuracy = (correct / total * 100) if total > 0 else 0.0
            output.append(f"{template:<30} | {direction:<15} | {total:<10} | {correct:<10} | {accuracy:.2f}%")

    # Write to file
    with open('steered_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print("Analysis complete. Results written to steered_analysis_results.txt")
    # Also print to stdout for immediate view if small enough, but file is safer
    print('\n'.join(output))

if __name__ == "__main__":
    file_path = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_4_2025-10-15\annotated\steered\annotated_steered_sprint4_2025-10-27.jsonl"
    analyze_steered_accuracy(file_path)
