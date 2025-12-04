import json
from collections import defaultdict

def analyze_gradient_steered_accuracy(file_path):
    # Data structures
    # 1. Total (All layers, All targets) - Stratified by Direction
    total_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # 2. Total Stratified (All layers, All targets) - Stratified by Template & Direction
    total_stratified_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))
    
    # 3. Specific (Layer 15, Target 1) - Stratified by Template & Direction
    specific_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

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
                    steering_target_value = data.get('steering_target_value')
                    steering_direction = data.get('steering_direction', 'unknown')
                    
                    # Filter: ground_truth != hint
                    if ground_truth and hint and ground_truth != hint:
                        
                        is_correct = (steered_accuracy == 'correct')

                        # 1. Update Total Stats (by Direction)
                        total_stats[steering_direction]['total'] += 1
                        if is_correct:
                            total_stats[steering_direction]['correct'] += 1

                        # 2. Update Total Stratified Stats (by Template & Direction)
                        total_stratified_stats[hint_template][steering_direction]['total'] += 1
                        if is_correct:
                            total_stratified_stats[hint_template][steering_direction]['correct'] += 1

                        # 3. Update Specific Stats (Layer 15, Target 1)
                        # Check if steering_target_value is 1 (handle int/float/string)
                        try:
                            target_val = float(steering_target_value)
                        except (ValueError, TypeError):
                            target_val = None

                        if steering_layer == 15 and target_val == 1.0:
                             specific_stats[hint_template][steering_direction]['total'] += 1
                             if is_correct:
                                 specific_stats[hint_template][steering_direction]['correct'] += 1

                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Output Results
    output = []
    
    # 3. Specific Analysis
    output.append("=== Specific Analysis (Layer 15, Target 1) ===\n")
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
    
    # 2. Total Stratified Analysis
    output.append("=== Total Analysis (All Layers, All Targets) - Stratified by Hint Template ===\n")
    output.append(f"{'Hint Template':<30} | {'Direction':<15} | {'Total':<10} | {'Correct':<10} | {'Accuracy':<10}")
    output.append("-" * 85)
    
    for template in sorted(total_stratified_stats.keys()):
        for direction in sorted(total_stratified_stats[template].keys()):
            stats = total_stratified_stats[template][direction]
            total = stats['total']
            correct = stats['correct']
            accuracy = (correct / total * 100) if total > 0 else 0.0
            output.append(f"{template:<30} | {direction:<15} | {total:<10} | {correct:<10} | {accuracy:.2f}%")

    output.append("\n" + "="*85 + "\n")

    # 1. Total Analysis
    output.append("=== Total Analysis (All Layers, All Targets) - Aggregated ===\n")
    output.append(f"{'Direction':<15} | {'Total':<10} | {'Correct':<10} | {'Accuracy':<10}")
    output.append("-" * 55)
    
    for direction in sorted(total_stats.keys()):
        stats = total_stats[direction]
        total = stats['total']
        correct = stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0.0
        output.append(f"{direction:<15} | {total:<10} | {correct:<10} | {accuracy:.2f}%")

    # Write to file
    with open('gradient_steered_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print("Analysis complete. Results written to gradient_steered_analysis_results.txt")
    print('\n'.join(output))

if __name__ == "__main__":
    file_path = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_5_2025-11-15\steered\steered_gradient_val_scie_hist_psy_X_grader_prof_meta_2025-12-01.jsonl"
    analyze_gradient_steered_accuracy(file_path)
