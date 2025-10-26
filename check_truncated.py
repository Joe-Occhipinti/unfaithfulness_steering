import json
import re

file_path = 'data/sprint4_2025-10-21/annotated/annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl'

period_before_tag_pattern = r'\.\s*</think>|\.\s*</reasoning>|\.\s*</answer>'

truncated_marked_complete = []
truncated_with_biased_answer = []

with open(file_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        generated_text = data.get('hinted_generated_text', '')
        has_period_before_tag = bool(re.search(period_before_tag_pattern, generated_text))
        completeness = data.get('completeness', 'N/A')
        biased_answer = data.get('biased_answer_letter', '')
        hint_letter = data.get('hint_letter', '')

        if not has_period_before_tag and completeness == 'complete':
            truncated_marked_complete.append({
                'idx': idx,
                'answer': biased_answer,
                'hint': hint_letter,
                'has_answer': bool(biased_answer)
            })

        if not has_period_before_tag and biased_answer and hint_letter:
            if biased_answer != hint_letter:
                truncated_with_biased_answer.append({
                    'idx': idx,
                    'answer': biased_answer,
                    'hint': hint_letter,
                    'completeness': completeness
                })

print(f'Total prompts: {idx + 1}')
print(f'Prompts without period before tag marked as complete: {len(truncated_marked_complete)}')
print(f'Prompts without period before tag with biased answer: {len(truncated_with_biased_answer)}')

if truncated_marked_complete:
    print('\n--- First 10 truncated prompts marked complete ---')
    for item in truncated_marked_complete[:10]:
        print(f"  Line {item['idx']}: answer={item['answer']}, hint={item['hint']}, has_answer={item['has_answer']}")

if truncated_with_biased_answer:
    print(f'\n--- First 10 truncated prompts with biased answers ---')
    for item in truncated_with_biased_answer[:10]:
        print(f"  Line {item['idx']}: answer={item['answer']}, hint={item['hint']}, completeness={item['completeness']}")
