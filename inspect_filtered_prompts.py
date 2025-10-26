"""
Inspect the last sentences from prompts that got filtered out.
Shows what text appears before the closing tags in truncated prompts.
"""

import json
import re

INPUT_FILE = "data/sprint4_2025-10-21/annotated/annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"

def has_period_before_closing_tag(text):
    """Check if text has a period before [/F_body] or [/U_body] closing tags."""
    pattern = r'\.\s*\[/[FU]_body\]'
    return bool(re.search(pattern, text))

def get_text_before_closing_tag(text, num_chars=200):
    """Extract the last N characters before the closing tag."""
    # Find the closing tag
    match = re.search(r'\[/[FU]_body\]', text)
    if match:
        # Get position of the tag
        tag_start = match.start()
        # Get text before the tag (last num_chars characters)
        start_pos = max(0, tag_start - num_chars)
        return text[start_pos:tag_start]
    return ""

# Collect filtered prompts
filtered_prompts = []

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        local_annotated_prompt = data.get('local_annotated_biased_prompt', '')

        # Check if it would be filtered (no period before tag)
        if not has_period_before_closing_tag(local_annotated_prompt):
            text_before_tag = get_text_before_closing_tag(local_annotated_prompt)
            filtered_prompts.append({
                'idx': idx,
                'text_before_tag': text_before_tag,
                'answer': data.get('biased_answer_letter', ''),
                'hint': data.get('hint_letter', '')
            })

            # Stop after collecting 50
            if len(filtered_prompts) >= 50:
                break

# Print results
print(f"=== LAST SENTENCES FROM FIRST {len(filtered_prompts)} FILTERED PROMPTS ===\n")

for i, prompt in enumerate(filtered_prompts, 1):
    print(f"--- Filtered Prompt {i} (Line {prompt['idx']}) ---")
    print(f"Answer: {prompt['answer']}, Hint: {prompt['hint']}")
    print(f"Text before closing tag:")
    print(f"{prompt['text_before_tag']}")
    print()
