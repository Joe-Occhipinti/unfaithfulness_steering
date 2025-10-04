"""
Check if the faithful prompt has proper closing [/F] tags
"""

import re
from src.data import load_jsonl

ANNOTATED_FILE = "data/annotated/annotated_biased_high_school_macroeconomics_microeconomics_2025-10-03.jsonl"

print(f"Loading {ANNOTATED_FILE}...")
annotated = load_jsonl(ANNOTATED_FILE)

# Find faithful prompt
faithful = [r for r in annotated if r.get('faithfulness_classification') == 'faithful'][0]

text = faithful.get('annotated_biased_prompt', '')

# Count tags
f_open = re.findall(r'\[F\]', text)
f_close = re.findall(r'\[/F\]', text)

print(f"\nTag counts:")
print(f"  [F] opening tags: {len(f_open)}")
print(f"  [/F] closing tags: {len(f_close)}")
print(f"\nTags match: {len(f_open) == len(f_close)}")

if len(f_open) > 0:
    # Find snippet around [F] tag
    idx = text.find('[F]')
    start = max(0, idx - 50)
    end = min(len(text), idx + 300)
    snippet = text[start:end]

    print(f"\nSnippet around [F] tag:")
    print(snippet)

    # Check if there's a closing tag in the snippet
    has_close_in_snippet = '[/F]' in snippet
    print(f"\nHas [/F] in snippet: {has_close_in_snippet}")
else:
    print("\nNo [F] tags found!")
