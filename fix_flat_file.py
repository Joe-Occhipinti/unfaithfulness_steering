"""
Fix flat file by extracting generated text from sampled_prompt.
The original generation script saved full prompt in sampled_prompt but left generated_text as None.
"""

from src.data import load_jsonl, save_jsonl

# Load original nested file
INPUT_FILE = "data/Lorenz_suggestion_2025-11-12/sampled_q13-36-115-26_2025-11-12.jsonl"
OUTPUT_FILE = "data/Lorenz_suggestion_2025-11-12/sampled_q13-36-115-26_2025-11-12_flat.jsonl"

print("Loading nested file...")
nested_data = load_jsonl(INPUT_FILE)
print(f"Loaded {len(nested_data)} nested records")

flat_records = []

for record in nested_data:
    # Extract base fields (common to all samples)
    base_fields = {
        'question_id': record.get('question_id'),
        'question': record.get('question'),
        'subject': record.get('subject'),
        'choices': record.get('choices'),
        'answer': record.get('answer'),
        'ground_truth_letter': record.get('ground_truth_letter'),
        'hint_letter': record.get('hint_letter'),
        'hint_template': record.get('hint_template'),
        'biased_input_prompt': record.get('biased_input_prompt'),
        'baseline_answer_letter': record.get('baseline_answer_letter'),
        'biased_answer_letter': record.get('biased_answer_letter'),
        'faithfulness_classification': record.get('faithfulness_classification'),
        # Add metadata about sampling
        'temperature': record.get('metadata', {}).get('temperature'),
        'model': record.get('metadata', {}).get('model'),
        'date': record.get('metadata', {}).get('date')
    }

    biased_input_prompt = base_fields['biased_input_prompt']

    # Flatten samples
    for sample in record.get('samples', []):
        sampled_prompt = sample.get('sampled_prompt', '')

        # Extract generated text by removing the biased_input_prompt prefix
        if sampled_prompt and biased_input_prompt:
            if sampled_prompt.startswith(biased_input_prompt):
                generated_text = sampled_prompt[len(biased_input_prompt):]
            else:
                # Fallback: use the full sampled_prompt if it doesn't start with the prompt
                generated_text = sampled_prompt
        else:
            generated_text = sampled_prompt

        flat_record = {
            **base_fields,
            'sample_id': sample.get('sample_id'),
            'sampled_generated_text': generated_text,
            'sampled_prompt': sampled_prompt
        }
        flat_records.append(flat_record)

# Save flattened output
save_jsonl(flat_records, OUTPUT_FILE)
print(f"Saved {len(flat_records)} flat records to: {OUTPUT_FILE}")

# Verify fix
print("\nVerifying fix...")
verification = load_jsonl(OUTPUT_FILE)
null_count = sum(1 for r in verification if r.get('sampled_generated_text') is None or r.get('sampled_generated_text') == '')
print(f"Records with null/empty generated text: {null_count}/{len(verification)}")
if verification:
    sample_text = verification[0].get('sampled_generated_text', '')
    print(f"Sample generated text preview: {sample_text[:200]}...")
