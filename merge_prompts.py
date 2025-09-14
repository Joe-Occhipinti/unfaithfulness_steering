import json
import os

def merge_jsonl_fields(input_file, output_file):
    """
    Merge original_prompt and generated_biased_answer fields into biased_prompt_train,
    and extract relevant fields for the output JSONL.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            # Merge original_prompt and generated_biased_answer
            original_prompt = data.get('original_prompt', '')
            generated_biased_answer = data.get('generated_biased_answer', '')
            biased_prompt_train = original_prompt + generated_biased_answer
            
            # Create output record with required fields
            output_record = {
                'biased_prompt_train': biased_prompt_train,
                'faithfulness_classification': data.get('faithfulness_classification', {}).get('faithfulness', ''),
                'unbiased_answer': data.get('extracted_letter', ''),
                'hinted_answer': data.get('biased_hint_letter', ''),
                'biased_answer': data.get('extracted_biased_letter', '')
            }
            
            # Write to output file
            outfile.write(json.dumps(output_record) + '\n')

if __name__ == "__main__":
    input_file = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\datasets\activation_prompts_mmlu_psychology_train_2025-08-12.jsonl"
    output_file = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\datasets\merged_biased_prompts.jsonl"
    
    merge_jsonl_fields(input_file, output_file)
    print(f"Successfully merged fields and created {output_file}")