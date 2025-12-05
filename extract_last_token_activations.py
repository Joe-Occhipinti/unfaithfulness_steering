import argparse
import json
import os
import torch
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_and_concatenate_prompts(jsonl_file):
    """
    Loads prompts from a JSONL file and concatenates biased_input_prompt and off_policy_response.
    Returns a list of dictionaries with 'text' and original metadata.
    """
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                biased_prompt = record.get('biased_input_prompt', '')
                response = record.get('off_policy_response', '')
                
                if not biased_prompt or not response:
                    continue
                    
                # Concatenate with a newline
                full_text = f"{biased_prompt}\n{response}"
                
                # Only keep 'label' in metadata as requested
                metadata = {'label': record.get('label', 'unknown')}
                
                item = {
                    'text': full_text,
                    'metadata': metadata
                }
                data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping malformed line in {jsonl_file}")
    return data

def extract_activations(args):
    """
    Main extraction loop.
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    # Load data
    print(f"Loading data from: {args.input_file}")
    all_data = load_and_concatenate_prompts(args.input_file)
    
    # Slice data if start/end indices provided
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(all_data)
    data_to_process = all_data[start_idx:end_idx]
    
    print(f"Processing {len(data_to_process)} prompts (indices {start_idx} to {end_idx})")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Processing loop
    for i, item in enumerate(tqdm(data_to_process)):
        global_idx = start_idx + i
        text = item['text']
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract activations
        # outputs.hidden_states is a tuple of (layer_0, layer_1, ..., layer_N)
        # Each layer tensor has shape (batch_size, seq_len, hidden_dim)
        # We want the last token of the first sequence in the batch: [0, -1, :]
        
        prompt_activations = {}
        for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
            # Extract last token activation
            last_token_activation = layer_tensor[0, -1, :].cpu()
            
            # Store in structure compatible with analysis scripts
            # Using "last_token" as the tag
            prompt_activations[layer_idx] = {
                "last_token": last_token_activation
            }
            
        # Save individual file
        save_path = os.path.join(args.output_dir, f"prompt_{global_idx}_activations.pt")
        torch.save(prompt_activations, save_path)

    print("Extraction complete.")

def build_dataset(args):
    """
    Consolidates individual activation files into a single dataset with metadata.
    """
    print("Building consolidated dataset...")
    
    # Load original data to get metadata mapping
    all_data = load_and_concatenate_prompts(args.input_file)
    
    dataset = {
        "data": {},
        "info": {
            "model": args.model_id,
            "source_file": args.input_file,
            "tags": ["last_token"]
        }
    }
    
    # Find all activation files
    activation_files = [f for f in os.listdir(args.output_dir) if f.endswith('_activations.pt')]
    
    count = 0
    for filename in tqdm(activation_files, desc="Consolidating"):
        try:
            # Parse index from filename "prompt_{i}_activations.pt"
            parts = filename.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1])
            else:
                continue
                
            # Load activations
            file_path = os.path.join(args.output_dir, filename)
            activations = torch.load(file_path, map_location='cpu')
            
            # Get metadata
            if 0 <= idx < len(all_data):
                metadata = all_data[idx]['metadata']
            else:
                print(f"Warning: Index {idx} out of bounds for input file")
                metadata = {}
            
            # Add to dataset
            dataset["data"][idx] = {
                "metadata": metadata,
                "layers": activations
            }
            count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save consolidated dataset
    output_file = os.path.join(args.output_dir, "activations_dataset.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"Saved dataset with {count} entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract last token activations from a model.")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Hugging Face model ID")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end_index", type=int, default=None, help="End index for processing")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip extraction and only build dataset")
    
    args = parser.parse_args()
    
    if not args.skip_extraction:
        extract_activations(args)
        
    build_dataset(args)

if __name__ == "__main__":
    main()
