
import json
import sys
from pathlib import Path

# Try to import tiktoken or transformers, else fallback
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

def count_tokens(text):
    # Priority: Tiktoken -> Transformers -> Char/4
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except:
            pass
            
    if HAS_TRANSFORMERS:
        try:
            # Fallback to a standard tokenizer if specific one fails
            tokenizer = AutoTokenizer.from_pretrained("gpt2") 
            return len(tokenizer.encode(text))
        except:
            pass
            
    # Fallback
    return len(text) / 4.0

def main():
    file_path = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\definitive_pipeline_data\Qwen3-32B\faithfulness_annotated_Qwen3-32B_2025-12-29.jsonl"
    target_field = "local_annotated_biased_prompt"
    filter_field = "split"
    filter_value = "val"
    
    print(f"Analyzing {file_path}")
    print(f"Target field: {target_field}")
    print(f"Filter: {filter_field} == '{filter_value}'")
    
    counts = {
        "under_512": 0,
        "under_1024": 0, 
        "over_1024": 0,
        "1024_to_1200": 0,
        "total": 0
    }
    
    bins = {
        "0-512": 0,
        "512-1024": 0,
        "1024+": 0
    }
    
    over_1024_lengths = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Apply filter
                    if data.get(filter_field) != filter_value:
                        continue
                        
                    if target_field in data:
                        prompt = data[target_field]
                        token_count = count_tokens(prompt)
                        
                        counts["total"] += 1
                        
                        if 1024 < token_count < 1200:
                            counts["1024_to_1200"] += 1
                        
                        if token_count < 512:
                            bins["0-512"] += 1
                            counts["under_512"] += 1
                            counts["under_1024"] += 1
                        elif token_count < 1024:
                            bins["512-1024"] += 1
                            counts["under_1024"] += 1
                        else:
                            bins["1024+"] += 1
                            counts["over_1024"] += 1
                            over_1024_lengths.append(token_count)
                            
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print("File not found.")
        return

    total = counts["total"]
    
    import statistics
    
    with open("val_split_analysis.txt", "w", encoding="utf-8") as f:
        f.write(f"Analysis for split='{filter_value}'\n")
        f.write(f"Total records matching filter: {total}\n")
        
        if total > 0:
            f.write("\nBreakdown:\n")
            f.write(f"Under 512 tokens: {counts['under_512']} ({counts['under_512']/total:.1%})\n")
            f.write(f"Under 1024 tokens (cumulative): {counts['under_1024']} ({counts['under_1024']/total:.1%})\n")
            f.write(f"Over 1024 tokens: {counts['over_1024']} ({counts['over_1024']/total:.1%})\n")
            f.write(f"Between 1024 and 1200 tokens: {counts['1024_to_1200']}\n")
            
            f.write("\nDetailed Bins:\n")
            f.write(f"0 - 512:    {bins['0-512']}\n")
            f.write(f"512 - 1024: {bins['512-1024']}\n")
            f.write(f"1024+:      {bins['1024+']}\n")
            
            if over_1024_lengths:
                avg_len = statistics.mean(over_1024_lengths)
                median_len = statistics.median(over_1024_lengths)
                f.write(f"\nStats for prompts > 1024 tokens:\n")
                f.write(f"Count: {len(over_1024_lengths)}\n")
                f.write(f"Average Length: {avg_len:.2f}\n")
                f.write(f"Median Length: {median_len:.2f}\n")
                f.write(f"Max Length: {max(over_1024_lengths)}\n")
        
        if not HAS_TIKTOKEN and not HAS_TRANSFORMERS:
            f.write("\nNote: Token counts are estimated using characters / 4.\n")
            
    print("Results written to val_split_analysis.txt")

if __name__ == "__main__":
    main()
