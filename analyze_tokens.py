
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
    
    print(f"Analyzing {file_path}...")
    print(f"Target field: {target_field}")
    
    counts = {
        "under_512": 0,
        "under_1024": 0, # Exclusive of under_512? User asked for "under 1024", usually implies cumulative. I'll report both.
        "over_1024": 0,
        "total": 0
    }
    
    # We will track:
    # < 512
    # 512 <= x < 1024
    # >= 1024
    
    bins = {
        "0-512": 0,
        "512-1024": 0,
        "1024+": 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if target_field in data:
                        prompt = data[target_field]
                        token_count = count_tokens(prompt)
                        
                        counts["total"] += 1
                        
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
                            
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print("File not found.")
        return

    with open("token_analysis_results_utf8.txt", "w", encoding="utf-8") as f:
        f.write(f"Analyzing {file_path}\n")
        f.write(f"Target field: {target_field}\n")
        
        total = counts["total"]
        if total == 0:
            f.write("No records found.\n")
            return

        f.write("\nResults:\n")
        f.write(f"Total records: {total}\n")
        
        f.write("\nBreakdown:\n")
        f.write(f"Under 512 tokens: {counts['under_512']} ({counts['under_512']/total:.1%})\n")
        f.write(f"Under 1024 tokens (cumulative): {counts['under_1024']} ({counts['under_1024']/total:.1%})\n")
        f.write(f"Over 1024 tokens: {counts['over_1024']} ({counts['over_1024']/total:.1%})\n")
        
        f.write("\nDetailed Bins:\n")
        f.write(f"0 - 512:    {bins['0-512']}\n")
        f.write(f"512 - 1024: {bins['512-1024']}\n")
        f.write(f"1024+:      {bins['1024+']}\n")
        
        if not HAS_TIKTOKEN and not HAS_TRANSFORMERS:
            f.write("\nNote: Token counts are estimated using characters / 4.\n")
    print("Results written to token_analysis_results_utf8.txt")

if __name__ == "__main__":
    main()
