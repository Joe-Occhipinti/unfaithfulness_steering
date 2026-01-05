
import json
import sys
from pathlib import Path

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

def count_tokens_tiktoken(text):
    if not HAS_TIKTOKEN:
        return None
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as e:
        print(f"Tiktoken error: {e}")
        return None

def count_tokens_transformers(text, model_name="Qwen/Qwen2.5-32B-Instruct"):
    if not HAS_TRANSFORMERS:
        return None
    try:
        # Try to load a tokenizer. This might fail if not logged in or model doesn't exist locally/remotely
        # We'll try a generic one if specific fails, or just skip
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return len(tokenizer.encode(text))
    except Exception as e:
        print(f"Transformers tokenizer error for {model_name}: {e}")
        return None

def main():
    file_path = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\definitive_pipeline_data\Qwen3-32B\steered_linear_Qwen3-32B_2026-01-04.jsonl"
    
    print(f"Reading {file_path}...")
    
    longest_prompt = ""
    longest_prompt_len = 0
    record_idx = -1
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    prompt = data.get('steered_prompt', '')
                    if len(prompt) > longest_prompt_len:
                        longest_prompt_len = len(prompt)
                        longest_prompt = prompt
                        record_idx = i
                except json.JSONDecodeError:
                    print(f"Error decoding line {i}")
                    continue
    except FileNotFoundError:
        print("File not found.")
        return

    with open("analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Longest prompt found at index {record_idx}\n")
        f.write(f"Character length: {longest_prompt_len}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Content snippet: {longest_prompt[:100]}...{longest_prompt[-100:]}\n")
        f.write("-" * 40 + "\n")
        
        # Token counting
        f.write("\nToken Count Estimates:\n")
        
        # 1. Simple estimation
        f.write(f"Simple estimate (chars / 4): {longest_prompt_len / 4:.1f}\n")
        
        # 2. Tiktoken
        if HAS_TIKTOKEN:
            tik_count = count_tokens_tiktoken(longest_prompt)
            f.write(f"Tiktoken (cl100k_base): {tik_count}\n")
        else:
            f.write("Tiktoken not installed.\n")

        # 3. Transformers
        if HAS_TRANSFORMERS:
            # Try Qwen tokenizer if possible, otherwise maybe gpt2
            f.write("Attempting to use transformers tokenizer (Qwen/Qwen2.5-32B-Instruct)...\n")
            trans_count = count_tokens_transformers(longest_prompt, "Qwen/Qwen2.5-32B-Instruct")
            if trans_count:
                 f.write(f"Transformers (Qwen2.5-32B): {trans_count}\n")
            else:
                 f.write("Could not load Qwen tokenizer, trying gpt2...\n")
                 trans_count = count_tokens_transformers(longest_prompt, "gpt2")
                 f.write(f"Transformers (gpt2): {trans_count}\n")
        else:
            f.write("Transformers not installed.\n")
            
        if longest_prompt_len > 0:
            is_under_512 = False
            # Use the best available metric
            if HAS_TRANSFORMERS and trans_count:
                is_under_512 = trans_count < 512
                f.write(f"\nVerdict (using Transformers): Under 512 tokens? {is_under_512} ({trans_count})\n")
            elif HAS_TIKTOKEN and tik_count:
                is_under_512 = tik_count < 512
                f.write(f"\nVerdict (using Tiktoken): Under 512 tokens? {is_under_512} ({tik_count})\n")
            else:
                est = longest_prompt_len / 4
                is_under_512 = est < 512
                f.write(f"\nVerdict (using char/4 estimate): Under 512 tokens? {is_under_512} (approx {est})\n")
    print("Analysis written to analysis_report.txt")

if __name__ == "__main__":
    main()
