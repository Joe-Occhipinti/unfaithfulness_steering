import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

def load_model(model_name: str = "Qwen/Qwen2.5-Math-1.5B"):
    """
    Load tokenizer and model from Hugging Face.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (tokenizer, model)
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    return tokenizer, model

def generate_text(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int = 100):
    """
    Generate text using the loaded model.
    
    Args:
        tokenizer: Loaded tokenizer
        model: Loaded model
        messages: List of message dictionaries with 'role' and 'content' keys
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        str: Generated text
    """
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (skip the input)
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], 
        skip_special_tokens=True
    )
    
    return generated_text

def main():
    """
    Simple demo of the text generation functionality.
    """
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    
    # Load model
    try:
        tokenizer, model = load_model(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Demo with your original example
    print("\n=== Demo Mode ===")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    
    try:
        response = generate_text(tokenizer, model, messages, max_new_tokens=40)
        print(f"Question: {messages[0]['content']}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\nModel is ready for use in other scripts via import")

if __name__ == "__main__":
    main()