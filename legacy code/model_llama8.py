import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from typing import Optional, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # change as needed
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    use_auth_token: Optional[str] = None,
    max_memory: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Load reasoning model from Hugging Face with GPU optimization for vast.ai.
    
    Args:
        model_name (str): HuggingFace model identifier. Default: DeepSeek-R1-Distill-Llama-8B
        device (str, optional): Device to load model on. Auto-detects if None.
        torch_dtype (torch.dtype, optional): Data type for model weights. Default: bfloat16 if available
        load_in_8bit (bool): Use 8-bit quantization (requires bitsandbytes)
        load_in_4bit (bool): Use 4-bit quantization (requires bitsandbytes)
        trust_remote_code (bool): Whether to trust remote code in model
        use_auth_token (str, optional): HF auth token for private models
        max_memory (dict, optional): Max memory per GPU device
        cache_dir (str, optional): Cache directory for downloaded models
    
    Returns:
        Tuple[model, tokenizer]: Loaded model and tokenizer
    
    Raises:
        RuntimeError: If CUDA is not available when expected
        Exception: If model loading fails
    """
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA detected. Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            logger.warning("CUDA not available. Falling back to CPU (not recommended for 8B models)")
    
    # Set optimal dtype for the device
    if torch_dtype is None:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}, Dtype: {torch_dtype}")
    
    # Get auth token from environment if not provided
    if use_auth_token is None:
        use_auth_token = os.getenv("HF_TOKEN")
    
    # Set up model loading arguments
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    
    # Add auth token if available
    if use_auth_token:
        model_kwargs["token"] = use_auth_token
        logger.info("Using HuggingFace authentication token")
    
    # Add cache directory if specified
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    
    # Configure quantization
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using 8-bit quantization")
            
            model_kwargs["quantization_config"] = quantization_config
        except ImportError:
            logger.error("bitsandbytes not installed. Install with: pip install bitsandbytes")
            raise
    
    # Configure device mapping for multi-GPU
    if device == "cuda":
        if max_memory is None and torch.cuda.device_count() > 1:
            # Auto-distribute across available GPUs
            model_kwargs["device_map"] = "auto"
            logger.info(f"Using auto device mapping across {torch.cuda.device_count()} GPUs")
        elif max_memory:
            model_kwargs["max_memory"] = max_memory
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": 0}  # Use first GPU
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if use_auth_token:
            tokenizer_kwargs["token"] = use_auth_token
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info("Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs and device != "cpu":
            model = model.to(device)
        
        # Set model to eval mode
        model.eval()
        
        # Print model info
        if hasattr(model, 'num_parameters'):
            param_count = model.num_parameters()
        else:
            param_count = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Parameters: {param_count / 1e9:.2f}B")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32000,
    temperature: float = 0.0,
    do_sample: bool = True,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> str:
    """
    Generate a response using the loaded model.
    
    Args:
        model: Loaded model instance
        tokenizer: Loaded tokenizer instance
        prompt (str): Input prompt
        max_new_tokens (int): Maximum tokens to generate
        temperature (float): Sampling temperature
        do_sample (bool): Whether to use sampling
        top_p (float): Nucleus sampling parameter
        repetition_penalty (float): Repetition penalty
    
    Returns:
        str: Generated response
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (excluding the input prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    # Example usage
    try:
        # Load model (adjust model_name as needed)
        model, tokenizer = load_model(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            load_in_4bit=True,  # Use 4-bit quantization to save memory
        )
        
        # Test with a reasoning prompt
        test_prompt = """User: What is 2 + 2? Think step-by-step, then provide the answer. Assistant: Let's think step by step:"""
        
        response = generate_response(model, tokenizer, test_prompt, max_new_tokens=100)
        print(f"Model response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")